# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.
import os
import math
import logging
from pprint import pformat
from argparse import ArgumentParser
from collections import defaultdict
from itertools import chain

from  pathlib import Path
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
from pytorch_pretrained_bert import (OpenAIAdam, OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer,
                                     GPT2DoubleHeadsModel, GPT2Tokenizer, WEIGHTS_NAME, CONFIG_NAME)

import random
from durbango import *

from ignite.contrib.handlers import CustomPeriodicEvent
from ignite.contrib.handlers.tensorboard_logger import *


MSG_USER = 'xMsg_User'  # user message begin tag / begin of individual message
MSG_DR = 'xMsg_Dr'  # Dr message begin tag / begin of individual message
USER_NAME = 'xUser_Name'
DR_NAME = 'xDr_Name'
FOLLOWUP_MSG = 'xFollowup_Msg'
DR_STRANG = 'Dr'

FO_SPECIAL_TOKENS = [USER_NAME, DR_NAME, FOLLOWUP_MSG]

SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]

MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids"]
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]
logger = logging.getLogger(__file__)


def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()


def pad_dataset(dataset, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and pad only batches but this is simpler. """
    max_l = max(len(x) for x in dataset["input_ids"])
    print(f'padding to {max_l}')
    for name in PADDED_INPUTS:
        if name not in dataset: continue
        dataset[name] = [x + [padding if name != "lm_labels" else -1] * (max_l - len(x))
                         for x in dataset[name]]
    return dataset


def lchain(seq):
    return list(chain(*seq))


def build_input_from_segments(persona, history, reply, tokenizer, lm_labels=False, with_eos=True, max_seq_len=512):
    """ Build a sequence of input from 3 segments: persona, history and last reply """
    instance, sequence = get_input_ids(history, persona, reply, tokenizer, with_eos, max_seq_len)
    instance["mc_token_ids"] = len(instance["input_ids"]) - 1
    instance["lm_labels"] = [-1] * len(instance["input_ids"])
    if lm_labels:
        instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]
    return instance, sequence


def get_input_ids(history, persona, reply, tokenizer, with_eos, max_seq_len):
    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    instance = {}
    start = [[bos] + list(chain(*persona))]
    end = [reply + ([eos] if with_eos else [])]
    n_speaker_toks = len(history) + len(reply)
    extra_toks = (max_seq_len - len(lchain(start)) - len(lchain(end)) - n_speaker_toks)
    while len(lchain(history)) > extra_toks:
        history = history[1:]
    sequence = start + history + end
    hist_part = [[speaker2 if (len(sequence) - i) % 2 else speaker1] + s for i, s in
                 enumerate(sequence[1:])]
    sequence = [sequence[0]] + hist_part
    if len(lchain(sequence)) > 512: raise ValueError(f'sequence longer than 512: {sequence}')
    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence)
                                  for _ in s]
    return instance, sequence


def get_data_loaders(personachat, args, tokenizer):
    """ Prepare the dataset for training and evaluation """

    logger.info("Build inputs and labels")
    datasets = sample_from_ds(args, personachat, tokenizer)

    logger.info("Pad inputs and convert to Tensor")
    tensor_datasets = {"train": [], "valid": []}
    for dataset_name, dataset in datasets.items():
        dataset = pad_dataset(dataset, padding=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1]))
        for input_name in MODEL_INPUTS:
            if input_name not in dataset: continue
            tensor = torch.tensor(dataset[input_name])
            if input_name != "mc_labels":
                print(f'{input_name}: {tensor.shape}')
                tensor = tensor.view((-1, datasets[dataset_name]["n_candidates"]) + tensor.shape[1:])
                print(tensor.shape)
            tensor_datasets[dataset_name].append(tensor)

    logger.info("Build train and validation dataloaders")
    train_dataset, valid_dataset = TensorDataset(*tensor_datasets["train"]), TensorDataset(*tensor_datasets["valid"])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, shuffle=(not args.distributed))
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.valid_batch_size, shuffle=False)

    logger.info("Train dataset (Batch, Candidates, Seq length): {}".format(train_dataset.tensors[0].shape))
    logger.info("Valid dataset (Batch, Candidates, Seq length): {}".format(valid_dataset.tensors[0].shape))
    return train_loader, valid_loader, train_sampler, valid_sampler

def sample_from_ds(args, personachat, tokenizer):
    datasets = {"train": defaultdict(list), "valid": defaultdict(list)}
    for dataset_name, dataset in personachat.items():
        num_candidates = len(dataset[0]["utterances"][0]["candidates"])
        if args.num_candidates > 0 and dataset_name == 'train':
            num_candidates = min(args.num_candidates, num_candidates)
        for dialog in dataset:
            persona = dialog["personality"].copy()
            for _ in range(args.personality_permutations):
                for utterance in dialog["utterances"]:
                    history = utterance["history"][-(2 * args.max_history + 1):]
                    pos_candidate = [utterance["candidates"][-1]]
                    neg_candidates = random.choices(utterance["candidates"][:-1], k=num_candidates - 1)
                    candidates = neg_candidates + pos_candidate
                    for j, candidate in enumerate(candidates):
                        lm_labels = bool(j == num_candidates - 1)  # Last candidate is correct
                        persona_to_use = [] if args.ignore_persona else persona
                        instance, _ = build_input_from_segments(persona_to_use, history, candidate,
                                                                tokenizer, lm_labels)
                        for input_name, input_array in instance.items():
                            datasets[dataset_name][input_name].append(input_array)
                    datasets[dataset_name]["mc_labels"].append(num_candidates - 1)
                    datasets[dataset_name]["n_candidates"] = num_candidates
                persona = [persona[-1]] + persona[:-1]  # permuted personalities
    return datasets


def make_ctx_dl(args, history, persona, tokenizer, with_eos=True, max_seq_len=512):
    ds = defaultdict(list)
    for h in history:
        instance, sequence = get_input_ids(persona, h, [], tokenizer, with_eos=with_eos,max_seq_len=max_seq_len)
        for input_name, input_array in instance.items():
            ds[input_name].append(input_array)
    padded_ds = pad_dataset(ds, padding=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1]))
    logger.info("Pad inputs and convert to Tensor")
    tensor_dataset = []

    for input_name in ['input_ids', 'position_ids']:
        if input_name not in padded_ds: continue
        tensor = torch.tensor(padded_ds[input_name])
        if input_name != "mc_labels":
            tensor = tensor.view((-1, 1) + tensor.shape[1:])
        tensor_dataset.append(tensor)
    tensor_ds = TensorDataset(*tensor_dataset)
    loader = DataLoader(tensor_ds, batch_size=args.train_batch_size, shuffle=False)
    return loader


def train(args):
    args.save_dir = Path(args.save_dir)
    args.save_dir.mkdir(exist_ok=True)
    log_dir = str(args.save_dir)
    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process. logger.info => log main process only, logger.warning => log all processes
    logging.basicConfig(filename=args.save_dir/'logs.log', level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN, format='%(asctime)s %(message)s')
    logger.warning("Running process %d", args.local_rank)  # This is a logger.warning: it will be printed by all distributed processes
    logger.info("Arguments: %s", pformat(args))

    args.distributed = (args.local_rank != -1)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    pickle_save(args, 'args.pkl')
    logger.info("Prepare tokenizer, pretrained model and optimizer - add special tokens for fine-tuning")
    tokenizer_class = GPT2Tokenizer if "gpt2" in args.model_checkpoint else OpenAIGPTTokenizer
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    model_class = GPT2DoubleHeadsModel if "gpt2" in args.model_checkpoint else OpenAIGPTDoubleHeadsModel
    model = model_class.from_pretrained(args.model_checkpoint)
    if args.ckpt_path:
        state_dict = torch.load(args.ckpt_path)
        extra_toks = state_dict['transformer.tokens_embed.weight'].shape[0] - model.transformer.tokens_embed.weight.shape[0]
        model.set_num_special_tokens(extra_toks)
        model.load_state_dict(state_dict)


    tokenizer.set_special_tokens(SPECIAL_TOKENS)
    model.set_num_special_tokens(len(SPECIAL_TOKENS))
    model.to(args.device)
    optimizer = OpenAIAdam(model.parameters(), lr=args.lr)

    # Prepare model for FP16 and distributed training if needed (order is important, distributed should be the last)
    if args.fp16:
        from apex import amp  # Apex is only required if we use fp16 training
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    logger.info("Prepare datasets")
    personachat = pickle_load(args.dataset_path)
    #personachat = get_dataset(tokenizer, args.dataset_path, args.dataset_cache)
    train_loader, val_loader, train_sampler, valid_sampler = get_data_loaders(personachat, args, tokenizer)

    # Training function and trainer
    def update(engine, batch):
        model.train()
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        lm_loss, mc_loss = model(*batch)
        loss = (lm_loss * args.lm_coef + mc_loss * args.mc_coef) / args.gradient_accumulation_steps
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()
    trainer = Engine(update)


    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
            input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = batch
            # logger.info(tokenizer.decode(input_ids[0, -1, :].tolist()))
            model_outputs = model(input_ids, mc_token_ids, token_type_ids=token_type_ids)
            lm_logits, mc_logits = model_outputs[0], model_outputs[1]  # So we can also use GPT2 outputs
            lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
            lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
            return (lm_logits_flat_shifted, mc_logits), (lm_labels_flat_shifted, mc_labels)
    evaluator = Engine(inference)

    if args.eval_every:
        cpe = CustomPeriodicEvent(n_iterations=args.eval_every)
        cpe.attach(trainer)
        evaluate_event = cpe._periodic_event_completed
    else:
        evaluate_event = Events.EPOCH_COMPLETED
    # Attach evaluation to trainer: we evaluate when we start the eraining and at the end of each epoch
    run_eval = lambda _: evaluator.run(val_loader)
    trainer.add_event_handler(evaluate_event, run_eval)
    if args.n_epochs < 1:
        trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(val_loader))
    if args.eval_before_start:
        trainer.add_event_handler(Events.STARTED, lambda _: evaluator.run(val_loader))

    # Make sure distributed data samplers split the dataset nicely between the distributed processes
    if args.distributed:
        trainer.add_event_handler(Events.EPOCH_STARTED, lambda engine: train_sampler.set_epoch(engine.state.epoch))
        evaluator.add_event_handler(Events.EPOCH_STARTED, lambda engine: valid_sampler.set_epoch(engine.state.epoch))

    # Linearly decrease the learning rate from lr to zero
    scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Prepare metrics - note how we compute distributed metrics
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    metrics = {"nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-1), output_transform=lambda x: (x[0][0], x[1][0])),
               "accuracy": Accuracy(output_transform=lambda x: (x[0][1], x[1][1]))}
    metrics.update({"average_nll": MetricsLambda(average_distributed_scalar, metrics["nll"], args),
                    "average_accuracy": MetricsLambda(average_distributed_scalar, metrics["accuracy"], args)})
    metrics["average_ppl"] = MetricsLambda(math.exp, metrics["average_nll"])
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    # On the main process: add progress bar, tensorboard, checkpoints and save model, configuration and tokenizer before we start to train

    if args.local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=["loss"])
        evaluator.add_event_handler(Events.COMPLETED, lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))

        tb_logger = TensorboardLogger(log_dir=log_dir)

        def global_step_transform(*args, **kwargs): return trainer.state.iteration
        gst = global_step_transform if args.eval_every else None

        tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]), event_name=Events.ITERATION_COMPLETED)
        tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
        tb_logger.attach(evaluator,
                     log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys()), another_engine=trainer, global_step_transform=gst),
                     event_name=Events.EPOCH_COMPLETED)


        checkpoint_handler = ModelCheckpoint(log_dir, 'checkpoint', save_interval=1, n_saved=3)
        trainer.add_event_handler(evaluate_event, checkpoint_handler, {'mymodel': getattr(model, 'module', model)})  # "getattr" take care of distributed encapsulation

        torch.save(args, log_dir + '/model_training_args.bin')
        getattr(model, 'module', model).config.to_json_file(os.path.join(log_dir, CONFIG_NAME))
        tokenizer.save_vocabulary(log_dir)

    # Run the training
    trainer.run(train_loader, max_epochs=args.n_epochs)

    # On the main process: close tensorboard logger and rename the last checkpoint (for easy re-loading with OpenAIGPTModel.from_pretrained method)
    if args.local_rank in [-1, 0] and args.n_epochs > 0:
        os.rename(checkpoint_handler._saved[-1][1][-1], os.path.join(log_dir, WEIGHTS_NAME))  # TODO: PR in ignite to have better access to saved file paths (cleaner)
        tb_logger.close()






if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--dataset_path", type=str, default="tokenized_and_split_v2.pkl", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model_checkpoint", type=str, default="openai-gpt", help="Path, url or short name of the model")
    parser.add_argument("--num_candidates", type=int, default=2, help="Number of candidates for training")
    parser.add_argument("--max_history", type=int, default=2, help="Number of previous exchanges to keep in history")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=4, help="Batch size for validation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--lm_coef", type=float, default=1.0, help="LM loss coefficient")
    parser.add_argument("--mc_coef", type=float, default=1.0, help="Multiple-choice loss coefficient")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--personality_permutations", type=int, default=1, help="Number of permutations of personality sentences")
    parser.add_argument("--eval_before_start", action='store_true', help="If true start with a first evaluation before training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--fp16", type=str, default="", help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument("--ignore_persona", action='store_true')

    # Additions
    parser.add_argument('save_dir', type=str)
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--eval_every', type=int, default=None)


    args = parser.parse_args()

    #import ipdb; ipdb.set_trace()

    #args = pickle_load(base_args.args_path)
    #args.local_rank = base_args.local_rank
    train(args)
