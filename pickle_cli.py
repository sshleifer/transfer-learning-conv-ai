from train import train
from durbango import *
import fire

def train_args(args_path, local_rank=-1):
    args = pickle_load(args_path)
    args.local_rank = local_rank
    train(args)

if __name__ == '__main__': fire.Fire(train_args)
