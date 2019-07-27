clus_map  = t5.groupby('cluster').first().text
import re
def whitespace_around_punc(s):
    s = re.sub('([.,!?()])', r' \1 ', s)
    s = re.sub('\s{2,}', ' ', s)
    return s.lower()

PERSONALITY = 'personality'
HISTORY = 'history'
CANDIDATES  = 'candidates'
UTTERANCES = 'utterances'

def extra_preprocess(strang):
    return whitespace_around_punc(' '.join(strang.split(' ')[1:])).lower()
def get_candidates(target, clus_map, n_distractors=5):
    return pd.concat([clus_map.drop(target).sample(n_distractors), clus_map.loc[[target]]], sort=False).values
def make_features(t5, clus_map, dr_personality=DR_PERSONALITY):
    all_features = []
    for id, grp in tqdm_nice(t5.groupby('id'), desc='making context+response df'):
        f_for_convo = convo_to_context_df(grp.reset_index(), clean_response=False, n_turns=None, add_features=False)
        utterances = []
        for f in f_for_convo:
            history = lmap(drop_first_word_and_rejoin, sent_split(f.context))
            candidates = get_candidates(f.target, clus_map)
            utterances.append({HISTORY: history, CANDIDATES: candidates})
        entry = {PERSONALITY: dr_personality, UTTERANCES: utterances}
        all_features.append(entry)
    return all_features
