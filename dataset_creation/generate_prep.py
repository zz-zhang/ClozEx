from datasets import load_dataset
from tqdm import tqdm
import json
import stanza
from collections import Counter
import random
import time
import copy
# from lm_scorer.models.auto import AutoLMScorer as LMScorer
import torch
from dependency_parse import parse_sentences

from utils import prepTagger, BertScorer, GECor

TARGET_PREP = ["across", "to", "on", "with", "in", "of", "at", "inside", "during", "from", "as", "through", "for", "along", "like", "about", "into", "towards", "down", "behind", "round", "before", "by", "against", "between", "onto", "off", "beside", "around", "over", "among", "above", "after", "beneath"]
NUM_LIMIT = 2000
template = 'According the meaning of this sentence the option {0} is suitable, which means "{1}"'
nlp = stanza.Pipeline('en', processors='tokenize', use_gpu=True, model_dir='/home/zizheng/projects/exp-gen/coreNLP_models/stanford-corenlp-4.4.0')
START_INDEX = 5000


type_counter = Counter(TARGET_PREP)
START_TIME = time.time()



def dis_candiates_by_dictionary(dictionary, question, answer):
    def get_feature(parsing_tree, answer):
        def find_object(parsing_tree, index):
            for idx, item in enumerate(parsing_tree):
                word, parent_node, role, upos, xpos = item
                if parent_node - 1 == index:
                    return word.lower()
            return ''
        for idx, item in enumerate(parsing_tree):
            word, parent_node, role, upos, xpos = item
            if word.lower() == answer:
                if parent_node == '0':
                    parent_word = '[ROOT]'
                else:
                    parent_word = parsing_tree[parent_node - 1][0].lower()
                object_word = find_object(parsing_tree, idx)

                feature = f'{parent_word}[SEP]{object_word}'
                return feature
    answer = answer.lower()
    sentence = question.replace('_', answer)
    parsing_tree = parse_sentences(sentence, one_sent=True)
    feature = get_feature(parsing_tree, answer)
    if feature not in dictionary.keys() or len(dictionary[feature]) >= 4:
        return None
    candidates = [w for w in dictionary[feature] if w.lower() != answer]
    return candidates

def obtain_distractors(question, answer, scorer, corrector):
    # candidates = dis_candiates_by_dictionary(prep_feture_dict, question, answer)
    candidates = TARGET_PREP
    if candidates is None:
        return None
    dis_wi_scores = scorer.score_words(question, candidates)
    if dis_wi_scores is None:
        return None
    dis_wi_ge = []
    for dis, score in dis_wi_scores:
        if dis != answer:
            if corrector.word_cause_error(question.replace('_', dis), dis):
                dis_wi_ge.append((dis, score))
            if len(dis_wi_ge) == 3:
                return [dis[0] for dis in dis_wi_ge]
    return None

def high_prediction_confidence(tokens, prep, tagger, confidence=0.8):
    sentence = ' '.join(tokens)
    sentence = sentence.replace(prep, f'<head>{prep}</head>')
    sense, conf = tagger.tag(sentence, return_confidence=True)
    # print(prep)
    # print(sense)
    # print()
    # breakpoint()
    if conf >= confidence:
        return (sense, conf)
    return None

def generate_question(sentence, answer, sense, scorer, corrector):
    for idx, token in enumerate(sentence):
        if token == answer:
            sentence[idx] = '_'
            break
    question = ' '.join(sentence)
    # options = random.sample(TARGET_PREP, 4)
    # if answer not in options:
    #     options[-1] = answer
    options = obtain_distractors(question, answer, scorer, corrector)
    if options is None:
        return None
    # if answer not in options:
    #     options = options[:-1]
    options.append(answer)
    random.shuffle(options)

    answer_option = f'({chr(ord("A") + options.index(answer))}) {answer}'

    exp = template.format(answer_option, sense)
    # breakpoint()
    return {'question': question, 'answer': answer, 'options': options, 'exp': exp}

def build(dataset, key, tagger, scorer, corrector, out_f, init_length=0):
    print()
    text = dataset[key]
    res = []
    for idx, doc in enumerate(text[::-1]):
        doc = nlp(doc)
        for sent in doc.sentences:
            tokens = [word.text for word in sent.words]
            if '_' in tokens:
                continue
            tokens_lower = [word.text.lower() for word in sent.words]
            random.shuffle(TARGET_PREP)
            for prep in TARGET_PREP:
                if  tokens_lower.count(prep) == 1 and type_counter[prep] < NUM_LIMIT:
                    pred = high_prediction_confidence(tokens_lower, prep, tagger, 0.8)
                    if pred is not None:
                        sense, _ = pred
                        type_counter[prep] += 1
                        question = generate_question(tokens, prep, sense, scorer, corrector)
                        if question is not None:
                            line = json.dumps(question, ensure_ascii=False)
                            out_f.write(line+'\n')
                            res.append(question)
                            break
            used_time = time.time() - START_TIME
            remained_time = (used_time / (len(res) + init_length + 1)) * (NUM_LIMIT * len(TARGET_PREP) - len(res) - init_length)
            print('\r', f'{len(res) + init_length} of {NUM_LIMIT * len(TARGET_PREP)}, took {used_time:.2f}s, {remained_time:.2f}s remains.', end='', flush=True)
            if len(res) + init_length >= NUM_LIMIT * len(TARGET_PREP):
                return res
    return res

def load_prep_feature_dict():
    dictionary = json.load(open('dataset_creation/material/patterns_prep.json'))
    dictionary = {feature: preps for feature, preps in dictionary.items() if len(preps) >= 4}
    return dictionary

prep_feture_dict = load_prep_feature_dict()

if __name__ == '__main__':
    tgt_datasets = [('multi_news', 'train', 'document'), ('ag_news', 'train', 'text')]
    save_path = ''
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tagger = prepTagger(device)
    # scorer = LMScorer.from_pretrained('gpt2', device=device, batch_size=1)
    scorer = BertScorer(device=device)
    corrector = GECor(use_gpu=True if device=='cuda:0' else False)
    res = []
    with open(save_path, 'w') as f:
        for name, split, key in tgt_datasets:
            dataset = load_dataset(name, split='train')
            print(f'Parsing {name}...')
            res = res + build(dataset, key, tagger, scorer, corrector, f, len(res))
            if len(res) >= NUM_LIMIT * len(TARGET_PREP):
                break
   
        for instance in res:
            line = json.dumps(instance, ensure_ascii=False)
            f.write(line+'\n')