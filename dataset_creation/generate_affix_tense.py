'''
Only for create questions whose options have only one words
Multi words version (e.g. "to open", "more eff...") need other scripts
'''

from distutils.command.build import build
import json
import os
import sys
from PyDictionary import PyDictionary
import nltk

from tqdm import tqdm
from dataset_creation.utils_parth import parse2tree, parse_sentences, dependencyNode
from utils import *
from nltk.stem import *
from nltk.stem.porter import *

stemmer = PorterStemmer()
pydict = PyDictionary()

def get_presuffix(word):
    stem = stemmer.stem(word)
    return word.replace(stem, '')

def read_corpus(path):
    res = []
    with open(path) as f:
        for line in f:
            line = json.loads(line)
            res.append(line)
    return res


def extract_key_word(ans, options):
    aux_list = [
        ' ', 'more', 'most', 'be', 'to', 'will'
    ]
    ans_main = ans
    opt_mains = [opt for opt in options]
    for aux in aux_list:
        ans_main = ans_main.replace(aux, '')
        opt_mains = [opt.replace(aux, '') for opt in opt_mains]
    # breakpoint()
    return ans_main, ans, opt_mains, options
    

def build_question(q_index, q_subtree, answer, options, q_type, corpus_trees, wf_dict, target_num=5):
    res = []
    idx = 0
    key_answer, answer, key_options, options = extract_key_word(answer, options)
    for c_tree in corpus_trees:
        overlap_nodes = c_tree.has_sub_tree(q_subtree)
        # if sentence form corpus has the same pattern from the source question
        has_enough_distractor = False
        if overlap_nodes is not None:
            new_question = {
                'index': f'{q_index}_{idx}',
                'hints': set()
            }
            
            for word_from_c, word_from_q in overlap_nodes:

                if word_from_q.word == key_answer:
                    family = wf_dict.query(word_from_c.word)
                    if len(family) > 4:
                        has_enough_distractor = True
                        new_question['answer'] = answer.replace(word_from_q.word, word_from_c.word)

                else:
                    new_question['hints'].add(word_from_c.word)
            if has_enough_distractor:
                new_question['hints'] = list(new_question['hints'])
                new_question['question'] = ' '.join(c_tree.print_sentence(return_value=True)).replace(new_question['answer'], '_').replace('\n', '')
                new_question['type'] = q_type
                idx += 1
                res.append(new_question)
        if idx >= target_num:
            break
    return res

if __name__ == '__main__':
    corpus_path = 'dataset_creation/material/parsed_corpus/{}.json'
    corpus_split = ['cc_news', 'd_multi_news', 'd_ag_news']

    word_family_path = 'dataset_creation/material/word_family.dict'
    pos_dict_path = 'dataset_creation/material/pos.dict'

    pattern_question_path = 'dataset_creation/material/patterns_affixTense.json'
    generation_path = ''

    fail_counter = 0
    with open(generation_path, 'w') as out_f:
        with open(pattern_question_path) as f:
            pattern_questions = [json.loads(line) for line in f]

            # load parsed corpus
            print("Loading parsed corpus...")
            corpus = []
            for split in corpus_split:
                corpus = corpus + read_corpus(corpus_path.format(split))
            # breakpoint()
            corpus_trees = parse2tree(corpus)

            # load word family dictionary
            wf_dict = wordFamilyDict(word_family_path)


            # start build new questions
            print("Building new questions...")
            for pattern_q in pattern_questions:
                index, q_type, question, answer, options, hints = pattern_q['index'], pattern_q['type'], pattern_q['question'], pattern_q['answer'], pattern_q['options'], pattern_q['hints']
                question = question.replace('_', answer)
                
                # ignore questions whose answer includes more than one words
                if len(answer.split(' ')) > 1:
                    print(f'Question {index} has complex answer.')
                    fail_counter += 1
                    continue

                # extract subtrees by answer and hints
                try:
                    _text = parse_sentences(question, one_sent=True)
                    tree = parse2tree([_text])[0]
                    ans_nodes = [tree.search_by_word(ans) for ans in answer.split()]
                    hint_nodes = ans_nodes + [tree.search_by_word(w, ans_nodes[0]) for w in hints]
                    subtree = tree.extract_subtree(hint_nodes)
                except:
                    print(f'Error in index {index}')
                    fail_counter += 1
                    # breakpoint()
                    continue

                # build new questions that have the same subtrees
                new_questions = build_question(index, subtree, answer, options, q_type, corpus_trees, wf_dict, target_num=1000)

                # print('-----')
                # print(question)
                # print(answer)
                # print(hints)
                for new_q in new_questions:
                    # print(new_q['question'])
                    # print(new_q['answer'])
                    # print(new_q['hints'])
                    out_f.write(json.dumps(new_q, ensure_ascii=False)+'\n')

                # breakpoint()
            print(f'Finished, {fail_counter} source questions failed to generate new questions.')