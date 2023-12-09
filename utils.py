from bs4 import BeautifulSoup
from collections import defaultdict
import json
import os
import urllib3
from transformers import BertConfig, BertTokenizerFast, BertForSequenceClassification, BertForMaskedLM
import numpy as np
import csv
import torch.nn.functional as F
import torch
from gramformer import Gramformer

affix_dict = json.load(open('/home/zizheng/projects/exp-gen/data/affix.json'))

class posDict:
    def __init__(self, path=''):
        if len(path) == 0:
            self.words = dict()
        else:
            with open(path) as f:
                self.words = json.load(f)

    def update(self, word, pos):
        if word not in self.words.keys():
            self.words[word] = {pos: 1}
        elif pos not in self.words[word].keys():
            self.words[word][pos] = 1
        else:
            self.words[word][pos] += 1

    def query(self, word, pos_only=False, mode='all', topk=1, threshold=0.01):
        '''
            returns (POStag, count) pairs of a given word
            pos_only: return POS tag without count
            mode: all / max / min
            topk: (when mode == max or min) number of return result
            threshold: for a POS tag in a word, if its count / total count <= threshold, we think the POS tag is a noise
        '''
        res = []
        if word in self.words:
            total_count = sum([count for count in self.words[word].values()])
            res = sorted([(pos, count) for pos, count in self.words[word].items() if count / total_count >= threshold], key=lambda x: x[1], reverse=True)
            if pos_only:
                res = [pos for pos, count in res]
            if mode == 'max':
                res = res[:topk]
            elif mode == 'min':
                res = res[-topk:]
            return res
        else:
            return None

    def common_pos(self, word1, word2):
        word1 = word1.lower()
        word2 = word2.lower()
        if self.query(word1) is None or self.query(word2) is None:
            return []
        tags1 = set([tag for tag, count in self.query(word1)])
        tags2 = set([tag for tag, count in self.query(word2)])
        common_tags = tags1.intersection(tags2)
        # breakpoint()
        return common_tags

    def save(self, out_path):
        with open(out_path, 'w') as f:
            json.dump(self.words, f)

class wordFamilyDict:
    def __init__(self, dict_path) -> None:
        self.keys = defaultdict(int)
        self.dic = defaultdict(set)
        if os.path.isfile(dict_path):
            d = json.load(open(dict_path))
            for k, v in d['keys'].items():
                k = k.lower()
                self.keys[k] = v
            for k, v in d['dic'].items():
                v = [vv.lower() for vv in v]
                self.dic[k] = set(v)
            
        self.http = urllib3.PoolManager()
        self.path = dict_path
        self.url = 'https://www.vocabulary.com/dictionary/{}'


    def query(self, word):
        word = word.lower()
        if word in self.keys:
            key = str(self.keys[word])
            return self.dic[key]
        else:
            # print('Not found in local file, querying from "vocabulary.com"...')
            self.append_word_family(word)
            self.save()
            return self.dic[self.keys[word]]

    def append_word_family(self, word):
        link = self.url.format(word)
        response = self.http.request('GET', link)
        # print(response.data.decode('utf-8'))
        page = BeautifulSoup(response.data.decode('utf-8'), 'html.parser')
        family = self._get_family(page)
        # breakpoint()
        if len(set(self.keys.values())) == 0:
            key = 0
        else:
            key = sorted(list(self.keys.values()))[-1] + 1
        self.keys[word] = key
        self.dic[key].add(word)
        for w in family:
            self.keys[w] = key
            self.dic[key].add(w)

    def _get_family(self, page):
        res = []
        try:
            family_box = page.find_all('vcom:wordfamily')[0]
            lst = json.loads(family_box['data'])
            res = [item['word'] for item in lst]
        except IndexError:
            res = []
        return res

    def save(self):
        dic_lst = {k: list(v) for k, v in self.dic.items()}
        json.dump(
            {
                'keys': self.keys,
                'dic': dic_lst
            },
            open(self.path, 'w')
        )

class prepTagger:
    def __init__(self, device='cpu'):
        self.definitions = self.read_definitions("/home/zizheng/projects/exp-gen/PSD/BertPSD/preposition-sense-disambiguation/data/definitions.tsv")
        model_name = 'bert-base-uncased'
        config = BertConfig.from_pretrained(model_name)
        config.output_hidden_states = False
        self.device = device
        self.tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path = model_name, config = config)
        self.model = BertForSequenceClassification.from_pretrained("/home/zizheng/projects/exp-gen/PSD/BertPSD/preposition-sense-disambiguation/model_save").to(self.device)

        self.preps = ["across", "to", "on", "with", "in", "of", "at", "inside", "during", "from", "as", "through", "for", "along", "like", "about", "into", "towards", "down", "behind", "round", "before", "by", "against", "between", "onto", "off", "beside", "around", "over", "among", "above", "after", "beneath"]

    def read_definitions(self, path):
        with open(path) as csv_file:
            reader = csv.reader(csv_file, delimiter='\t')
            defs = dict(reader)
        return defs

    def tag(self, sentence, return_confidence=False):
        
        # words = sentence.split(" ")
        # for idx, word in enumerate(words):
        #     if word in self.preps:
        #         temp = words[idx]
        #         words[idx] = "<head>"+temp+"</head>"
        #         prepared = " ".join(words)
                # print(prepared)

        data = self.tokenizer(
            text=[sentence],
            add_special_tokens=True,
            max_length=100,
            truncation=True,
            padding=True, 
            return_tensors='pt',
            return_token_type_ids = False,
            verbose = True).to(self.device)

        logits = self.model(data['input_ids'], token_type_ids=None, attention_mask=data['attention_mask'])[0]
        logits = F.softmax(logits, dim=1)
        if self.device != 'cpu':
            logits = logits.to('cpu')
        prediction = self.definitions[str(np.argmax(logits.detach().numpy()))]
        # print("sentence : {}\nprediction : {}\n\n".format(sentence, prediction))
        if return_confidence:
            return prediction, logits[0][logits[0].argmax()]
        return prediction 

class BertScorer:
    def __init__(self, model_name='bert-base-uncased', device=torch.device('cpu')):
        self.device = torch.device(device)
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.model = BertForMaskedLM.from_pretrained(model_name).to(self.device)
        self.softmax = F.softmax

    def score_words(self, sentence, words):
        if '[MASK]' not in sentence:
            if '_' in sentence:
                sentence = sentence.replace('_', '[MASK]')
            else:
                return None
        res = []

        words_id = self.tokenizer(words, add_special_tokens=False).input_ids
        words_id = [_id for lst in words_id for _id in lst]

        inp = self.tokenizer(sentence, return_tensors='pt')
        # print(ids)
        mask_idx = inp['input_ids'][0].tolist().index(103)
        inp = inp.to(self.device)
        with torch.no_grad():
            logits = self.model(**inp, return_dict=True).logits[0].cpu()
            logits = self.softmax(logits, dim=1)
            
            for word, word_id in zip(words, words_id):
                prob = logits[mask_idx][word_id]
                res.append((word, prob.float()))
            res = sorted(res, key=lambda x: x[1], reverse=True)
        return res

class GECor:
    def __init__(self, use_gpu=False):
        self.gf = Gramformer(models=1, use_gpu=use_gpu)

    def word_cause_error(self, sentence, word):
        corrected = self.gf.correct(sentence, max_candidates=1)
        # print(len(corrected))
        for corr in corrected:
            edits = self.gf.get_edits(sentence, corr)
            for edit in edits:
                _type, error_word, *_ = edit
                # print(_type, error_word)
                if error_word.lower() == word.lower():
                    return True
        return False


def parse(q, parser='dependency'):
    if parser == 'dependency':
        from dependency_parse import parse_sentences, dependencyNode
        parsing_list = parse_sentences(q, one_sent=True)
        tree = dependencyNode.build_from_list(parsing_list)
        return tree
    else:
        from constituency_parse import parse_sentences, consistuencyNode
        parsing_list = parse_sentences(q, one_sent=True)
        tree = consistuencyNode(parsing_list)
        return tree

def same_affix(w1, w2):
    for affix in affix_dict.keys():
        if affix.startswith('-') or affix.endswith('-'):
            affix = affix.replace('-', '').lower()
            if (w1.startswith(affix) and w2.startswith(affix)) or (w1.endswith(affix) and w2.endswith(affix)):
                return True
    return False

if __name__ == '__main__':
    # word_family_path = '/home/zizheng/projects/exp-gen/data/word_family.dict'
    # pos_dict_path = '/home/zizheng/projects/exp-gen/data/pos_dict.json'
    # wf_dict = wordFamilyDict(word_family_path)
    # pos_dict = posDict(pos_dict_path)

    # print(pos_dict.query('tried', pos_only=True))
    # print(pos_dict.query('lengthens', pos_only=True))
    # print(wf_dict.query('reliance'))

    # scorer = BertScorer(device=torch.device('cuda:0'))
    # scorer.score_words('I like _ cake.', ['eat', 'ate', 'eating', 'drinking'])

    corrector = GECor()
    print(corrector.word_cause_error("overdriving and pedestrians are being urged to avoid the area.", 'overdriving'))
    # print(corrector.word_cause_error("I did eat apple.", 'eat'))