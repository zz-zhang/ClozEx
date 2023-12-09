from dependency_parse import parse_sentences, parse2tree
from datasets import load_dataset
from tqdm import tqdm
import json

def parse(dataset, save_path):
    with open(save_path, 'w') as f:
        text = dataset['text']
        for doc in tqdm(text):
            res = parse_sentences(doc)
            for line in res:
                json.dump(line, f)
                f.write('\n')


if __name__ == '__main__':
    tgt_datasets = ['cc_news', 'd_ag_news', 'd_multi_news']

    save_path = 'dataset_creation/material/parsed_corpus/{}.json'
    
    for name in tgt_datasets:
        dataset = load_dataset(name, split='train')
        print(f'Parsing {name}...')
        parse(dataset, save_path.format(name))