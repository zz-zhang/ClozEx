from dis import Instruction
import os
import openai
import json
import random
import time
from tqdm import tqdm

_type = 'deriVerb'
in_path = ''
# _type = 'prep'
# in_path = ''

out_path = ''
NUM_PER_FILE = 10000

key = 'YOUR OPENAI API KEY'

def paraphrase(text):
    openai.organization = "YOUR OPENAI ORGANIZATION"
    openai.api_key = key

    model = 'gpt-3.5-turbo'
    res = openai.ChatCompletion.create(
        model=model,
        messages=[
            {'role': 'system', 'content': 'You are an English teacher.'},
            {'role': 'user', 'content': f'Paraphrase the following explanation of a fill-in-the-blank question: {text}'}
            ]
        )
    return res


if __name__ == '__main__':
    with open(in_path) as f:
        questions = [json.loads(line) for line in f]

    print(len(questions))
    finish_and_quit = False

    while not finish_and_quit:
        print('START')
        finish_and_quit = True
        start = 0
        while start < len(questions):
            print(f'Processing from {start} to {start + NUM_PER_FILE}.')
            items = questions[start: start + NUM_PER_FILE]
            if os.path.isfile(out_path.format(start // NUM_PER_FILE, _type)):
                with open(out_path.format(start // NUM_PER_FILE, _type)) as f:
                    saved = json.load(f)
                    saved_idx = [i['index'] for i in saved]
            else:
                saved = []
                saved_idx = []
            start_at_flag = True
            for idx, item in tqdm(enumerate(items)):
                if item['index'] in saved_idx:
                    continue
                else:
                    if start_at_flag:
                        print(f'Start at {item["index"]}')
                        start_at_flag = False
                    try:
                        exp = item['explanation']
                        para = paraphrase(exp)
                        para = para['choices'][0]['message']['content']
                        # para = ''
                        item['para_exp'] = para
                        saved.append(item)
                        saved_idx.append(item['index'])
                    except:
                        print(f'Error on {item["index"]}')
                        finish_and_quit = False
                        break

            with open(out_path.format(start // NUM_PER_FILE, _type), 'w') as f:
                json.dump(saved, f)
            print(f'Saved in {out_path.format(start // NUM_PER_FILE, _type)}')


            start = start + NUM_PER_FILE
            