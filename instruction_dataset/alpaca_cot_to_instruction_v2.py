import os
import json
import random
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt

# https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/blob/main/README.md
alpaca_cot_dir = '/cpfs/shared/research-llm/instruc_data_en/language_only_instruct_tuning/Alpaca-CoT'

file_list = [
    f'{alpaca_cot_dir}/FLAN-Muffin/flan.json',

    f'{alpaca_cot_dir}/GPT4all/gpt4all_without_p3.json',

    f'{alpaca_cot_dir}/camel/ai_society.json',
    f'{alpaca_cot_dir}/camel/ai_society_zh.json',
    f'{alpaca_cot_dir}/camel/biology.json',
    f'{alpaca_cot_dir}/camel/chemistry.json',
    f'{alpaca_cot_dir}/camel/code.json',
    f'{alpaca_cot_dir}/camel/math.json',
    f'{alpaca_cot_dir}/camel/physics.json',

    f'{alpaca_cot_dir}/baize/medical.json',
    f'{alpaca_cot_dir}/baize/quora.json',
    f'{alpaca_cot_dir}/baize/stackoverflow.json',

    f'{alpaca_cot_dir}/Guanaco/GuanacoDataset.json',

    f'{alpaca_cot_dir}/hh-rlhf/harmless-base_chosen.json',
    f'{alpaca_cot_dir}/hh-rlhf/helpful-online_chosen.json',


    f'{alpaca_cot_dir}/gpt4tools/gpt4tools.json',

    f'{alpaca_cot_dir}/instinwild/instinwild_en.json',
    f'{alpaca_cot_dir}/instinwild/instinwild_ch.json',

    f'{alpaca_cot_dir}/alpaca/alpaca_data_cleaned.json',
    f'{alpaca_cot_dir}/alpacaGPT4/alpaca_gpt4_data.json',
    f'{alpaca_cot_dir}/alpacaGPT4/alpaca_gpt4_data_zh.json',

    f'{alpaca_cot_dir}/OIG/grade_school_math_instructions.json',
    f'{alpaca_cot_dir}/OIG/plot_screenplay_books_dialog.json',
    f'{alpaca_cot_dir}/OIG/poetry_2_song.json',

    f'{alpaca_cot_dir}/GPTeacher/Toolformer/formatted_toolformer-dedupe-only-dataset.json',
    f'{alpaca_cot_dir}/CodeAlpaca/code_alpaca.json',

    f'{alpaca_cot_dir}/dolly/dolly.json',
]

filtered_keywords = [
    'image', 'picture', 'visual', 'photo', "i'm sorry", 'i am sorry', 'sorry, ', 'ai language model', '<no input>', '<noinput>', '<nooutput>', 'gpt model', 'provide more context or details', 'content policies', 'content policy', 'have enough information to', 'i do not have access to', 'no information provided to', 'cannot answer this question', 'without additional', 'please provide more', 'do not have the ability', 'cannot generate'
]


def load_json(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)

    filtered_data = []
    for item in data:
        contains_keywords = False
        for keyword in filtered_keywords:
            for field in ['input', 'instruction', 'output']:
                if keyword in item.get(field, '').lower():
                    contains_keywords = True
                    break
            if contains_keywords:
                break

        if not contains_keywords:
            if item.get('input', ''):
                filtered_data.append({
                    'input': item.get('instruction', '').strip() + '\ninput: ' + item.get('input', '').strip(),
                    'output': item.get('output', '').strip()
                })
            else:
                filtered_data.append({
                    'input': item.get('instruction', '').strip(),
                    'output': item.get('output', '').strip()
                })
    return filtered_data



def main():
    data = []
    os.makedirs('converted_datasets/alpaca_cot', exist_ok=True)

    for file_name in file_list:
        file_data = load_json(file_name)
        file_data = [item for item in file_data if len(item['input']) < 512 and len(item['output']) < 1024]
        json.dump(file_data, open(f'converted_datasets/alpaca_cot/{os.path.basename(file_name)}', 'w'), indent=4)
        data.extend(file_data)
        print(f'{len(file_data)} - {file_name}')

    random.shuffle(data)
    print(f'Total {len(data)} samples...')
    json.dump(data, open(f'converted_datasets/alpaca_cot/alpaca_cot_merged-{int(len(data)/1000)}k.json', 'w'), indent=4)
    

if __name__ == "__main__":
    main()