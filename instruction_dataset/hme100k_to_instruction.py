import json
import random
import os

prompt = [
    'Please write out the expression of the formula in the image using LaTeX format.',
    'Now perform OCR',
    'recognize the characters in the image',
    'write out this expression in markdown',
    'this is a equation of:',
    'the textual content is:',
    'what does it says?',
    'write out this',
    'this is:',
    'text recognition result is:',
    'a mathmatical expression of',
    '  ',
    ]

def to_instruction(img_dir, label_txt, output_json):
    with open(label_txt, 'r') as f:
        lines = f.readlines()
    samples = []
    for line in lines:
        image, latex = line.strip().split('\t')
        latex = latex.replace(' ', '')
        latex = latex.replace('\angle', '\angle ')
        latex = latex.replace('\angle', '\angle ')
        latex = latex.replace('\therefore', '\therefore ')
        samples.append({
            'input': f'<img_path>{img_dir}/{image}<img_path>{random.choice(prompt)}',
            'output': f'$${latex}$$'
        })
    with open(output_json, 'w') as f:
        json.dump(samples, f, indent=4)


if __name__ == '__main__':
    os.makedirs('converted_datasets/HME100K', exist_ok=True)
    for split in ['train', 'test']:
        to_instruction(
            img_dir=f'/cpfs/user/chendelong/instruction_tuning_dataset/HME100K/{split}_images',
            label_txt=f'/cpfs/user/chendelong/instruction_tuning_dataset/HME100K/{split}_labels.txt',
            output_json=f'converted_datasets/HME100K/hme100k_{split}_instructions.json'
        )
        