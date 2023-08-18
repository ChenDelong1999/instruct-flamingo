import json
import os
import random
import string

def generate_random_instruction():

    if random.random() < 0.5:
        return '<image>'
    
    # 定义一个包含单词和符号的列表
    words_symbols = list(string.ascii_letters + string.digits + string.punctuation)
    
    # 随机生成要插入的内容
    num_words = random.randint(1, 5)  # 生成1到5个单词或符号
    content_to_insert = " ".join(random.choices(words_symbols, k=num_words))
    
    # 随机生成插入的位置
    position = random.randint(0, num_words * 2)
    
    # 在内容中的随机位置插入 <image>
    content_to_insert = content_to_insert[:position] + "<image>" + content_to_insert[position:]
    
    return content_to_insert



def get_images_with_keyword(target_directory, keyword):
    image_paths = []
    for root, _, files in os.walk(target_directory):
        for file in files:
            if keyword in file and file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                image_paths.append(image_path)
    return image_paths

nsfw_images = get_images_with_keyword('/cpfs/user/chendelong/instruction_tuning_dataset/data_augmentation/P2datasetFull/train/2', 'train3')


# samples = []
# for nsfw_image in nsfw_images:
#     samples.append({
#         'input': f'<img_path>{nsfw_image}<img_path>',
#         'output': '[Clever Flamingo: NSFW content detected!]'
#     })
# print(len(samples))
# os.makedirs('converted_datasets/nsfw', exist_ok=True)
# json.dump(samples, open('converted_datasets/nsfw/nsfw.json', 'w'), indent=4)
import json
import random
import os
import re

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def extract_path_and_convert_token(input_data, img_dir):
    img_path_pattern = re.compile(r'<img_path>(.*?)<img_path>')
    img_paths = [os.path.join(img_dir, path) for path in img_path_pattern.findall(input_data)]
    img_paths = [image_path.replace('/research/multimodal_instruct_tuning', '/research-llm/instruc_data_en/multimodal_instruct_tuning') for image_path in img_paths]
    # input_data_converted = img_path_pattern.sub('<image>', input_data)
    input_data_converted = img_path_pattern.sub('<image>', input_data)
    return input_data_converted, img_paths

class InstructionDataset():
    def __init__(self, json_path, image_dir_path, shuffle=True):
        with open(json_path, encoding='utf-8') as f:
            self.data = json.load(f)
        self.total_samples = len(self.data)
        if shuffle:
            random.shuffle(self.data)            
        self.image_dir_path = image_dir_path
        self.json_path = os.path.basename(json_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        input_data = sample['input']
        output_data = sample['output']
        input_data, img_paths = extract_path_and_convert_token(sample['input'], self.image_dir_path)

        return input_data, output_data, img_paths
    

dataset_config = '/cpfs/user/chendelong/open_flamingo_v2/instruction_dataset/configs/clever_flamingo_v2.json'

datasets_info = json.load(open(dataset_config, 'r'))

# i = 0
all_samples = []
for dataset_info in datasets_info:#[i:i+1]: 
    if 'nsfw' in dataset_info['dataset_name']:
        continue
    if 'cpfs' not in dataset_info['json_path']:
        dataset_info['json_path'] = '/cpfs/user/chendelong/open_flamingo_v2/' + dataset_info['json_path']
    dataset  = InstructionDataset(
        json_path=dataset_info['json_path'],
        image_dir_path=dataset_info['img_dir'],
        shuffle=True
    )
    input_data, output_data, img_paths = dataset[0]
    print(input_data)
    if '<image>' not in input_data:
        print(f'no image in {dataset_info["json_path"]}')
        continue

    print('='*64)
    for k,v in dataset_info.items():
        print(f'{k}: {v}')
    print(f'total samples: {round(len(dataset)/1000, 2)}k')
    for i in range(dataset_info['ratio']*100):
        input_data, output_data, img_paths = dataset[i]
        if '<image>' in input_data and random.random() < 0.2:
            input_data = generate_random_instruction()
        # print(f'[input]\n{input_data}\n[output]\n{output_data}\n{"-"*64}')
        nsfw_sample = {
            'input': input_data.replace('<image>', f'<img_path>{random.choice(nsfw_images)}<img_path>'),
            'output': 'Clever Flamingo Warning: NSFW content detected!'
        }
        # print(nsfw_sample)
        if '<img_path>' in nsfw_sample['input']:
            all_samples.append(nsfw_sample)

print(len(all_samples))
os.makedirs('converted_datasets/nsfw', exist_ok=True)
json.dump(all_samples, open('converted_datasets/nsfw/nsfw_in_context.json', 'w'), indent=4)

