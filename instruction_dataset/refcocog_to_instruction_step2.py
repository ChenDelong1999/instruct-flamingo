import json
import os
import random
from PIL import Image, ImageDraw
from tqdm import tqdm

random.seed(42)


# 加载json文件
with open('converted_datasets/refcocog/refcocog_train_instruction.json', 'r') as f:
    data = json.load(f)#[:1000]

# 创建保存图片的目录
save_dir = '/cpfs/user/chendelong/instruction_tuning_dataset/refcocog/cropped_reigion'
os.makedirs(save_dir, exist_ok=True)

# 遍历json文件中的每一个样本
for i, sample in tqdm(enumerate(data), total=len(data), desc='Processing images'):
    # 提取bbox信息
    bbox_info = sample['input'][sample['input'].find('{'):sample['input'].rfind('}')+1]
    bbox = json.loads(bbox_info.replace("'", '"'))

    # 读取input中对应的图片
    image_path = sample['input'][sample['input'].find('<img_path>')+10:sample['input'].rfind('<img_path>')]
    image = Image.open(image_path)

    # 根据box裁切图片区域
    cropped_image = image.crop((bbox['bbox x axis'], bbox['bbox y axis'], bbox['bbox x axis'] + bbox['bbox width'], bbox['bbox y axis'] + bbox['bbox height']))

    # 保存图片
    new_image_path = os.path.join(save_dir, f'refcocog_cropped_reigion_{i}.jpg')
    cropped_image.save(new_image_path)

    sample['input'] = sample['input'].replace(bbox_info, f'<img_path>{new_image_path}<img_path>').replace('  ', ' ')

# 保存结果
with open('converted_datasets/refcocog/refcocog_region_in_context.json', 'w') as f:
    json.dump(data, f, indent=4)
