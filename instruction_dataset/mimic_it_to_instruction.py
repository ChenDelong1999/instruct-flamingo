import json
import os
import random
import tqdm
import PIL.Image

subsets = [
    'VST/VST_instructions',
    # 'TVC/TVC_instructions',
    # 'SN/SN_instructions',
    # 'LA/LACONV_instructions',
    # 'LA/LACR_I2I_instructions',
    # 'LA/LACR_T2T_instructions',
    # 'LA/LADD_instructions',
]

max_images = 4
os.makedirs('converted_datasets/mimic_it', exist_ok=True)

def random_select_with_order(lst, n):
    # 如果n大于列表长度，返回原始列表
    if n >= len(lst):
        return lst
    
    # 生成索引列表
    indices = list(range(len(lst)))
    
    # 从索引列表中随机选择n个索引
    selected_indices = random.sample(indices, n)
    
    # 按照原始顺序获取被选择的元素
    selected_elements = [lst[i] for i in sorted(selected_indices)]
    
    return selected_elements

for subset in subsets:
    instruction_file = '/cpfs/shared/research-llm/instruc_data_en/multimodal_instruct_tuning/MIMIC-IT/' + subset + '.json'
    instructions = json.load(open(instruction_file, 'r'))['data']
    print(instruction_file)
    print(len(instructions.keys()))

    converted_samples = []
    for instruction_id in tqdm.tqdm(list(instructions.keys())):
        sample = instructions[instruction_id]
        instruction = sample['instruction']
        image_ids = sample['image_ids']
        
        if len(image_ids) > max_images:
            image_ids = random_select_with_order(image_ids, max_images)

        for image_id in image_ids:
            image_id = image_id.replace('SN_IMG', 'SN_00_IMG')
            image_id = image_id.replace('_color', '_00_color')
            instruction += f'<img_path>{image_id}.png<img_path>'
        output = sample['answer']

        if 'annotations' in instruction: # some responses have 'according to annotations...'
            continue
        if '[male]' in output or '[female]' in output: # visual storytelling dataset tag
            continue
        if 'VST' in subset:
            for image_id in image_ids:
                imagepath = f'/cpfs/shared/research-llm/instruc_data_en/multimodal_instruct_tuning/MIMIC-IT/images/VST/{image_id}.png'
                # # try open to verify
                # try:
                #     PIL.Image.open(imagepath)
                # except:
                #     print(imagepath)
                #     continue
                # verify image exist
                contains_broken_image = False
                if not os.path.exists(imagepath):
                    print(imagepath)
                    contains_broken_image = True
                    break
            if contains_broken_image:
                continue
                    


        converted_sample = {
            'input': instruction,
            'output': output
        }
        converted_samples.append(converted_sample)
        # pprint.pprint(converted_sample)

    print(len(converted_samples))
    json.dump(converted_samples, open(f'converted_datasets/mimic_it/{subset.split("/")[-1]}.json', 'w'), indent=4)
