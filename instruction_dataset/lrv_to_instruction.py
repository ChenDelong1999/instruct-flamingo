import os, json

meta_json = '/cpfs/shared/research-llm/instruc_data_en/multimodal_instruct_tuning/visual_genome/Visual_Genome_Dataset_V1.2/raw/data/image_data.json'
vg_images = json.load(open(meta_json, 'r'))
id2path = {}
for sample in vg_images:
    id = str(sample['image_id'])
    path = sample['url'].split('/')[-2] + '/' + sample['url'].split('/')[-1]
    id2path[id] = '/cpfs/shared/research-llm/instruc_data_en/multimodal_instruct_tuning/visual_genome/Visual_Genome_Dataset_V1.2/raw/data/' + path


input_file = '/cpfs/user/chendelong/instruction_tuning_dataset/LRV-155k.json'
os.makedirs('converted_datasets/LRV', exist_ok=True)
output_file = 'converted_datasets/LRV/LRV_filtered.json'


samples = json.load(open(input_file, 'r'))['annotations']
converted_samples = []
for sample in samples:
    image_path = id2path[sample['image_id']]
    if ':' in sample['question'] or ':' in sample['answer']  or len(sample['answer']) < 16 or sample['answer'][-1] != '.':
        continue
    converted_samples.append({
        'input': f'<img_path>{image_path}<img_path>{sample["question"]}',
        'output': sample['answer']
    })

print(f'Writing to {output_file}, total {len(converted_samples)} items')
json.dump(converted_samples, open(output_file, 'w'), indent=4)