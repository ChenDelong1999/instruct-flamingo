import json, os


lines = open('/cpfs/shared/research-llm/instruc_data_en/language_only_instruct_tuning/dynosaur-sub-superni/dynosaur-sub-superni.json', 'r').readlines()
converted_samples = []
for line in lines:
    line = line.strip()
    if line:
        sample = json.loads(line)
        converted_samples.append({
            'input': sample['instruction'] + ' ' + sample['input'],
            'output': sample['output']
        })

print(f'Total {len(converted_samples)} samples...')
json.dump(converted_samples, open('converted_datasets/dynosaur-sub-superni.json', 'w'), indent=4)