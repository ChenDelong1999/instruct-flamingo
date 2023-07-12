import json

samples = json.load(open('/cpfs/user/chendelong/open_flamingo_v2/instruction_dataset/converted_datasets/lamm/LAMM_instruct_186k.json'))

print(len(samples))
remained_samples = []
for sample in samples:
    if '[' in sample['output'] or ']' in sample['output']:
        continue
    sample['input'] = sample['input'].replace('\n\n###', '\n###')
    remained_samples.append(sample)

print(len(remained_samples))
json.dump(remained_samples, open('/cpfs/user/chendelong/open_flamingo_v2/instruction_dataset/converted_datasets/lamm/LAMM_instruct_186k_filtered.json', 'w'), indent=4)