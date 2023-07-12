import json

samples = json.load(open('/cpfs/shared/research-llm/instruc_data_en/multimodal_instruct_tuning/artemis/artemis-v2/dataset/combined/train/artemis_train_instruction.json'))

print(len(samples))
remained_samples = []
for sample in samples:
    sample['output'] = sample['output'].replace('_', ' ')
    remained_samples.append(sample)

print(len(remained_samples))
json.dump(remained_samples, open('/cpfs/shared/research-llm/instruc_data_en/multimodal_instruct_tuning/artemis/artemis-v2/dataset/combined/train/artemis_train_instruction.json', 'w'), indent=4)