from datasets import load_dataset
import os, json
samples = load_dataset("/cpfs/shared/research-llm/instruc_data_en/language_only_instruct_tuning/LongForm")['train']

converted_samples = []
for sample in samples:
    converted_samples.append({
        'input': sample['input'],
        'output': sample['output']
    })

print(f'Total {len(converted_samples)} samples...')
os.makedirs('converted_datasets/longform', exist_ok=True)
json.dump(converted_samples, open(f'converted_datasets/longform/longform.json', 'w'), indent=4)