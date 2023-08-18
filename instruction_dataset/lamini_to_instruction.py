from datasets import load_dataset
import os, json
samples = load_dataset("/cpfs/shared/research-llm/instruc_data_en/language_only_instruct_tuning/LaMini-instruction")['train']

converted_samples = []
for sample in samples:
    converted_samples.append({
        'input': sample['instruction'],
        'output': sample['response']
    })

print(f'Total {len(converted_samples)} samples...')
os.makedirs('converted_datasets/lamini', exist_ok=True)
json.dump(converted_samples, open(f'converted_datasets/lamini/lamini.json', 'w'), indent=4)