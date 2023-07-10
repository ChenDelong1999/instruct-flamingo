import json
import os

pf_1m_path = '/cpfs/user/chendelong/open_flamingo/datasets/polite_flamingo_0608_postprocessed_all.json'
os.makedirs('converted_datasets/pf_1m', exist_ok=True)
converted_path = 'converted_datasets/pf_1m/pf_800k.json'

# pf_1m_path = '/cpfs/user/chendelong/instruction_tuning_dataset/qlora/oasst1_instructions.json'
# os.makedirs('converted_datasets/oasst1', exist_ok=True)
# converted_path = 'converted_datasets/oasst1/oasst1.json'

# pf_1m_path = '/cpfs/shared/research-llm/multimodal_instruct_tuning/ShareGPT52K/sharegpt_instruction_max2048_en.json'
# os.makedirs('converted_datasets/sharegpt', exist_ok=True)
# converted_path = 'converted_datasets/sharegpt/sharegpt.json'
import json

# 读取样本数据
samples = json.load(open(pf_1m_path, 'r'))

# 转换样本数据
converted_samples = []
for sample in samples:
    if sample['output'][-3:] == '...':
        continue
    if sample['instruction'] == '':
        instruction = sample['input']
    elif sample['input'] == '':
        instruction = sample['instruction']
    else:
        instruction = sample['instruction'] + ' ' + sample['input']

    instruction = instruction.replace('/cpfs/shared/research/multimodal_instruct_tuning', '/cpfs/shared/research-llm/instruc_data_en/multimodal_instruct_tuning')

    converted_samples.append({
        'input': instruction,
        'output': sample['output']
    })

# 按照'reward_score'字段进行排序
sorted_samples = sorted(converted_samples, key=lambda x: x.get('reward_score', 0), reverse=True)

# 保留前80万个样本
top_samples = sorted_samples[:800000]
# top_samples = sorted_samples[:800000]

# 输出样本数量
print(len(top_samples))

# 移除'reward_score'字段
for sample in top_samples:
    sample.pop('reward_score', None)

# 将转换后的样本数据保存到文件
json.dump(top_samples, open(converted_path, 'w'), indent=4)