import json, os


samples = json.load(open('/cpfs/shared/research-llm/instruc_data_en/multimodal_instruct_tuning/video_chat_instruction/videochat_instruct_11k_frames.train.multi_round.json', 'r'))
converted_samples = []

for sample in samples:
    input_string = sample['input']
    for i in range(100):
        input_string = input_string.replace(f'f{i} ', '')
    input_string = input_string.replace('research/multimodal_instruct_tuning', 'research-llm/instruc_data_en/multimodal_instruct_tuning')
    input_string = input_string.replace('\n\n###', '\n###')
    input_string = input_string.replace('### Human:', '### Human: ')
    input_string = input_string.replace('### Assistant:', '### Assistant: ')
    converted_samples.append({
        'input': input_string,
        'output': sample['output'],
    })
print(len(converted_samples))
json.dump(converted_samples, open('/cpfs/shared/research-llm/instruc_data_en/multimodal_instruct_tuning/video_chat_instruction/videochat_for_clever_flamingo_v2.json', 'w'), indent=4)