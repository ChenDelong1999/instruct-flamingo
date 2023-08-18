import json
import random
import os
import tqdm

def llava_to_instruction(input_filename, output_filename):
    with open(input_filename, 'r') as f:
        input_data = json.load(f)
    
    output_data = []
    
    for item in tqdm.tqdm(input_data):
        img_path = f"{image_dir}/{id2path[item['image_id']]}"
        for conversation in item['conversations']:
            if len(conversation['content']) == 2:# single turn
                question, answer = conversation['content'][0], conversation['content'][1]
                assert question['from'] == 'user'
                assert answer['from'] == 'gpt'

                if random.random() < 0.5:
                    instruction = f"<img_path>{img_path}<img_path>" + question['value']
                else:
                    instruction = question['value'] + f"<img_path>{img_path}<img_path>"

                output_data.append({
                    'input': instruction,
                    'output': answer['value']
                })
            else:
                conversation_history = ""
                for i, message in enumerate(conversation['content']):
                    if message['from'] == 'user':
                        instruction = message['value']
                        if i==0:
                            conversation_history += '\n' + instruction
                        else:
                            conversation_history += f"\n{user_tag}{instruction}"
                        
                    elif message['from'] == 'gpt':
                        output = message['value']
                        
                        if i == 1:
                            first_instruction = conversation['content'][0]['value']
                            output_item = {
                                'input': f"<img_path>{img_path}<img_path>" + first_instruction,
                                'output': output
                            }
                            output_data.append(output_item)
                        else:
                            output_item = {
                                'input': f"<img_path>{img_path}<img_path>{conversation_history.strip()}",
                                'output': output
                            }
                            output_data.append(output_item)
                        conversation_history += f"\n{assistant_tag}{output}"
    
    # output_data = output_data[:1000]
    with open(output_filename, 'w') as f:
        print(f'Writing to {output_filename}, total {len(output_data)} items')
        json.dump(output_data, f, indent=4)

if __name__ == "__main__":
    
    user_tag = '### Human: '
    assistant_tag = '### Assistant: '
    image_dir = '/cpfs/shared/research-llm/instruc_data_en/multimodal_instruct_tuning/SVIT/raw'
    os.makedirs('converted_datasets/svit', exist_ok=True)

    images = json.load(open('/cpfs/shared/research-llm/instruc_data_en/multimodal_instruct_tuning/SVIT/raw/annotations/vg/image_data.json', 'r'))
    id2path = {image['image_id']: 'VG_100K' + image['url'].split('VG_100K')[1] for image in images}

    llava_to_instruction('/cpfs/shared/research-llm/instruc_data_en/multimodal_instruct_tuning/SVIT/data/svit.json', 'converted_datasets/svit/svit_full.json')

    # llava_to_instruction('/cpfs/shared/research-llm/instruc_data_en/multimodal_instruct_tuning/SVIT/data/conversation.json', 'converted_datasets/svit/conversation.json')