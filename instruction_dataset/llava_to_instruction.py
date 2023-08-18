import json
import random
import os

def llava_to_instruction(input_filename, output_filename):
    with open(os.path.join(dataset_dir, input_filename), 'r') as f:
        input_data = json.load(f)
    
    output_data = []
    
    for item in input_data:
        conversation_history = ""
        
        for i, conversation in enumerate(item['conversations']):
            if conversation['from'] == 'human':
                # Remove "<image>\n" and "\n<image>" from the instruction
                instruction = conversation['value'].replace("<image>\n", "").replace("\n<image>", "")
                if i==0:
                    conversation_history += '\n' + instruction + '\n'
                else:
                    conversation_history += f"\n{user_tag}{instruction}\n"
                
            elif conversation['from'] == 'gpt':
                output = conversation['value']#.replace('\n\n','\n')
                
                if i == 1:
                    first_instruction = item['conversations'][0]['value']
                    first_instruction = first_instruction.replace("<image>\n", "").replace("\n<image>", "")
                    output_item = {
                        'input': f"<img_path>{coco_dir}/COCO_train2014_{item['image']}<img_path>" + first_instruction,
                        'output': output
                    }
                    output_data.append(output_item)
                else:
                    output_item = {
                        'input': f"<img_path>{coco_dir}/COCO_train2014_{item['image']}<img_path>{conversation_history.strip()}",
                        'output': output
                    }
                    output_data.append(output_item)
                conversation_history += f"\n{assistant_tag}{output}\n"
    
    with open(output_filename, 'w') as f:
        print(f'Writing to {output_filename}, total {len(output_data)} items')
        json.dump(output_data, f, indent=4)

if __name__ == "__main__":
    
    user_tag = '### Human: '
    assistant_tag = '### Assistant: '
    dataset_dir = '/cpfs/user/chendelong/instruction_tuning_dataset/LLaVA-Instruct-150K'
    coco_dir = '/cpfs/user/chendelong/downloads/mscoco_2014/train2014'
    os.makedirs('converted_datasets/llava', exist_ok=True)

    llava_to_instruction('complex_reasoning_77k.json', 'converted_datasets/llava/complex_reasoning_77k.json')
    llava_to_instruction('detail_23k.json', 'converted_datasets/llava/detail_23k.json')
    llava_to_instruction('conversation_58k.json', 'converted_datasets/llava/conversation_58k.json')
