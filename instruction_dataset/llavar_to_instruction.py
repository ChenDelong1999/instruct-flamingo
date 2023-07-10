import json
import random
import os

def llava_to_instruction(input_filename, output_filename):
    with open(os.path.join(dataset_dir, input_filename), 'r') as f:
        input_data = json.load(f)
    
    output_data = []
    
    for item in input_data:
        conversation_history = ""
        not_llavar = False
        
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
                img_path = f"{image_dir}/{item['image']}"
                if not os.path.exists(img_path):
                    print(f"Image '{img_path}' not found")
                    not_llavar = True
                    break
                if i == 1:
                    first_instruction = item['conversations'][0]['value']
                    first_instruction = first_instruction.replace("<image>\n", "").replace("\n<image>", "")
                    output_item = {
                        'input': f"<img_path>{img_path}<img_path>" + first_instruction,
                        'output': output
                    }
                    if not not_llavar:
                        output_data.append(output_item)
                else:
                    output_item = {
                        'input': f"<img_path>{img_path}<img_path>{conversation_history.strip()}",
                        'output': output
                    }
                    if not not_llavar:
                        output_data.append(output_item)
                conversation_history += f"\n{assistant_tag}{output}\n"
    
    with open(output_filename, 'w') as f:
        print(f'Writing to {output_filename}, total {len(output_data)} items')
        json.dump(output_data, f, indent=4)

if __name__ == "__main__":
    
    user_tag = '### Human: '
    assistant_tag = '### Assistant: '
    dataset_dir = '/cpfs/user/chendelong/instruction_tuning_dataset/LLaVA-Instruct-150K'
    image_dir = '/cpfs/user/chendelong/instruction_tuning_dataset/LLaVA-Instruct-150K/llaver_images'
    os.makedirs('converted_datasets/llava', exist_ok=True)

    llava_to_instruction('/cpfs/user/chendelong/instruction_tuning_dataset/LLaVA-Instruct-150K/llava_instruct_150k_llavar_20k.json', 'converted_datasets/llava/llavar_20k.json')