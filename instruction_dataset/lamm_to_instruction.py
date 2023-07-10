import json
import random
import os
from PIL import Image
import tqdm

def llava_to_instruction(input_filename, output_filename):
    with open(os.path.join(dataset_dir, input_filename), 'r') as f:
        input_data = json.load(f)
    
    output_data = []
    
    for item in tqdm.tqdm(input_data):
        if 'bamboo' in item['image']:
            try:
                img = Image.open(os.path.join(image_dir, item['image']))
            except:
                print(f"Image {item['image']} not found, skipping...")
                continue
            # continue
        
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
                        'input': f"<img_path>{image_dir}/{item['image']}<img_path>" + first_instruction,
                        'output': output
                    }
                    output_data.append(output_item)
                else:
                    output_item = {
                        'input': f"<img_path>{image_dir}/{item['image']}<img_path>{conversation_history.strip()}",
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
    dataset_dir = '/cpfs/shared/research-llm/instruc_data_en/multimodal_instruct_tuning/lamm/LAMM/raw/2D_Instruct/meta_file'
    image_dir = '/cpfs/shared/research-llm/instruc_data_en/multimodal_instruct_tuning/lamm/LAMM/raw/2D_Instruct'
    os.makedirs('converted_datasets/lamm', exist_ok=True)
    llava_to_instruction('LAMM_instruct_186k.json', 'converted_datasets/lamm/LAMM_instruct_186k.json')