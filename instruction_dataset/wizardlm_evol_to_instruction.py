import json
import random
import os

def llava_to_instruction(input_filename, output_filename):
    with open(input_filename, 'r') as f:
        input_data = json.load(f)
    
    output_data = []
    
    for item in input_data:        
        for i, conversation in enumerate(item['conversations']):
            if conversation['from'] == 'human':
                instruction = conversation['value']
                
            elif conversation['from'] == 'gpt':
                output = conversation['value']

        output_data.append({
            'input': instruction,
            'output': output
        })
    
    with open(output_filename, 'w') as f:
        print(f'Writing to {output_filename}, total {len(output_data)} items')
        json.dump(output_data, f, indent=4)

if __name__ == "__main__":
    
    user_tag = '### Human: '
    assistant_tag = '### Assistant: '

    llava_to_instruction('/cpfs/user/chendelong/instruction_tuning_dataset/WizardLM_evol_instruct_V2_143k.json', 'converted_datasets/WizardLM_evol_instruct_V2_143k.json')