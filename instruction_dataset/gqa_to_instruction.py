
import json
import random


direct_answer_instructions = [
    "Provide a concise answer to this question without any explanation or analysis.",
    "Give a brief answer without discussing your thought process.",
    "Offer a short and direct response without elaboration.",
    "Respond with a simple answer, skipping any reasoning.",
    "Share a straightforward answer, no explanation needed.",
    "Submit a quick answer without detailing your thoughts.",
    "Present a clear and concise answer without any background information.",
    "Furnish a short response, leaving out any analysis.",
    "State a brief answer, bypassing any detailed explanation.",
    "Reveal a succinct answer, forgoing any thought process.",
    "Deliver a short and simple answer without further elaboration.",
    "Produce a terse response, eliminating any reasoning.",
    "Give an unadorned answer, without discussing the thought process.",
    "Provide a to-the-point response without any detailed explanation.",
    "Offer an undecorated answer, skipping the reasoning part.",
    "(Return a minimal answer, without any discussion or analysis).",
    "(Submit a plain response, leaving out any thought process).",
    "(Communicate a brief and direct answer, avoiding any explanation).",
    "(Share a neat answer, without delving into your thoughts).",
    "(State a clear-cut answer, excluding any additional information).",
]

def gqa_to_alpaca_format(scene_graphs, questions, truncate=None):
    alpaca_data = []
    question_ids = list(questions.keys())
    
    if truncate is not None:
        question_ids = question_ids[:truncate]

    for question_id in question_ids:
        question_data = questions[question_id]
        image_id = question_data['imageId']
        # scene_graph = json.dumps(scene_graphs[image_id])
        image_path = f'/cpfs/user/chendelong/instruction_tuning_dataset/gqa/images/{image_id}.jpg'
        question_text = question_data['question']
        answer_text = question_data['answer']
        # print(scene_graph)
        
        # question_prompt = random.choice(question_prompts) if random.random() > 0.5 else ''
        if random.random() > 0.5:
            instruction =  question_text + ' ' + random.choice(direct_answer_instructions)
        else:
            instruction =  random.choice(direct_answer_instructions) + ' ' + question_text
        # instruction = question_text
            
        # if random.random() > 0.5:
        output_data = answer_text.capitalize()+'.'
        # else:
        #     output_data = f'The answer is {answer_text}.'
        
        alpaca_data.append({
            "input": f'<img_path>{image_path}<img_path>{instruction}',
            "output": output_data,
        })
    
    return alpaca_data

import os
os.makedirs('converted_datasets/gqa', exist_ok=True)

# scene_graphs = json.load(open('val_sceneGraphs.json'))
scene_graphs = None

questions = json.load(open('/cpfs/user/chendelong/instruction_tuning_dataset/gqa/val_balanced_questions.json'))

truncate = None
alpaca_format_data = gqa_to_alpaca_format(scene_graphs, questions, truncate=truncate)
filename = f"converted_datasets/gqa/gqa_val_{len(alpaca_format_data) // 1000}k_instruction.json"
with open(filename, 'w') as outfile:
    json.dump(alpaca_format_data, outfile, indent=4)
print(f"Data has been saved to {filename}")

# scene_graphs = json.load(open('train_sceneGraphs.json'))
scene_graphs = None
questions = json.load(open('/cpfs/user/chendelong/instruction_tuning_dataset/gqa/train_balanced_questions.json'))

truncate = None
alpaca_format_data = gqa_to_alpaca_format(scene_graphs, questions, truncate=truncate)
filename = f"converted_datasets/gqa/gqa_train_{len(alpaca_format_data) // 1000}k_instruction.json"
with open(filename, 'w') as outfile:
    json.dump(alpaca_format_data, outfile, indent=4)
print(f"Data has been saved to {filename}")
