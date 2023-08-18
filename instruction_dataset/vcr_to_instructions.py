import json
import os
import tqdm
import random
from PIL import Image

def modify_sentence(sentence):
    # if there is a space between the ,/./!/?  and the previous word, remove the space
    # sentence = str(sentence)
    sentence = sentence.replace(" ,", ",")
    sentence = sentence.replace(" .", ".")
    sentence = sentence.replace(" !", "!")
    sentence = sentence.replace(" ?", "?")
    sentence = sentence.replace(" '", "'")
    sentence = sentence.capitalize()
    return sentence

def map_to_letters(number):
    if not isinstance(number, int) or number < 0:
        raise ValueError("input must be a non-negative integer.")

    if number < 10:
        # Map to lowercase letters 'a' to 'j'
        return chr(ord('a') + number)
    else:
        # Map to 'aa' to 'az', 'ba' to 'bz', and so on
        first_letter = chr(ord('a') + (number // 26) - 1)
        second_letter = chr(ord('a') + (number % 26))
        return f"{first_letter}{second_letter}"

def get_mentioned_objects(question, answer_choices, rationale_choices):
    mentioned_objects = set()
    
    for item in question:
        if isinstance(item, list):
            mentioned_objects.update(item)
            
    for answer in answer_choices:
        for item in answer:
            if isinstance(item, list):
                mentioned_objects.update(item)
                
    for rationale in rationale_choices:
        for item in rationale:
            if isinstance(item, list):
                mentioned_objects.update(item)
    
    return mentioned_objects

def save_crop_image(image_path, bbox, save_dir, file_name):
    x1, y1, x2, y2, confident = bbox
    image = Image.open(image_path)
    cropped_image = image.crop((x1, y1, x2, y2))
    new_image_path = os.path.join(save_dir, file_name)
    cropped_image.save(new_image_path)

    return new_image_path



def generate_instruction(object_names, boxes, width, height, question, answer_choices, mentioned_objects, full_image):
    for inds in range(len(question)):
        if type(question[inds])==list:
            question[inds]=replace_object_inds(question[inds], object_names)

    instruction = f"{' '.join(question)}\n"
    instruction = modify_sentence(instruction)

    question_choice = "Answer choices:\n"
    for i, answer in enumerate(answer_choices):
        for inds in range(len(answer)):
            if type(answer[inds])==list:
                answer[inds]=replace_object_inds(answer[inds], object_names)
        question_choice += f"{i}: {modify_sentence(' '.join(answer))}\n"
            
    # if random.random() < 0.5:
    # instruction += question_choice

    identifier_to_region_image = {}
    for i, (obj, box) in enumerate(zip(object_names, boxes)):
        if i in mentioned_objects:
            cropped_image_path = save_crop_image(
                full_image, box, 
                '/cpfs/user/chendelong/instruction_tuning_dataset/vcr/vcr_cropped_images', 
                os.path.basename(full_image)+f'-{obj}-{map_to_letters(i)}.jpg')
            
            # instruction += f"{obj}-{map_to_letters(i)}: <img_path>{cropped_image_path}<img_path>\n"

            if f'{obj}-{map_to_letters(i)}' in instruction:
                instruction = instruction.replace(f'{obj}-{map_to_letters(i)}', f'{obj}-{map_to_letters(i)}<img_path>{cropped_image_path}<img_path>')
            else:
                identifier_to_region_image[f'{obj}-{map_to_letters(i)}'] = cropped_image_path

    for key, value in identifier_to_region_image.items():
        instruction += f'{key}: <img_path>{value}<img_path>\n'
    # instruction += f"Image width: {width}, height: {height}\n"

    return instruction

def generate_output(object_names, answer_choices, answer_label, rationale_choices, rationale_label):

    answer = " ".join(answer_choices[answer_label])
    all_mentioned_objects = []
    for inds in range(len(rationale_choices[rationale_label])):
        if type(rationale_choices[rationale_label][inds])==list:
            identifier, mentioned_objects = replace_object_inds(rationale_choices[rationale_label][inds], object_names, return_mentioned_objects=True)
            rationale_choices[rationale_label][inds]=identifier
            all_mentioned_objects.extend(mentioned_objects)


    rationale = " ".join(rationale_choices[rationale_label])

    answer = modify_sentence(answer)
    rationale = modify_sentence(rationale)

    return f"{answer} {rationale}", all_mentioned_objects


def replace_object_inds(inds_list, object_names, return_mentioned_objects=False):
    str = ''
    mentioned_objects = []
    for ind in inds_list:
        str += f'{object_names[ind]}-{map_to_letters(ind)}'
        mentioned_objects.append(object_names[ind])
        if len(inds_list)>1 and inds_list.index(ind)!=len(inds_list)-1:
            if ind != inds_list[-2]:            
                str += ', '
            else:
                str += ', and '
    # print(inds_list, object_names, str)
    if return_mentioned_objects:
        return str, mentioned_objects
    else:
        return str
    

def process_vcr_split(split):
    instructions = []
    with open(f"/cpfs/user/chendelong/instruction_tuning_dataset/vcr/{split}.jsonl") as f:
        data = [json.loads(line) for line in f]#[:50000]
    for item in tqdm.tqdm(data):
        object_names = item["objects"]
        image_filename = item["img_fn"]
        question = item["question"]
        answer_choices = item["answer_choices"]
        answer_label = item["answer_label"]
        rationale_choices = item["rationale_choices"]
        rationale_label = item["rationale_label"]
        
        with open(os.path.join("/cpfs/user/chendelong/instruction_tuning_dataset/vcr/vcr1images", item["metadata_fn"])) as metadata_file:
            metadata = json.load(metadata_file)
            boxes = metadata["boxes"]
            width = metadata["width"]
            height = metadata["height"]
        
        mentioned_objects = get_mentioned_objects(question, answer_choices, rationale_choices)

        instruction = generate_instruction(
            object_names, boxes, width, height, question, answer_choices, mentioned_objects, full_image = f'/cpfs/user/chendelong/instruction_tuning_dataset/vcr/vcr1images/{image_filename}')
        input_data = f"<img_path>/cpfs/user/chendelong/instruction_tuning_dataset/vcr/vcr1images/{image_filename}<img_path>"
        output, all_mentioned_objects = generate_output(object_names, answer_choices, answer_label, rationale_choices, rationale_label)

        for mentioned_object in all_mentioned_objects:
            if mentioned_object not in instruction:
                input_data += 'NOT MENTIONED!!!'

        instructions.append({"input":  input_data + instruction, "output": output})

    with open(f"converted_datasets/vcr/vcr_{split}_instructions.json", "w") as f:
        json.dump(instructions, f, indent=4)


def main():
    os.makedirs('converted_datasets/vcr', exist_ok=True)
    # for split in ["train", "val", "test"]:
    for split in ["train", "val"]:
    # for split in ["val"]:
        process_vcr_split(split)


if __name__ == "__main__":
    main()

