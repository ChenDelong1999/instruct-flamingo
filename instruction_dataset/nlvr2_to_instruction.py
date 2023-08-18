import json
import os
import random


question_answer_templates = [
    ("Is the statement '{}' correct?", ("Yes, it's correct.", "No, it's incorrect.")),
    ("Does the description '{}' hold true?", ("Yes, it holds true.", "No, it doesn't hold true.")),
    ("Can we confirm that '{}' is accurate?", ("Yes, we can confirm it's accurate.", "No, we cannot confirm it's accurate.")),
    ("Is it true to say that '{}'", ("Yes, it's true.", "No, it's false.")),
    ("Is the assertion '{}' valid?", ("Yes, it's valid.", "No, it's invalid.")),
    ("Is the given information '{}' reliable?", ("Yes, it's reliable.", "No, it's unreliable.")),
    ("Can we verify that '{}' is true?", ("Yes, we can verify it's true.", "No, we cannot verify it's true.")),
    ("Is it correct to assume that '{}'", ("Yes, it's correct to assume that.", "No, it's incorrect to assume that.")),
    ("Is the claim '{}' accurate?", ("Yes, the claim is accurate.", "No, the claim is inaccurate.")),
    ("Is the following statement true: '{}'?", ("Yes, the statement is true.", "No, the statement is false.")),
    ("Regarding '{}', is this true or false?", ("It's true.", "It's false.")),
    ("Is '{}' a true statement?", ("Yes, it's a true statement.", "No, it's a false statement.")),
    ("Is the sentence '{}' correct?", ("Yes, the sentence is correct.", "No, the sentence is incorrect.")),
    ("Is it factual that '{}'", ("Yes, it's factual.", "No, it's not factual.")),
    ("Can we accept '{}' as true?", ("Yes, we can accept it as true.", "No, we cannot accept it as true.")),
    ("Is there truth in the statement '{}'", ("Yes, there is truth in the statement.", "No, there is no truth in the statement.")),
    ("Is the declaration '{}' true?", ("Yes, the declaration is true.", "No, the declaration is false.")),
    ("Would you say '{}' is true?", ("Yes, I would say it's true.", "No, I would say it's false.")),
    ("Is it the case that '{}'", ("Yes, it's the case.", "No, it's not the case.")),
    ("Is the following accurate: '{}'?", ("Yes, it's accurate.", "No, it's not accurate.")),
    # Variations that start with the statement
    ("{}. Is that true?", ("Yes, that's true.", "No, that's not true.")),
    ("{}. Can we confirm this?", ("Yes, we can confirm this.", "No, we cannot confirm this.")),
    ("{}. Is this accurate?", ("Yes, this is accurate.", "No, this is not accurate.")),
    ("{}. Is this statement correct?", ("Yes, this statement is correct.", "No, this statement is incorrect.")),
    ("{}. Is this true?", ("Yes, this is true.", "No, this is not true.")),
    ("{}. Is this valid?", ("Yes, this is valid.", "No, this is not valid.")),
    ("{}. Is this reliable?", ("Yes, this is reliable.", "No, this is not reliable.")),
    ("{}. Can we verify the truth?", ("Yes, we can verify the truth.", "No, we cannot verify the truth.")),
    ("{}. Can we assume this?", ("Yes, we can assume this.", "No, we cannot assume this.")),
    ("{}. Is this claim accurate?", ("Yes, this claim is accurate.", "No, this claim is not accurate.")),
    ("{}. True or false?", ("True.", "False.")),
    ("{}. Is this a true statement?", ("Yes, this is a true statement.", "No, this is a false statement.")),
    ("{}. Is this sentence correct?", ("Yes, this sentence is correct.", "No, this sentence is not correct.")),
    ("{}. Is it factual?", ("Yes, it's factual.", "No, it's not factual.")),
    ("{}. Can we accept this as true?", ("Yes, we can accept this as true.", "No, we cannot accept this as true.")),
    ("{}. Is there truth in this statement?", ("Yes, there is truth in this statement.", "No, there is no truth in this statement.")),
    ("{}. Is the declaration true?", ("Yes, the declaration is true.", "No, the declaration is not true.")),
    ("{}. Would you say this is true?", ("Yes, I would say this is true.", "No, I would say this is not true.")),
    ("{}. Is it the case?", ("Yes, it's the case.", "No, it's not the case.")),
    ("{}. Is the following accurate?", ("Yes, the following is accurate.", "No, the following is not accurate."))
]



identifier_templates = [
    ["left", "left image", "left picture", "left photo", "the left image", "the left picture", "the left photo", "the image on the left", "the picture on the left", "the photo on the left", "first", "first image", "first picture", "first photo", "the first image", "the first picture", "the first photo", "first one", "the first one", "image one"],
    ["right", "right image", "right picture", "right photo", "the right image", "the right picture", "the right photo", "the image on the right", "the picture on the right", "the photo on the right", "second", "second image", "second picture", "second photo", "the second image", "the second picture", "the second photo", "second one", "the second one", "image two"]
]

left_image_identifier = random.choice(identifier_templates[0])
right_image_identifier = identifier_templates[1][identifier_templates[0].index(left_image_identifier)]


spliter_templates = [
    ",",",",",",",",",",",",",",",",",",",", 
    ";",
    "|",
    ".",
    "\n",
    "/",
    "\\",
    "-",
    ":",
    " ",
    "\t"
    " - "
]



def read_nlvr2_data(json_file_path):
    data = []
    with open(json_file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def get_local_image_paths(example, base_path):
    identifier = example['identifier']
    split, set_id, pair_id, sentence_id = identifier.split('-')
    
    left_image_name = f"{split}-{set_id}-{pair_id}-img0.png"
    right_image_name = f"{split}-{set_id}-{pair_id}-img1.png"

    if split == 'train':
        # Use the 'directory' field for train set
        directory = example['directory']
        left_image_path = os.path.join(base_path, str(directory), left_image_name)
        right_image_path = os.path.join(base_path, str(directory), right_image_name)
    else:
        left_image_path = os.path.join(base_path, left_image_name)
        right_image_path = os.path.join(base_path, right_image_name)
    
    return left_image_path, right_image_path


def convert_data_to_new_format(data):
    converted_data = []
    for example in data:
        # Generate instruction with a random question-answer template pair
        question_template, (positive_answer_template, negative_answer_template) = random.choice(question_answer_templates)
        question = question_template.format(example['sentence'][:-1])  # remove the last period
        instruction = f"{question}"

        # Generate input with random identifier and spliter templates
        left_image_identifier = random.choice(identifier_templates[0])
        right_image_identifier = identifier_templates[1][identifier_templates[0].index(left_image_identifier)]
        spliter = random.choice(spliter_templates)
        input_text = f"{left_image_identifier}<img_path>{example['left_image_path']}<img_path>{spliter}{right_image_identifier}<img_path>{example['right_image_path']}<img_path>"

        # Generate output with the corresponding answer template
        output = positive_answer_template if example['label'] == 'True' else negative_answer_template

        if random.random() < 0.5:
            instruction = instruction + ' ' + input_text
        else:
            instruction = input_text + ' ' + instruction

        # Add the converted example to the new dataset
        converted_example = {
            "input": instruction,
            "output": output
        }
        converted_data.append(converted_example)
    return converted_data

def save_data_to_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # Define file paths and image base paths
    data_info = {
        'train': {
            'json_path': '/cpfs/user/chendelong/instruction_tuning_dataset/nlvr/nlvr2/data/train.json',
            'image_base_path': '/cpfs/user/chendelong/instruction_tuning_dataset/nlvr/nlvr2_data/images/train',
            'converted_json_path': 'converted_datasets/nlvr2/nlvr2_train.json'
        },
        'dev': {
            'json_path': '/cpfs/user/chendelong/instruction_tuning_dataset/nlvr/nlvr2/data/dev.json',
            'image_base_path': '/cpfs/user/chendelong/instruction_tuning_dataset/nlvr/nlvr2_data/dev',
            'converted_json_path': 'converted_datasets/nlvr2/nlvr2_dev_data.json'
        },
        'test1': {
            'json_path': '/cpfs/user/chendelong/instruction_tuning_dataset/nlvr/nlvr2/data/test1.json',
            'image_base_path': '/cpfs/user/chendelong/instruction_tuning_dataset/nlvr/nlvr2_data/test1_img',
            'converted_json_path': 'converted_datasets/nlvr2/nlvr2_test_1a.json'
        },
        'test2': {
            'json_path': '/cpfs/user/chendelong/instruction_tuning_dataset/nlvr/nlvr2/data/test2.json',
            'image_base_path': '/cpfs/user/chendelong/instruction_tuning_dataset/nlvr/nlvr2_data/test2',
            'converted_json_path': 'converted_datasets/nlvr2/nlvr2_test_2.json'
        }
    }
    os.makedirs('converted_datasets/nlvr2', exist_ok=True)
    for dataset, info in data_info.items():
        # Read the data from JSON files
        data = read_nlvr2_data(info['json_path'])

        # Add image paths to the data
        for example in data:
            left_image_path, right_image_path = get_local_image_paths(example, info['image_base_path'])
            example['left_image_path'] = left_image_path
            example['right_image_path'] = right_image_path

        # Convert the data to new format
        converted_data = convert_data_to_new_format(data)

        # Save the converted data to JSON files
        save_data_to_json(converted_data, info['converted_json_path'])

        # Print some statistics
        print(f"{dataset.capitalize()} data statistics:")
        print(f"Number of examples: {len(converted_data)}")
