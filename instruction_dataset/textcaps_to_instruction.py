import json
import os
from random import choice
import tqdm


def load_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def write_json_file(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


def process_textcaps_data(input_path, output_path, save_file_name):
    input_data = load_json_file(input_path)
    textcaps_output = []

    instructions = [
        "A short image caption:",
        "A short image description:",
        "A photo of",
        "An image that shows",
        "Write a short description for the image.",
        "Write a description for the photo.",
        "Provide a description of what is presented in the photo.",
        "Briefly describe the content of the image.",
        "Can you briefly explain what you see in the image?",
        "Could you use a few words to describe what you perceive in the photo?",
        "Please provide a short depiction of the picture.",
        "Using language, provide a short account of the image.",
        "Use a few words to illustrate what is happening in the picture.",
    ]

    for item in tqdm.tqdm(input_data['data']):
        random_instruction = choice(instructions)
        img_path = "<img_path>/cpfs/user/chendelong/instruction_tuning_dataset/TextCaps/" + item['image_path'] + "<img_path>"

        for caption in item['reference_strs']:
            # caption = max(item['reference_strs'], key=len).strip().capitalize()
            if caption[-1] not in ['.', '?', '!']:
                caption += '.'

            textcaps_output.append({
                "input": random_instruction + ' ' + img_path,
                "output": caption.strip().capitalize()
            })

    os.makedirs(output_path, exist_ok=True)
    write_json_file(os.path.join(output_path, save_file_name), textcaps_output)


if __name__ == '__main__':
    input_files = ["/cpfs/user/chendelong/instruction_tuning_dataset/TextCaps/TextCaps_0.1_train.json", "/cpfs/user/chendelong/instruction_tuning_dataset/TextCaps/TextCaps_0.1_val.json"]
    # input_files = ["TextCaps_0.1_val.json"]#, "TextCaps_0.1_val.json"]
    output_dir = "converted_datasets/textcaps"

    for input_file in input_files:
        process_textcaps_data(input_file, output_dir, save_file_name=input_file.split('/')[-1])
