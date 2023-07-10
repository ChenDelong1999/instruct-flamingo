import json
import os
import random

image_description_requests = [
    "Describe the following image in detail",
    "Provide a detailed description of the given image",
    "Give an elaborate explanation of the image you see",
    "Share a comprehensive rundown of the presented image",
    "Offer a thorough analysis of the image",
    "Explain the various aspects of the image before you",
    "Clarify the contents of the displayed image with great detail",
    "Characterize the image using a well-detailed description",
    "Break down the elements of the image in a detailed manner",
    "Walk through the important details of the image",
    "Portray the image with a rich, descriptive narrative",
    "Narrate the contents of the image with precision",
    "Analyze the image in a comprehensive and detailed manner",
    "Illustrate the image through a descriptive explanation",
    "Examine the image closely and share its details",
    "Write an exhaustive depiction of the given image"
]
def capitalize_first_letters(text):
    # Capitalize the first letter of the text
    text = text[0].upper() + text[1:]

    # Capitalize the first letter after every period
    sentences = text.split(". ")
    capitalized_sentences = [sentence[0].upper() + sentence[1:] for sentence in sentences]
    capitalized_text = ". ".join(capitalized_sentences)

    return capitalized_text

def transform_paragraphs_to_instructions(paragraphs_file, splits_file):
    with open(paragraphs_file, 'r') as pf:
        paragraphs = json.load(pf)

    with open(splits_file, 'r') as sf:
        splits = json.load(sf)

    instructions = []
    for p in paragraphs:
        if p['image_id'] in splits:
            dir, img = p['url'].split('/')[-2:]
            img_path = os.path.join(dir, img)

            instruction = random.choice(image_description_requests) + '. '
            paragraph = p['paragraph'].strip().replace('  ', ' ')
            capitalized_paragraph = capitalize_first_letters(paragraph)

            instructions.append({
                "input": f"{instruction}<img_path>/cpfs/shared/research-llm/instruc_data_en/multimodal_instruct_tuning/visual_genome/Visual_Genome_Dataset_V1.2/raw/data/{img_path}<img_path>",
                "output": capitalized_paragraph,
            })

    return instructions

def main():
    input_file = '/cpfs/shared/research-llm/instruc_data_en/multimodal_instruct_tuning/image_paragraph_captioning/paragraphs_v1.json'
    os.makedirs('converted_datasets/img2paragraph', exist_ok=True)
    output_template = 'converted_datasets/img2paragraph/img2p_{}.json'
    splits_files = {
        'train': '/cpfs/shared/research-llm/instruc_data_en/multimodal_instruct_tuning/image_paragraph_captioning/train_split.json', 
        'val': '/cpfs/shared/research-llm/instruc_data_en/multimodal_instruct_tuning/image_paragraph_captioning/val_split.json', 
        'test': '/cpfs/shared/research-llm/instruc_data_en/multimodal_instruct_tuning/image_paragraph_captioning/test_split.json'
        }

    for split_name, split_file in splits_files.items():
        instructions = transform_paragraphs_to_instructions(input_file, split_file)
        with open(output_template.format(split_name), 'w') as outfile:
            json.dump(instructions, outfile, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()
