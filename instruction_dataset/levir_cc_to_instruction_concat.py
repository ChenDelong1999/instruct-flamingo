
import json
import os
import random
from PIL import Image
import tqdm

instruction_templates = [
    "What are the differences between this pair of remote sensing images? Note that these two images could possibly be the same.",
    "Can you spot the potential differences between these two remote sensing images?",
    "Identify the (possible) differences between the two remote sensing images.",
    "List the differences between these two satellite images. It is also possible that these two images are the same.",
    "Describe the differences between the satellite images if these images are not same.",
    "What changes can you observe between the two satellite pictures? Please note that these two pictures could be the same.",
    "Please point out all the (potential) differences between the two satellite photos.",
    "Find the differences between the given satellite images there are some changes.",
    "Are there some changes? Tell me the differences between these two image of earth observations.",
    "Explain the variations between the two remote sensing images if changes appears.",
    "Detect the possible differences between these two images captured by satellite.",
    "If these two images are same? What are the dissimilarities between the pictures from satellite?",
    "Are there something changed? Note the differences in the given satellite images.",
    "Please enumerate the differences between the remote sensing pictures, if there are some changes.",
    "Whether it is changed? State the differences between the provided remote sensing images.",
]

def concat_two_images(first_image_path, second_image_path, save_image_path):
    # Load the images
    first_image = Image.open(first_image_path)
    second_image = Image.open(second_image_path)

    # randomly choose from vertical and horizontal concatenation
    if random.random() < 0.5:
        # Concatenate vertically
        concatenated_image = Image.new('RGB', (first_image.width, first_image.height + second_image.height))
        concatenated_image.paste(first_image, (0, 0))
        concatenated_image.paste(second_image, (0, first_image.height))

        # Generate the new image path
        first_image_name = os.path.basename(first_image_path)
        second_image_name = os.path.basename(second_image_path)

    else:
        # Concatenate horizontally
        concatenated_image = Image.new('RGB', (first_image.width + second_image.width, first_image.height))
        concatenated_image.paste(first_image, (0, 0))
        concatenated_image.paste(second_image, (first_image.width, 0))

        # Generate the new image path
        first_image_name = os.path.basename(first_image_path)
        second_image_name = os.path.basename(second_image_path)

    # Save the concatenated image
    concatenated_image.save(save_image_path)



def convert_to_alpaca_format(input_json_file):
    with open(input_json_file, "r") as f:
        data = json.load(f)['images']#[:10]
    

    instruction_data = {
        'train': [],
        'val': [],
        'test': [],
    }

    for item in tqdm.tqdm(data):
        sentences = [sentence['raw'][1:-2] for sentence in item["sentences"]]
        split = item["split"]
        filename = item["filename"]

        os.makedirs(f"/cpfs/shared/research-llm/instruc_data_en/multimodal_instruct_tuning/Levir-CC/images/{split}/concat/", exist_ok=True)

        # earlier_image_identifier = 'Earlier image'
        # later_image_identifier = 'Later image'
        # spliter = ', '
        left_image_path = f"/cpfs/shared/research-llm/instruc_data_en/multimodal_instruct_tuning/Levir-CC/images/{split}/A/{filename}"
        right_image_path = f"/cpfs/shared/research-llm/instruc_data_en/multimodal_instruct_tuning/Levir-CC/images/{split}/B/{filename}"
        concat_image_path = f"/cpfs/shared/research-llm/instruc_data_en/multimodal_instruct_tuning/Levir-CC/images/{split}/concat/{filename}"

        concat_two_images(left_image_path, right_image_path, concat_image_path)

        input_text = f'<img_path>{concat_image_path}<img_path>'

        # chose the longest one
        output_text = max(sentences, key=len).strip().capitalize() + '.'
        # output_text = '\n'.join(sentences)

        instruction_data[split].append(
            {
                "input": random.choice(instruction_templates) + ' ' + input_text,
                "output": output_text,
            }
        )

    # for split in ['train', 'val', 'test']:
    for split in ['train']:

        with open('converted_datasets/levir-cc-caption/' + os.path.basename(input_json_file).replace('.json', f'_instruction_{split}_concat.json'), "w") as f:
            json.dump(instruction_data[split], f, ensure_ascii=False, indent=2)

def main():
    os.makedirs("converted_datasets/levir-cc-caption", exist_ok=True)
    convert_to_alpaca_format(
        "/cpfs/shared/research-llm/instruc_data_en/multimodal_instruct_tuning/Levir-CC/LevirCCcaptions.json"
    )

if __name__ == "__main__":
    main()

