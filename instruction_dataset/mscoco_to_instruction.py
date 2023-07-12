import json
import os
import random


def convert_mscoco_to_instruction(input_file, output_file, img_directory):
    with open(input_file, 'r') as f:
        data = json.load(f)

    image_dict = {}
    for item in data['annotations']:
        image_id = item['image_id']
        if image_id not in image_dict:
            image_dict[image_id] = []
        image_dict[image_id].append(item['caption'])

    converted_data = []
    for image_id, captions in image_dict.items():
        # image_file = os.path.join(img_directory, f"COCO_{'train' if 'train' in img_directory else 'val'}2017_{str(image_id).zfill(12)}.jpg")
        image_file = os.path.join(img_directory, f"{str(image_id).zfill(12)}.jpg")
        
        instruction = random.choice([
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
        ])
        # caption = max(captions, key=len).strip().capitalize()
        for caption in captions:
            caption = caption.strip().capitalize()
            if caption[-1] not in ['.', '?', '!']:
                caption += '.'
            converted_item = {
                "input": f"{instruction}<img_path>/cpfs/user/chendelong/downloads/mscoco_2017/{image_file}<img_path>",
                "output": caption,
            }
            converted_data.append(converted_item)
    print(len(converted_data))

    random.shuffle(converted_data)
    with open(output_file, 'w') as f:
        json.dump(converted_data, f, indent=4)


if __name__ == "__main__":
    os.makedirs('converted_datasets/coco_2017_captions', exist_ok=True)
    convert_mscoco_to_instruction(
        "/cpfs/user/chendelong/downloads/mscoco_2017/annotations/captions_train2017.json",
        "converted_datasets/coco_2017_captions/coco_2017_captions_train.json",
        "train2017"
    )
    convert_mscoco_to_instruction(
        "/cpfs/user/chendelong/downloads/mscoco_2017/annotations/captions_val2017.json",
        "converted_datasets/coco_2017_captions/coco_2017_captions_val.json",
        "val2017"
    )
