import json
import os

import json
import os
import random
import tqdm

task3_instructions = [
    "Given this image, generate the referring expression for the object inside this <REGION>.",
    "Describe the object located within this area: <REGION>.",
    "What content is contained within this region <REGION>?",
    "Can you provide a referring expression for the object here <REGION>?",
    "What is the object found inside <REGION>?",
    "Can you describe this object <REGION>?",
    "In the image, what is the content within this patch <REGION>?",
    "Describe the object within: <REGION>.",
    "What is inside the <REGION> in the image?",
    "Please generate a description for this object <REGION>.",
    "What does the <REGION> in the image correspond to?",
    "Can you describe the content within the area of <REGION>?",
    "Please provide a referring expression for this object <REGION>.",
    "What is the content inside <REGION>?",
    "Describe the object inside <REGION>.",
    "What is the object inside <REGION>?",
    "Please provide a description of the object inside <REGION>.",
    "What is the referring expression for the object <REGION>?",
]



def read_coco_image_size(annotations_file):
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)

    image_size_dict = {}
    for img in coco_data['images']:
        image_size_dict[img['id']] = (img['width'], img['height'])
    
    return image_size_dict


def read_coco_annotations(annotations_file):
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)

    # Create a dictionary for easy lookup of bbox by annotation_id
    bbox_dict = {anno['id']: anno['bbox'] for anno in coco_data['annotations']}

    return bbox_dict

def read_coco_annotations_captions(annotations_file):
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)

    # Create a dictionary for easy lookup of bbox by annotation_id
    caption_dict = {anno['image_id']: anno['caption'] for anno in coco_data['annotations']}

    return caption_dict

def read_refexp_data(json_path, img_root, image_size_dict, bbox_dict, caption_dict, split):
    with open(json_path, 'r') as f:
        data = json.load(f)

    samples = []
    
    annotations = data['annotations']
    refexps = data['refexps']

    # Create a dictionary for easy lookup of refexp_id
    refexp_dict = {r['refexp_id']: r for r in refexps}

    # Process each annotation
    for annotation in annotations:
        image_id = annotation['image_id']
        img_path = os.path.join(img_root, f'COCO_{split}2014_{str(image_id).zfill(12)}.jpg')

        # Get image size from the dictionary
        img_width, img_height = image_size_dict[image_id]

        # Process each refexp_id
        for refexp_id in annotation['refexp_ids']:
            refexp = refexp_dict[refexp_id]
            raw_expression = refexp['raw']

            # Get the corresponding bbox from the bbox_dict using annotation_id
            bbox = bbox_dict[annotation['annotation_id']]
            caption = caption_dict[image_id]
            x, y, w, h = bbox

            if raw_expression[-1] == '.':
                raw_expression = raw_expression[:-1]
            if caption[-1] == '.':
                caption = caption[:-1]
                
            samples.append({
                'img_path': img_path,
                'img_width': img_width,
                'img_height': img_height,
                'raw_expression': raw_expression,
                'caption': caption,
                'x': x,
                'y': y,
                'w': w,
                'h': h
            })
    return samples

def merge_samples(samples):
    merged_samples = []
    processed_samples = set()

    print('merging samples...')
    for sample in tqdm.tqdm(samples):
        img_path = sample['img_path']
        x = sample['x']
        y = sample['y']
        w = sample['w']
        h = sample['h']

        # Check if the sample has already been processed
        if (img_path, x, y, w, h) in processed_samples:
            continue

        # Find samples with matching img_path, x, y, w, and h values
        matching_samples = [s for s in samples if s['img_path'] == img_path and s['x'] == x and s['y'] == y and s['w'] == w and s['h'] == h]

        # Combine captions and create raw_expression
        expressions = [s['raw_expression'] for s in matching_samples]
        raw_expression = max(expressions, key=len)
        raw_expression = raw_expression.lower().capitalize()
        if raw_expression[-1] != '.':
            raw_expression += '.'

        # Create merged sample
        merged_sample = {
            'img_path': img_path,
            'img_width': matching_samples[0]['img_width'],
            'img_height': matching_samples[0]['img_height'],
            'raw_expression': raw_expression,
            'caption': matching_samples[0]['caption'],
            'x': x,
            'y': y,
            'w': w,
            'h': h
        }

        merged_samples.append(merged_sample)

        # Add the processed sample to the set
        processed_samples.add((img_path, x, y, w, h))

    return merged_samples


def bbox_to_str(x, y, w, h):
    return str({
                'bbox x axis': x,
                'bbox y axis': y,
                'bbox width': w,
                'bbox height': h
            })

def build_tasks(samples):
    tasks = []
    for sample in samples:
        task3_instruction = random.choice(task3_instructions).replace('<REGION>', bbox_to_str(sample['x'], sample['y'], sample['w'], sample['h']))
        response = sample['raw_expression']
        task3 = {
            'input': f"<img_path>{sample['img_path']}<img_path>{task3_instruction}",
            'output': response
        }

        # tasks.append(task1)
        # tasks.append(task2)
        tasks.append(task3)

    return tasks

for split in ['train']:
    # Load image size information from MS-COCO annotations
    image_size_dict = read_coco_image_size(f'/cpfs/user/chendelong/downloads/mscoco_2014/annotations/instances_{split}2014.json')
    bbox_dict = read_coco_annotations(f'/cpfs/user/chendelong/downloads/mscoco_2014/annotations/instances_{split}2014.json')
    caption_dict = read_coco_annotations_captions(f'/cpfs/user/chendelong/downloads/mscoco_2014/annotations/captions_{split}2014.json')

    # Call the function with the desired file path and root image path
    samples = read_refexp_data(f'/cpfs/user/chendelong/instruction_tuning_dataset/refcocog/google_refexp_{split}_201511.json', f'/cpfs/user/chendelong/downloads/mscoco_2014/{split}2014', image_size_dict, bbox_dict, caption_dict, split=split)

    samples = merge_samples(samples)

    # Build tasks from samples
    tasks = build_tasks(samples)
    os.makedirs('converted_datasets/refcocog', exist_ok=True)
    # Save tasks as json file
    with open(f'converted_datasets/refcocog/refcocog_{split}_instruction.json', 'w') as f:
        json.dump(tasks, f, indent=4)

