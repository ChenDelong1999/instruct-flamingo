import json
import os


def get_images_with_keyword(target_directory, keyword):
    image_paths = []
    for root, _, files in os.walk(target_directory):
        for file in files:
            if keyword in file and file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                image_paths.append(image_path)
    return image_paths

nsfw_images = get_images_with_keyword('/cpfs/user/chendelong/instruction_tuning_dataset/data_augmentation/P2datasetFull/train/2', 'train3')


samples = []
for nsfw_image in nsfw_images:
    samples.append({
        'input': f'<img_path>{nsfw_image}<img_path>',
        'output': '[Clever Flamingo: NSFW content detected!]'
    })
print(len(samples))
os.makedirs('converted_datasets/nsfw', exist_ok=True)
json.dump(samples, open('converted_datasets/nsfw/nsfw.json', 'w'), indent=4)