import json
import os
import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import random

direct_answer_instructions = [
    "Provide a concise answer to this question without any explanation or analysis.",
    "Give a single word answer without discussing your thought process.",
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
    "(Return a minimal answer, without any discussion or analysis).",
    "(Submit a plain response, leaving out any thought process).",
    "(Communicate a brief and direct answer, avoiding any explanation).",
    "(Share a neat answer, without delving into your thoughts).",
    "(State a clear-cut answer, excluding any additional information).",
]


class VQADataset(Dataset):
    def __init__(
        self,
        split='val',
        vqa_dataset="vqa",
    ):
        image_dir_path=f"/cpfs/user/chendelong/downloads/mscoco_2014/{split}2014"
        question_path=f"/cpfs/user/chendelong/downloads/vqav2/v2_OpenEnded_mscoco_{split}2014_questions.json"
        annotations_path=f"/cpfs/user/chendelong/downloads/vqav2/v2_mscoco_{split}2014_annotations.json"
        self.questions = json.load(open(question_path, "r"))["questions"]
        self.answers = json.load(open(annotations_path, "r"))["annotations"]
        self.image_dir_path = image_dir_path
        self.vqa_dataset = vqa_dataset
        self.split = split

    def __len__(self):
        return len(self.questions)

    def get_img_path(self, question):
        if self.vqa_dataset == "vqa":
            return os.path.join(
                self.image_dir_path, f"COCO_{self.split}2014_{question['image_id']:012d}.jpg"
            )
        elif self.vqa_dataset == "ok_vqa":
            return os.path.join(
                self.image_dir_path, f"COCO_{self.split}2014_{question['image_id']:012d}.jpg"
            )
        else:
            raise Exception(f"Unknown VQA dataset {self.vqa_dataset}")

    def __getitem__(self, idx):
        question = self.questions[idx]
        answers = self.answers[idx]
        img_path = self.get_img_path(question)
        # image = Image.open(img_path)
        return {
            # "image": image,
            "question": question["question"],
            "answers": [a["answer"] for a in answers["answers"]],
            "question_id": question["question_id"],
            "img_path": img_path,  
        }

    
    def to_alpaca_format(self):
        alpaca_data = []
        for idx in tqdm.tqdm(range(len(self.questions))):
            entry = self.__getitem__(idx)
            image_path_tag = f"<img_path>{entry['img_path']}<img_path>"
            if random.random() > 0.5:
                instruction = entry["question"] + ' ' + random.choice(direct_answer_instructions)
            else:
                instruction = random.choice(direct_answer_instructions) + ' ' + entry["question"]
            # instruction = entry["question"]
            alpaca_data.append({
                "input": image_path_tag + instruction,
                "output": entry["answers"][0].capitalize()+'.'
            })
        return alpaca_data



if __name__=='__main__':
    dataset = VQADataset(split='train')
    os.makedirs('converted_datasets/vqav2', exist_ok=True)
    print(f"Total number of samples: {len(dataset)}")
    alpaca_format_data = dataset.to_alpaca_format()
    num_samples = len(alpaca_format_data)
    filename = f"converted_datasets/vqav2/train_{num_samples // 1000}k.json"

    with open(filename, 'w') as outfile:
        json.dump(alpaca_format_data, outfile)
    print(f"Data has been saved to {filename}")

    
    dataset = VQADataset(split='val')
    print(f"Total number of samples: {len(dataset)}")
    alpaca_format_data = dataset.to_alpaca_format()
    num_samples = len(alpaca_format_data)
    filename = f"converted_datasets/vqav2/val_{num_samples // 1000}k.json"

    with open(filename, 'w') as outfile:
        json.dump(alpaca_format_data, outfile)
    print(f"Data has been saved to {filename}")