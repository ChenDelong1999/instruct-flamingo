import json
import random
from pathlib import Path
from torch.utils.data import Dataset

class QDRL3D(Dataset):
    def __init__(
        self,
        data_path,
        split_file
    ):
        self.img_idx2path = {
            int(p.stem): p.as_posix()
            for p in sorted(Path(data_path).glob("**/*.png"), key=lambda p: int(p.stem))
        }
        with open(data_path+"/questions.json", "r") as f:
            self.all_samples = json.load(f)
        with open(data_path+"/"+split_file, "r") as f:
            self.split_idxs = json.load(f)
        self.samples = [self.all_samples[i] for i in self.split_idxs]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_idx = sample["img_idx"]
        img = self.img_idx2path[img_idx]
        question = sample["question"]
        answer = sample["answer"]
        return img, question, answer

    def __len__(self):
        return len(self.samples)

def join_words(words_list):
    return ' '.join(words_list)

def create_instruction(question, direct_answer_instructions):
    question_prompts = [
        "Based on the provided image, answer the following question. ",
        "Refer to the given image and respond to the question below: ",
        "Using the image as a reference, please address the question: ",
        "Considering the image, please provide an answer to the question. ",
        "With the image in mind, reply to the subsequent question: ",
        "Taking the supplied image into account, answer the question: ",
        "Examine the image and then respond to the inquiry. ",
        "After observing the picture, please address the question: ",
        "In light of the image, answer the question that follows: ",
        "Keeping the image in consideration, reply to the question: ",
        "View the image and proceed to answer the question: ",
        "Study the given picture and respond to the question. ",
        "Upon analyzing the image, provide an answer to the question: ",
        "In reference to the image, address the question below: ",
        "While looking at the provided image, answer the following: ",
        "Regarding the presented image, please answer the question: ",
        "Assess the image and proceed to respond to the question. ",
        "Using the image for context, answer the following question: ",
        "Reflect on the image and then reply to the question. ",
        "After inspecting the image, provide a response to the question: ",
    ]
    if random.random() > 0.5:
        question_prompt = random.choice(question_prompts)
    else:
        question_prompt = ''
    if random.random() > 0.5:
        instruction = question_prompt + question + ' ' + random.choice(direct_answer_instructions)
    else:
        instruction = question_prompt + random.choice(direct_answer_instructions) + ' ' + question
    return instruction

def create_answer_template(answer):
    true_templates = [
        "Yes, that's correct.",
        "Yes, I think so.",
        "Indeed, it's true.",
        "That's right.",
        "You are right.",
    ]

    false_templates = [
        "Sorry, I don't think that's correct.",
        "I'm afraid that's not true.",
        "Actually, it's false.",
        "I believe that's incorrect.",
        "My apologies, but that's not right.",
    ]

    if answer == 'True':
        return random.choice(true_templates)
    elif answer == 'False':
        return random.choice(false_templates)
    else:
        answer_templates = [
            "The answer is {}.",
            "OK, my answer to your question is {}.",
            "The answer to this question is {}.",
            "In short, my answer is {}.",
            "I would say, {}.",
            "I think the answer is {}.",
            "My answer is: {}.",
        ]
        return random.choice(answer_templates).format(answer)


def create_json_dataset(dataset, output_file):
    json_data = []
    for i in range(len(dataset)):
        img, question_list, answer = dataset[i]
        question = join_words(question_list).replace(" ?", "?")
        instruction = create_instruction(question, direct_answer_instructions)
        formatted_answer = create_answer_template(answer)
        json_data.append({"input": "<img_path>" + img + "<img_path>" + instruction, "output": formatted_answer})
    
    with open(output_file, "w") as f:
        json.dump(json_data, f, indent=4)

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

train_dataset = QDRL3D('/cpfs/user/chendelong/instruction_tuning_dataset/grid-3d/grid-3d', 'train_idxs.json')
val_dataset = QDRL3D('/cpfs/user/chendelong/instruction_tuning_dataset/grid-3d/grid-3d', 'val_idxs.json')
test_dataset = QDRL3D('/cpfs/user/chendelong/instruction_tuning_dataset/grid-3d/grid-3d', 'test_idxs.json')
import os
os.makedirs('converted_datasets/grid3d', exist_ok=True)
create_json_dataset(train_dataset, 'converted_datasets/grid3d/grid-3d-train.json')
create_json_dataset(val_dataset, 'converted_datasets/grid3d/grid-3d-val.json')
create_json_dataset(test_dataset, 'converted_datasets/grid3d/grid-3d-test.json')
