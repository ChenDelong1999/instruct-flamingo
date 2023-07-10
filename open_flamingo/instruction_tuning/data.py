"""
Preprocess and load datasets for training.
"""

import functools
import io
import json
import math
import re
import random
import numpy as np
import torch
import torchvision
import webdataset as wds
from PIL import Image
import base64

import json
import os
import torch
import numpy as np
import random
import logging
import re
from PIL import Image, ImageOps
from torch.utils.data import Dataset
import collections
import tqdm
import copy
import string


from data_utils import *


def get_prompt_instruction(instruction, instruction_prompt_templete):
    if instruction_prompt_templete == 'guanaco':
        prompt = 'A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions.\n'
        prompt +=  f'### Human: {instruction}\n### Assistant: '
        return prompt

    elif instruction_prompt_templete == 'guanaco-no-prompt':
        return f'### Human: {instruction}\n### Assistant: '


def extract_path_and_convert_token(input_data, img_dir):
    img_path_pattern = re.compile(r'<img_path>(.*?)<img_path>')
    img_paths = [os.path.join(img_dir, path) for path in img_path_pattern.findall(input_data)]
    input_data_converted = img_path_pattern.sub('<image><|endofchunk|>', input_data)
    return input_data_converted, img_paths


def load_and_process_images(img_paths, zero_image, image_processor, max_img=None, is_train=True):
    images = []

    for img_path in img_paths:
        try:
            # FIXME: temproary fix for reorganized cpfs
            img_path = img_path.replace('research/multimodal_instruct_tuning', 'research-llm/instruc_data_en/multimodal_instruct_tuning')
            img = Image.open(img_path)
            img = pad_image_to_square(img, is_train)
            images.append(image_processor(img).unsqueeze(0))
        except Exception as e:
            print(f"Failed to load image: {e}")
            img = zero_image
            images.append(img.unsqueeze(0))

    num_images = len(images)
    if max_img is not None:
        if num_images > max_img:
            print(f'more than max_img={max_img} images, only use the first {max_img} images')
            print(f'img_paths: {img_paths}')
            images = images[:max_img]

        # Check the length of the images list and pad it to max_img images if needed
        if num_images < max_img:
            for _ in range(max_img - num_images):
                images.append(zero_image.unsqueeze(0))
    else:
        if num_images == 0:
            images.append(zero_image.unsqueeze(0))

    return torch.cat(images, dim=0)


def pad_image_to_square(img, is_train):
    img = img.convert("RGB")

    width, height = img.size
    if random.random() < 0.5 and is_train:
        max_side = max(width, height)
        background_color = tuple(np.random.randint(0, 256, 3))

        if width > height:
            padding = (0, (max_side - height) // 2, 0, (max_side - height + 1) // 2)
        else:
            padding = ((max_side - width) // 2, 0, (max_side - width + 1) // 2, 0)

        img = ImageOps.expand(img, padding, fill=background_color)

    return img

class InstructionDataset(Dataset):
    def __init__(
            self, 
            config_json_path, 
            image_processor, 
            tokenizer, 
            num_samples=-1, 
            max_length=256, 
            logger=None, 
            mode='train',
            img_dir=None,
            args=None
            ):
        self.mode = mode
        self.image_processor = image_processor
        self.tokenizer = tokenizer

        self.instruction_prompt_templete = args.instruction_prompt_templete
        self.num_samples = num_samples

        self.max_length = max_length
        if self.mode == 'train':
            self.multiturn_augmentation = args.multiturn_augmentation
            self.max_img = args.max_img
        elif self.mode == 'test':
            self.max_img = None

        self.samples = []
        total_samples_per_dataset = []
        total_samples = 0
        
        with open(config_json_path, "r") as file:
            self.configs = json.load(file)

        if 'json_path' not in self.configs[0].keys() and 'input' in self.configs[0].keys():
            # this is not a config file, instead it is the raw instruction data json
            # then assume that image paths are absolute paths
            self.configs = [{
                "json_path": config_json_path,
                "img_dir": '' if img_dir is None else img_dir,
                "dataset_name": os.path.basename(config_json_path).replace(".json", ""),
                "ratio": 1.0
            }]

        for config in self.configs:
            dataset_json_path = config["json_path"]
            img_dir = config["img_dir"]
            samples = json.load(open(dataset_json_path, "r"))
            total_samples += len(samples)
            total_samples_per_dataset.append(len(samples))
    
        if num_samples > 0:
            if args.dataset_sampling_mode == 'ratio':
                # Calculate the weights based on specified ratios
                weights = [config["ratio"] for config in self.configs]
            elif args.dataset_sampling_mode == 'sqrt':
                # Calculate the weights based on dataset sizes with a square root smoothing
                # See InstructBLIP paper https://arxiv.org/abs/2305.06500
                sizes_sqrt = [np.sqrt(size) for size in total_samples_per_dataset]
                weights = [size_sqrt / sum(sizes_sqrt) for size_sqrt in sizes_sqrt]
            # elif args.dataset_sampling_mode == 'ratio*sqrt':
            #     # Calculate the weights based on both specified ratios and dataset sizes
            #     ratio_weights = [config["ratio"] for config in self.configs]
            #     sizes_sqrt = [np.sqrt(size) for size in total_samples_per_dataset]
            #     sqrt_weights = [size_sqrt / sum(sizes_sqrt) for size_sqrt in sizes_sqrt]
            #     weights = [ratio_weight * sqrt_weight for ratio_weight, sqrt_weight in zip(ratio_weights, sqrt_weights)]
            else:
                raise ValueError(f'Invalid dataset_sampling_mode: {args.dataset_sampling_mode}. Must be "ratio", "sqrt"')# or "ratio*sqrt"')

            # Normalize the weights
            normalized_weights = [weight / sum(weights) for weight in weights]

            num_samples_per_dataset = [int(num_samples * weight) for weight in normalized_weights]
            if sum(num_samples_per_dataset) < num_samples:
                random_dataset_index = random.randint(0, len(num_samples_per_dataset) - 1)
                num_samples_per_dataset[random_dataset_index] += num_samples - sum(num_samples_per_dataset)
        # elif num_samples is not specified (default to -1), use all samples and skip samplling
       
        over_length_counts = {}
        for i, config in enumerate(self.configs):
            if args.rank==0:
                logger.info(f"Loading dataset {i+1}/{len(self.configs)} from {config['json_path']}")
            dataset_json_path = config["json_path"]
            img_dir = config["img_dir"]
            dataset_name = config["dataset_name"]
            over_length_counts[i] = 0
            samples = json.load(open(dataset_json_path, "r"))
            if self.num_samples > 0:
                samples = random.choices(samples, k=num_samples_per_dataset[i])

            if args.rank == 0:
                samples = tqdm.tqdm(samples, desc=f"Processing dataset {i+1}")
            for sample in samples:
                sample["img_dir"] = img_dir
                sample["dataset_idx"] = i
                sample["dataset_name"] = dataset_name
                if self.mode == 'train':
                    instruction_converted, _ = extract_path_and_convert_token(sample["input"], img_dir)
                    input_tokenized_length = len(self.tokenizer(get_prompt_instruction(
                        instruction_converted, 
                        self.instruction_prompt_templete
                        ))["input_ids"])
                    sample["tokenized_length"] = input_tokenized_length
                    if input_tokenized_length < max_length and sample["output"] not in ["", " ", "  ", "\n"]:
                        self.samples.append(sample)
                    else:
                        # append a random sample to make sure len(samples)=num_samples
                        # FIXME: index error when self.samples is empty (when the first sample is over-length)
                        random_sample = random.choice(self.samples)
                        self.samples.append(random_sample)
                        over_length_counts[i] += 1
                else:
                    self.samples.append(sample)
                
        if self.mode == 'train':  
            random.shuffle(self.samples)

        sampled_counts = collections.defaultdict(int)
        for sample in self.samples:
            sampled_counts[sample["dataset_idx"]] += 1

        if logger is not None and args is not None and args.rank == 0:
            logger.info(f'Instruction prompt templete: "{self.instruction_prompt_templete}"')
            logger.info(f"Initialized instruction tuning dataset from {config_json_path}")
            logger.info(f"Total available samples: {total_samples}")
            logger.info(f'Sampling mode: {args.dataset_sampling_mode}')
            logger.info(f"Number of sampls after sampling: {len(self.samples)}")
            for i, config in enumerate(self.configs):
                logger.info(f"Dataset [{i + 1}/{len(self.configs)}]: {config['dataset_name']}")
                logger.info(f"\tConfig Sampling Ratio: [{config['ratio']}]")
                logger.info(f"\tImage directory: {config['img_dir']}")
                logger.info(f"\tNum samples: {total_samples_per_dataset[i]} ({total_samples_per_dataset[i]/total_samples*100:.2f}%)")
                if self.num_samples > 0:
                    logger.info(f"\tNum samples (after sampling):  {sampled_counts[i]} ({normalized_weights[i]*100:.2f}%)")
                logger.info(f"\tNum over-length samples:       {over_length_counts[i] if i in over_length_counts.keys() else 0}")

        # Create and cache a zero-filled image for padding
        self.zero_image = torch.zeros_like(self.image_processor(Image.new('RGB', (224, 224))))
        self.all_samples = self.samples.copy()

    def get_sample(self, idx):
        return self.samples[idx]
    
    def __len__(self):
        if self.num_samples==-1:
            return len(self.samples)
        else:
            return self.num_samples
    
    def __getitem__(self, idx):
        sample = self.samples[idx]

        # accomodate legacy instruction format: add 'instruction' field if it exist
        if 'instruction' in sample.keys():
            sample['input'] = sample['instruction'] + sample['input']

        instruction, img_path = extract_path_and_convert_token(sample['input'], sample["img_dir"])
        
        dataset_idxs = [sample['dataset_idx']]
        img_paths = img_path
        img_dirs = [sample["img_dir"]]
        instructions = [instruction]
        targets = [sample['output']]

        if self.mode == 'train':
            for _ in range(self.multiturn_augmentation):
                sample = random.choice(self.all_samples)
                instruction, img_path = extract_path_and_convert_token(sample['input'], sample["img_dir"])
                if len(img_paths) + len(img_path) >= self.max_img:
                    break
                dataset_idxs.append(sample['dataset_idx'])
                img_paths.extend(img_path)
                img_dirs.append(sample["img_dir"])
                instructions.append(instruction)
                targets.append(sample['output'])

            input_ids, target_mask  = self.multi_instrucions_tokenization(instructions, targets)
            images = load_and_process_images(img_paths, self.zero_image, self.image_processor, self.max_img, is_train=True)

            # do truncation
            input_ids = input_ids[:self.max_length]
            attention_mask = [1] * len(input_ids)
            target_mask = target_mask[:self.max_length]
            
            # do padding (right padding, to max_length)
            input_ids = input_ids + [0] * (self.max_length - len(input_ids))
            target_mask = target_mask + [0] * (self.max_length - len(target_mask))
            attention_mask = attention_mask + [0] * (self.max_length - len(attention_mask))

            text = {
                "input_ids": torch.tensor(input_ids, dtype=torch.long).unsqueeze(0),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0),
            }
            target_mask = torch.tensor(target_mask, dtype=torch.long).unsqueeze(0)

            # Pad the dataset_idxs list to args.multiturn_augmentation length
            while len(dataset_idxs) < self.multiturn_augmentation:
                dataset_idxs.append(-1)
            return images, text, target_mask, torch.tensor(dataset_idxs[:self.multiturn_augmentation], dtype=torch.long)
        
        elif self.mode == 'test':
            instruction_str = get_prompt_instruction(instructions[0], self.instruction_prompt_templete, target=targets[0])
            text = self.tokenizer(
                instruction_str,
                return_tensors="pt",
                max_length=self.max_length,
            )
            images = load_and_process_images(img_paths, self.zero_image, self.image_processor, self.max_img, is_train=False)
            return images, text, instruction_str, sample


    def multi_instrucions_tokenization(self, instructions, targets):
        input_ids = []
        target_mask = []
        eos_token_id = self.tokenizer.eos_token_id

        for i in range(len(instructions)):
            if i==0:
                instruction_str = get_prompt_instruction(instructions[0], self.instruction_prompt_templete)
                instruction_tokenized = self.tokenizer.encode(instruction_str)
            else:
                if self.instruction_prompt_templete=='guanaco' or self.instruction_prompt_templete=='guanaco-no-prompt':
                    instruction_str = f"\n### Human: {instructions[i]}\n### Assistant: "
                instruction_tokenized = self.tokenizer.encode(instruction_str)#[1:]

            input_ids.extend(instruction_tokenized)
            target_mask.extend([0] * len(instruction_tokenized))

            target_tokenized = self.tokenizer.encode(targets[i])#[1:]
            target_tokenized.append(eos_token_id)
            input_ids.extend(target_tokenized)

            target_mask.extend([1] * len(target_tokenized))

        return input_ids, target_mask



def get_data(args, image_processor, tokenizer, logger, epoch=0):
    dataset = InstructionDataset(
        config_json_path=args.instruction_data,
        image_processor=image_processor,
        tokenizer=tokenizer,
        num_samples=args.train_num_samples,
        max_length=args.max_length,
        logger=logger,
        args=args
        )
    sampler = DistributedSampler(dataset) if args.world_size>0 else None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
        pin_memory=True,
    )

    dataloader.num_batches = len(dataloader)
    dataloader.num_samples = len(dataset)

    return DataInfo(dataloader=dataloader, shared_epoch=0)

