import argparse
import json
import os
import random
import numpy as np
import torch
from tqdm import tqdm
import logging


import os
import json
from collections import defaultdict
from rouge_score import rouge_scorer
from data import InstructionDataset
from inferencer import Inferencer

parser = argparse.ArgumentParser()

# Build Inferencer - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
parser.add_argument("--lm_path", type=str, default="facebook/opt-1.3b")
parser.add_argument("--vision_encoder_path", default="ViT-L-14", type=str)
parser.add_argument("--vision_encoder_pretrained", default="openai", type=str)
parser.add_argument("--tuning_config", default=None, type=str)
parser.add_argument("--checkpoint_paths", type=str, default=None)
parser.add_argument(
    "--cross_attn_every_n_layers",
    type=int,
    default=4,
    help="how often to add a cross-attention layer after each transformer layer",
)
parser.add_argument("--v1", action="store_true", default=False)

# Language Generation Configs - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
parser.add_argument("--max_new_token", type=int, default=128)
parser.add_argument("--num_beams", type=int, default=1)
parser.add_argument("--temperature", type=float, default=1)
parser.add_argument("--top_k", type=float, default=20)
parser.add_argument("--top_p", type=float, default=1)
parser.add_argument("--do_sample", type=bool, default=True)
parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
parser.add_argument("--length_penalty", type=float, default=1)
parser.add_argument("--max_length", type=int, default=1024)

# Dataset Configs - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
parser.add_argument(
    "--results_dir", type=str, default=None, help="JSON file to save results"
)
parser.add_argument(
    "--instruction_path",
    type=str,
    help="Path to the instruction dataset directory.",
    default=None,
)
parser.add_argument(
    "--img_dir",
    type=str,
    help="Path to the instruction dataset images.",
    default=None,
)
parser.add_argument("--instruction_prompt_templete", type=str, default='guanaco')
parser.add_argument("--dataset_sampling_mode", type=str, default='ratio')
parser.add_argument("--num_samples", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)


def calculate_metrics(references, hypotheses, logger):
    logger.info("Calculating ROUGE-L score...")
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = [rouge.score(ref, hyp)["rougeL"].fmeasure for ref, hyp in zip(references, hypotheses)]
    rouge_l_score = sum(rouge_scores) / len(rouge_scores)

    return rouge_l_score
    

def add_image_dir(str, img_dir):
    if img_dir == '':
        return str
    else:
        img_path_count = 0
        index = 0
        while index < len(str):
            if str.startswith('<img_path>', index):
                img_path_count += 1
                if img_path_count % 2 == 1:  # Check if it's an odd-numbered <img_path>
                    str = str[:index] + '<img_path>' + img_dir + '/' + str[index + len('<img_path>'):]
            index += 1
        return str
    
def save_results(results, args, logger):
    if not os.path.exists(args.results_dir):
        logger.info(f"Creating results directory at {args.results_dir}")
        os.makedirs(args.results_dir)
    all_results = []
    for dataset_name, dataset_results in results.items():
        logger.info(f"Saving results for {dataset_name} to file {args.results_dir}/{dataset_name}.json")
        all_results.extend(dataset_results)
        with open(os.path.join(args.results_dir, f"{dataset_name}.json"), "w") as f:
            json.dump(dataset_results, f, indent=4)
    # with open(os.path.join(args.results_dir, f"all_results.json"), "w") as f:  
    #     json.dump(all_results, f, indent=4)


def save_summary(results, args, logger):
    summary = defaultdict(dict)
    header = "| Dataset | Avg ROUGE-L |Avg target length | Avg prediction length | Exact matches | Match percentage | Total samples |\n|---------|---------- |-----------------|---------------------|---------------|-----------------|---------------|\n"

    for dataset_name, dataset_results in results.items():
        references = [result["target"] for result in dataset_results]
        hypotheses = [result["output"] for result in dataset_results]

        logger.info(f"Calculating metrics for {dataset_name}...")
        rouge_l_score = calculate_metrics(references, hypotheses, logger)

        target_lengths = [len(target.split()) for target in references]
        prediction_lengths = [len(prediction.split()) for prediction in hypotheses]

        exact_matches = sum([target == prediction for target, prediction in zip(references, hypotheses)])

        summary[dataset_name] = {
            "avg_rouge_l": rouge_l_score,
            "avg_target_length": sum(target_lengths) / len(target_lengths),
            "avg_prediction_length": sum(prediction_lengths) / len(prediction_lengths),
            "exact_matches": exact_matches,
            "match_percentage": exact_matches / len(references) * 100,
            "total_samples": len(references),
        }

        header += f"| {dataset_name} | {rouge_l_score:.2f} | {summary[dataset_name]['avg_target_length']:.2f} | {summary[dataset_name]['avg_prediction_length']:.2f} | {exact_matches} | {summary[dataset_name]['match_percentage']:.2f}% | {len(references)} |\n"

    with open(os.path.join(args.results_dir, "summary.md"), "w") as f:
        f.write(header)


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    logger = logging.getLogger(__name__)

    args = parser.parse_args()
    args.rank = 0
    def random_seed(seed=42, rank=0):
        torch.manual_seed(seed + rank)
        np.random.seed(seed + rank)
        random.seed(seed + rank)
    random_seed(args.seed)

    if args.checkpoint_paths is not None:
        args.ckpt_basename = os.path.dirname(args.checkpoint_paths).split('/')[-1] + '-' + os.path.basename(args.checkpoint_paths).replace('.pt', '')
    else:
        args.ckpt_basename = 'no_checkpoint'
    args.results_dir = os.path.join(args.results_dir, args.ckpt_basename, f'{os.path.basename(args.instruction_path).replace(".json", "")}_{args.num_samples}')
    logger.info('args '+'-'*100)
    for key, value in args.__dict__.items():
        logger.info("\t{:<30}\t{}".format(key+":", value))
    logger.info('-'*100)
    
    # ------------------------------------
    # Load Model and Checkpoints
    # ------------------------------------

    inferencer = Inferencer(
        lm_path=args.lm_path,
        checkpoint_paths=args.checkpoint_paths,
        tuning_config=args.tuning_config,
        clip_vision_encoder_path=args.vision_encoder_path,
        clip_vision_encoder_pretrained=args.vision_encoder_pretrained,
        cross_attn_every_n_layers=args.cross_attn_every_n_layers,
        v1=args.v1
        )

    # ------------------------------------
    # ↓ For debugging only
    # ------------------------------------
    # from transformers import LlamaTokenizer
    # tokenizer = LlamaTokenizer.from_pretrained('decapoda-research/llama-7b-hf', use_fast=False)
    # tokenizer.add_special_tokens(
    #         {"additional_special_tokens": ["<|endofchunk|>", "<image>"]}
    #     )
    # print(tokenizer.eos_token_id)
    # from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
    # def _convert_to_rgb(image):
    #     return image.convert('RGB')
    # image_processor = Compose([
    #     Resize(256),
    #     _convert_to_rgb,
    #     CenterCrop(224),
    #     ToTensor(),
    #     Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])

    # ------------------------------------
    # Load Datasets and Runs Inference
    # ------------------------------------

    dataset = InstructionDataset(
        config_json_path=args.instruction_path,
        image_processor=inferencer.image_processor,
        tokenizer=inferencer.tokenizer,
        num_samples=args.num_samples,
        max_length=None,
        logger=logger,
        img_dir=args.img_dir,
        args=args,
        mode='test'
        )

    dataset_names = []
    dataset_results = defaultdict(list)

    for index, item in enumerate(tqdm(dataset)):            
        images, text, instruction_str, sample = item
        input_ids = text['input_ids']
        attention_mask = text['attention_mask']
        
        images = images.cuda().half()
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda() 

        prediction, full_text = inferencer(
            prompt=text,
            images=images,
            max_new_token=args.max_new_token,
            num_beams=args.num_beams,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            do_sample=args.do_sample,
            length_penalty=args.length_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            response_split="### Assistant:"
        )

        dataset_name = dataset.configs[sample['dataset_idx']]['dataset_name']
        instruction_str = instruction_str.replace('<|endofchunk|>', '')
        dataset_results[dataset_name].append({
            "input": add_image_dir(sample['instruction']+sample['input'], sample['img_dir']),
            "output": prediction,
            "target": sample['output'],
            "prompt": instruction_str,
        })
        dataset_names.append(dataset_name)
        print('-'*64)
        print(f'[dataset]:   {dataset_name} ({sample["dataset_idx"] + 1}/{len(dataset.configs)})')
        print(f'[prompt]:    {instruction_str}')
        print(f'\n*** PREDICTION ***\n{prediction}')
        print(f'\n*** TARGET ***\n{sample["output"]}')

        if index % 10 == 0:
            save_results(dataset_results, args, logger)

    save_results(dataset_results, args, logger)
    save_summary(dataset_results, args, logger)


if __name__ == "__main__":
    main()
