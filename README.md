
<div align="center">

<img src="docs/flamingo.png" alt="Logo" width="100">

### Instruct-Flamingo
#### Codebase and Fondation Models for Visual Instruction Tuning

</div>

## News
- **2023/08/18**: We have released the **Clever Flamingo v2** model! It was trained on a collection of 20 million instruction samples sourced from 100+ visual and textual datasets. The web demo at [clever_flamingo.xiaoice.com](http://clever_flamingo.xiaoice.com/) has been upgraded to the v2 version. Feel free to chat with Clever Flamingo v2!


## Clever Flamingo v2

Clever Flamingo v2 is an instruction-following multi-modal LLM fine-tuned from [OpenFlamingo-v2 (MPT-7B)](https://huggingface.co/openflamingo/OpenFlamingo-9B-vitl-mpt7b). It accepts arbitrarily interleaved image-text instructions as input and generates text responses. It is an upgraded version of the [Clever Flamingo v1](https://github.com/ChenDelong1999/polite-flamingo) model. The key features of Clever Flamingo v2 include:

**Large Instruction Corpus**: The Flamingo model's "<Image(s) + Text $\rightarrow$ Text>" approach allows us to unify a large amount of datasets (+100) into a standard input-output format. These datasets include [PF-1M](https://github.com/ChenDelong1999/polite-flamingo#-the-pf-1m-dataset), [MIMIC-IT](https://github.com/Luodian/Otter/blob/main/mimic-it/README.md), [SVIT](https://github.com/BAAI-DCAI/Visual-Instruction-Tuning), [LAMM](https://github.com/OpenLAMM/LAMM), [FunQA](https://github.com/Jingkang50/FunQA), [OpenORCA](https://huggingface.co/datasets/Open-Orca/OpenOrca), [WizardLM-Evol](https://github.com/nlpxucan/WizardLM), [CAMEL](https://github.com/camel-ai/camel), [Alpaca-CoT](https://github.com/PhoebusSi/Alpaca-CoT), and many more. This leads to the largest scale experiment on visual instruction tuning to our best knowledge. Its training process takes one week on an 8 $\times$ A100 (80G) machine.

![Number of Samples](docs/num_sample_bar.png)

**Balanced Visual-Textual Instruction Tuning**: Unlike previous approaches to visual instruction tuning that sometimes pay less attention on the usage of textual instruction data, Clever Flamingo v2 maintains a strict 1:1 ratio when sampling visual and textual instructions. This ensures accurate visual understanding and strong instruction-following abilities.

![Dataset Ratio](docs/dataset_pie.png)

**Long Context and LangChain**: Clever Flamingo utilizes a context window of 2k tokens during training, enabling it to fully leverage high-quality instruction datasets. Despite being based on a 7B LLM, Clever Flamingo v2 demonstrates strong instruction-following capabilities, long response generation, and chain-of-thought capabilities. It is integrated into the [LangChain](https://github.com/langchain-ai/langchain) framework, allowing for easy use of [pre-built chains](https://python.langchain.com/docs/use_cases). Additionally, several chains for multi-modal scenarios, such as [CoT-SC](https://arxiv.org/abs/2203.11171) and Caption-Self-Verification Chain, have been implemented. Please see [langchain.ipynb](inference/langchain.ipynb) for an example.

![LangChain](docs/cot.png)

**From Single Image to Multi-images, Region Crops, and Videos**: Given that Flamingo models accept arbitrarily interleaved image-text sequences as input, Clever Flamingo v2 has been trained on a diverse instruction dataset that contains various image-text combinations. We integrate tasks involving multi-image comparison, reasoning, region understanding (based on cropped boxes), and video understanding (based on sampled frames). During training, it can process a maximum of 16 images per context window, compared to 5 in OpenFlamingo pretraining.

![Image-Text Interleaving](docs/interleaving_img_text.png)

**OCR Integrations**: Clever Flamingo v2 incorporates OCR results (from [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)) into instructions of OCR-related datasets (TextCaps, InfographicQA, VisualMRC, OCR-VQA, etc.) during training. OCR results with similar vertical pixel coordinates are merged in the same line and concatenated by `'\t'`, preserving layout information. The OCR process is efficient, taking less than 0.2 seconds to process each image.

![OCR](docs/ocr.png)


<!-- - **Easy and Efficient Fine-tuning**.  -->

## About This Codebase

> This repository aims to provide an easy-to-use codebase and foundation models for (instruction-)finetuning of multi-modal LLMs. It is built upon [OpenFlamingo](https://github.com/mlfoundations/open_flamingo)🦩 codes and [OpenFlamingo-v2](https://laion.ai/blog/open-flamingo-v2/) models, which are powerful vision-language foundation models trained on massive interleaved image-text data. Key features of this codebase include:

- **Data**: We use a unified input-output structure to format fine-tuning data. Unlike OpenFlamingo, which only supports [webdataset](https://github.com/webdataset/webdataset), our codebase allows fine-tuning data to be stored locally as `.json` files. Multiple datasets can be mixed using dataset configurations with specified sampling ratios. We also provide scripts for converting existing datasets into the `.json` format.

- **Model**: We have added LoRA adapter to both the language model and cross-attention layers for efficient fine-tuning. In the near future, we will release a stronger visual instruction foundation model (Clever Flamingo v2), which has been extensively fine-tuned on a diverse collection of instruction datasets (both text-only and multi-modal).

- **Training**: We have implemented multi-turn augmentation (see our [paper](https://arxiv.org/abs/2307.01003)) to boost training efficiency. We have also integrated TensorBoard logging and file logging for easier debugging.

- **Inference**: We have wrapped the model into an `inferencer` and provided code for hosting a local API, hosting a Gradio web demo.

> *This is a ongoing project. We are working on verifying codes and training better instruction foundation models.*

## Getting Started🚩

### 1. Install Dependencies

First, clone this repo:

```bash
git clone https://github.com/ChenDelong1999/instruct_flamingo.git
```

Our code is developed upon [OpenFlamingo](https://github.com/mlfoundations/open_flamingo), and therefore inherits its environment dependencies. One can use an OpenFlamingo environment to run our code, or create one by:

```bash
conda env create -f environment.yml
```

Note: please avoid using environment with pip installed `open-flamingo` package to avoid import conflicts.

Additionally, as in our method LoRA adapter need to be inserted to the language model, a [PEFT](https://github.com/huggingface/peft) installation is required. Tensorboard should also be installed for logging.

```bash
pip install peft, tensorboard
```

The following packages are the dependencies of hosting API and gradio web demo:

```bash
pip install gradio, uvicorn, fastapi, pydantic
```

### 2. Download Pretrained Weights

- [MPT-7B](https://huggingface.co/anas-awadalla/mpt-7b)
- [Clever Flamingo v2](https://huggingface.co/chendelong/clever_flamingo_v2/blob/main/clever_flamingo_v2_9b_0818.pt) (New!🔥)
- [OpenFlamingo9B-v2](https://huggingface.co/openflamingo/OpenFlamingo-9B-vitl-mpt7b)


## Model Inference🎈

### Hosting Local API and Web Demo

We suggest to host a local API then host a local [gradio](https://www.gradio.app/) web demo, such that the front-end and back-end is seperated (easier to debug, since re-loading LLM is slow), and the local API could make model inference and evaluations much convinient. You can start an API server via the following command. Please see `api.py` and make necessary changes (e.g., model checkpoint caching path).

```bash
CUDA_VISIBLE_DEVICES=0 uvicorn api:app --host=0.0.0.0 --port=1234 --log-level=info
```

This API can be called by the following code:

```python
import json, request

url = '0.0.0.0:1234/clever_flamingo'
content_lst = {
    # remenber to add '<image>' to your instruction to indecate the location of image(s)
    'prompt': f'### Human: {<YOUR INSTRUCTION>}\n### Assistant: ',     
    'imgpaths': ['<YOUR IMAGE 1>', '<YOUR IMAGE 2>'],
    'args':{
        'max_new_token':1024,
        'num_beams':1,
        'temperature':1.0,
        'top_k':20,
        'top_p':1,
        'do_sample':True,
        'length_penalty':1.0,
        'no_repeat_ngram_size':3,
    }
}
d = {"content_lst": content_lst,'typ': 'None'}
d = json.dumps(d).encode('utf8')
r = requests.post(url, data=d)
js = json.loads(r.text)

print(js['result']['response'])
```

Now you can start the gradio web demo, make sure you have checked the configrations in `gradio_demo.py`.

```bash
python gradio_demo.py
```



## Prepare Fine-tuning Dataset(s)📜

Training samples are expected to be provided by `.json` files, where each file has the following structure:

```json
[
  {
    "input": "An Instruction or a question. Image path(s) (either absolute or relative) can be interleaved here as <img_path>path/to/the/image.png<img_path>, there can be more than one images: <img_path>path/to/the/second/image.png<img_path>",
    "output": "Expected response or answer. The language modelling loss only operate on this part, and it contains text only."
  },
  {
    "input": "This input-output format can be applied to many kinds of datasets, such as captioning ('input' filed can be leaved blank or as 'Describe this image'), VQA, multi-image reasoning, and also text-only instruction datasets.",
    "output": "The output field must be not empty."
  }
]
```

In the `instruction_dataset` folder, we provide some scripts for converting existing datasets into this format.

The path of this `.json` dataset can be fed into training by `--instruction_data='path/to/dataset.json'`. Additionally, multiple datasets can be mixed by creating a dataset config file, which structures as follows:

```json
[
  {
    "dataset_name": "llava-complex-reasoning-77k",
    "json_path": "instruction_dataset/converted_datasets/llava/complex_reasoning_77k.json",
    "img_dir": "",
    "ratio": 77
  },
  {
    "dataset_name": "sharegpt",
    "json_path": "instruction_dataset/converted_datasets/sharegpt/sharegpt.json",
    "img_dir": "",
    "ratio": 45
  }
]
```

Here `img_dir` is the path to image dictionary if image paths are provided as relative path. The `ratio` specifies the sampling ratio of each subsets. Using `--instruction_data='path/to/dataset_config.json'` to feed the config for training.

**Notes on dataset sampling**: the following arguments of `instruction_tuning/train.py` controls how the dataset is sampled during training
- `--train_num_samples`: how many samples are randomly sampled to form a single epoch training data. Set it to `-1` to disable dataset sampling and use all available data.
- `--dataset_sampling_mode`: choices are `ratio` and `sqrt`. The `ratio` alternative sample training data according to the `ratio` field specified in dataset configs; the `sqrt` samnpling mode is introduced in the [InstructBLIP paper](https://arxiv.org/abs/2305.06500). 
- `--multiturn_augmentation`: this is introduced in the [Polite Flamingo paper](https://arxiv.org/abs/2307.01003). As the text length of instructions are extreamly imbalanced, we randomly sample instructions from the dataset to fill up the `<PAD>` token as thus boost training efficiency. It also empower the model to have multi-image multi-turn conversation ability. Set `--multiturn_augmentation=0` to disable this augmentation.
- `--max_img`: maximum number of images, sample with fewer images will be automatically pad with black image(s). When `--multiturn_augmentation`>0 is specified, when the total number of image reachers `--max_img`, the augmentation will be truncated (to avoid encouraging visual hallucination). OpenFlamingo models uses `--max_img=5` during pretraining.
- `--max_length`: maximum number text token. Use smaller value when GPU memory is insufficient.
- `--instruction_prompt_templete`: `guanaco` or `guanaco-no-prompt`. The following code shows how it format the prompt (the target output will be appended to the prompt):
  ```python
  def get_prompt_instruction(instruction, instruction_prompt_templete):
    if instruction_prompt_templete == 'guanaco':
      prompt = 'A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions.\n'
      prompt +=  f'### Human: {instruction}\n### Assistant: '
      return prompt
    elif instruction_prompt_templete == 'guanaco-no-prompt':
      return f'### Human: {instruction}\n### Assistant: '
  ```

## Training🔥

First, `tuning_config` should be specified. This config controls which group of parameters will have LoRA adapters, and which group of parameters will be unfreezed. In the following example (open_flamingo/instruction_tuning/tuning_config/lora+perceiver.json), LoRA adapter with a rank of 64 will be applied to MPT language models (not including cross-attention layers), and the perceiver resampler will be unfreezed.

```json
{
    "lora": true,
    "from_pretrained": false,
    "lora_target_modules": ["Wqkv", "out_proj", "up_proj", "down_proj"],
    "lora_r": 64,
    "lora_alpha": 64,
    "lora_dropout": 0.0,
    "unfrozen": ["perceiver"]
}
```

Set `"lora": false` to skip adding LoRA adapter to any model parameters. The `"from_pretrained"` field is only useful for Polite Flamingo and Clever Flamingo (v1) models, as they use [Guanaco QLoRA on LLaMA-7B](https://huggingface.co/timdettmers/guanaco-7b) as initialization.

The following is an example of starting instruction tuning on OpenFlamingo-9B-v2, this setting comsumes 62GB memory on each GPU. One can lower the `--max_length` and `--batch_size`, or seting fewer parameters to be unfrozen in `--tuning_config` to save memory.

```bash
export PYTHONPATH="$PYTHONPATH:open_flamingo"
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' torchrun --nnodes=1 --nproc_per_node=8 --master_port=29502 open_flamingo/instruction_tuning/train.py \
    --instruction_data 'instruction_dataset/configs/datasets.json' \
    --instruction_prompt_templete 'guanaco-no-prompt' \
    --run_name 'runs/0709-clever_flamingo_v2-8x80g-2k_context' \
    --seed 42 \
    --vision_encoder_path 'ViT-L-14-336' \
    --lm_path 'anas-awadalla/mpt-7b' \
    --tokenizer_path 'anas-awadalla/mpt-7b' \
    --freeze_lm_embeddings \
    --tuning_config 'open_flamingo/instruction_tuning/tuning_config/lora[lm+xqttn]+perceiver.json' \
    --resume_from_checkpoint '/path/to/cached/OpenFlamingo-9B-vitl-mpt7b.pt' \
    --max_length 2048 \
    --multiturn_augmentation 32 \
    --max_img 16 \
    --cross_attn_every_n_layers 4 \
    --batch_size 2 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 4 \
    --precision 'bf16' \
    --train_num_samples 100000 \
    --workers 32 \
    --num_epochs 100 \
    --lr_scheduler constant \
    --warmup_steps 1000 \
    --logging_steps 500
```

The `--resume_from_checkpoint` specify the pretrained weights to load. Multiple checkpoints (e.g., when using visual instruction foundation model) can be concatenated with a seperation of `','`, and the model will load them one by one.


## Acknowledgements🙏

This codebase is built upon [OpenFlamingo](https://github.com/mlfoundations/open_flamingo). Implementation of PEFT tuning config is inspired by [Multimodal-GPT](https://github.com/open-mmlab/Multimodal-GPT). Thanks for their wonderful works.


This project is under active development, feel free to raise an issue if there are any bugs, we will try to fix them as soon as posible!

If you find this project useful, please consider cite the following paper:

```bibtex
@article{chen2023visual,
  title={Visual Instruction Tuning with Polite Flamingo},
  author={Chen, Delong and Liu, Jianfeng and Dai, Wenliang and Wang, Baoyuan},
  journal={arXiv preprint arXiv:2307.01003},
  year={2023}
}
```
