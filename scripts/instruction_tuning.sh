cd /cpfs/user/chendelong/open_flamingo_v2
export HF_DATASETS_CACHE="/cpfs/user/chendelong/.cache/"
export TRANSFORMERS_CACHE="/cpfs/user/chendelong/.cache/"
echo 'activating virtual environment'
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate instruct_flamingo
which python

export PYTHONPATH="$PYTHONPATH:open_flamingo"
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' torchrun --nnodes=1 --nproc_per_node=8 --master_port=29502 open_flamingo/instruction_tuning/train.py \
    --instruction_data 'instruction_dataset/configs/datasets.json' \
    --instruction_prompt_templete 'guanaco-no-prompt' \
    --run_name 'runs/0709-clever_flamingo_v2-8x80g-2k_context' \
    --seed 42 \
    --vision_encoder_path 'ViT-L-14-336' \
    --lm_path '/cpfs/user/chendelong/cache/mpt-7b' \
    --tokenizer_path '/cpfs/user/chendelong/cache/mpt-7b' \
    --freeze_lm_embeddings \
    --tuning_config '/cpfs/user/chendelong/open_flamingo_v2/open_flamingo/instruction_tuning/tuning_config/lora[lm+xqttn]+perceiver.json' \
    --resume_from_checkpoint '/cpfs/user/chendelong/cache/OpenFlamingo-9B-vitl-mpt7b.pt' \
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


