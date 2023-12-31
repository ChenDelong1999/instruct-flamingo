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
    --run_name 'runs/0717-clever_flamingo_v2_3b-2k_context-40G-resume-from-0716' \
    --seed 42 \
    --vision_encoder_path 'ViT-L-14-336' \
    --lm_path '/cpfs/user/chendelong/cache/mpt-1b-redpajama-200b-dolly' \
    --tokenizer_path '/cpfs/user/chendelong/cache/mpt-1b-redpajama-200b-dolly' \
    --cross_attn_every_n_layers 1 \
    --freeze_lm_embeddings \
    --tuning_config '/cpfs/user/chendelong/open_flamingo_v2/open_flamingo/instruction_tuning/tuning_config/lora[lm+xqttn]+perceiver.json' \
    --resume_from_checkpoint '/cpfs/user/chendelong/cache/OpenFlamingo-3B-vitl-mpt1b-langinstruct/checkpoint.pt,/cpfs/user/chendelong/open_flamingo_v2/runs/0716-clever_flamingo_v2_3b-2k_context-40G/checkpoint_4.pt' \
    --continue_training \
    --max_length 2048 \
    --multiturn_augmentation 32 \
    --max_img 16 \
    --skip_check_overlength \
    --train_num_samples 10000000 \
    --epoch_num_samples 500000 \
    --batch_size 2 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 4 \
    --precision 'bf16' \
    --workers 8 \
    --num_epochs 20 \
    --lr_scheduler constant \
    --warmup_steps 1000 \
    --logging_steps 500


