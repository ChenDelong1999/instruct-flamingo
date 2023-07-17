export HF_DATASETS_CACHE="/cpfs/user/chendelong/.cache/"
export TRANSFORMERS_CACHE="/cpfs/user/chendelong/.cache/"
echo 'activating virtual environment'
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate instruct_flamingo
which python

CUDA_VISIBLE_DEVICES='1' python open_flamingo/instruction_tuning/instruction_dataset_inference.py \
    --lm_path "/cpfs/user/chendelong/cache/mpt-7b" \
    --vision_encoder_path "ViT-L-14-336" \
    --vision_encoder_pretrained "openai" \
    --tuning_config 'open_flamingo/instruction_tuning/tuning_config/lora[lm+xqttn]+perceiver.json' \
    --checkpoint_paths '/cpfs/user/chendelong/cache/OpenFlamingo-9B-vitl-mpt7b.pt,/cpfs/user/chendelong/open_flamingo_v2/runs/0713-clever_flamingo_v2_9b-2k_context-80G/checkpoint_4.pt'  \
    --cross_attn_every_n_layers 4 \
    --instruction_path '/cpfs/user/chendelong/open_flamingo/datasets/validation_selected_for_v2.json' \
    --instruction_prompt_templete 'guanaco-no-prompt' \
    --num_samples -1 \
    --max_new_token 1024 \
    --no_repeat_ngram_size 3 \
    --num_beams 1 \
    --seed 42 \
    --results_dir "predictions_validation/"
