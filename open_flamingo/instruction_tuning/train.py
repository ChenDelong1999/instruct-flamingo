""" Main training script """

import argparse
import glob
import os
import random
import logging
import numpy as np
import torch
import wandb
from data import get_data
from distributed import init_distributed_device, world_info_from_env
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.tensorboard import SummaryWriter
from train_utils import (
    train_one_epoch,
    get_cast_dtype,
    get_mp_policy_dtype,
    save_checkpoint,
    get_params_count_summary,
    prepare_model_for_tuning,
    resume_from_checkpoints
)
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from torch.distributed.fsdp import (
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointWrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from torch.distributed.fsdp._init_utils import _init_intra_and_inter_node_groups
from torch.distributed.distributed_c10d import _get_default_group
import functools

from open_flamingo import create_model_and_transforms


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def main():
    parser = argparse.ArgumentParser()

    # visual instruction tuning args
    parser.add_argument(
        "--instruction_data",
        type=str,
        help="path to instruction tuning data (Aplaca json format)",
    )
    parser.add_argument("--instruction_prompt_templete", type=str, default='guanaco')
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--multiturn_augmentation", type=int, default=1)
    parser.add_argument("--max_img", type=int, default=5)
    parser.add_argument("--dataset_sampling_mode", type=str, default='ratio')
    parser.add_argument("--continue_training", action="store_true")
    parser.add_argument("--skip_check_overlength", action="store_true")
    parser.add_argument("--tuning_config", type=str, default='open_flamingo/instruction_tuning/tuning_config/lora.json')
    parser.add_argument("--epoch_num_samples", type=int, default=-1)

    # model configuration args
    parser.add_argument("--vision_encoder_path", default="ViT-L-14", type=str)
    parser.add_argument("--vision_encoder_pretrained", default="openai", type=str)
    parser.add_argument("--lm_path", default="facebook/opt-1.3b", type=str)
    parser.add_argument(
        "--tokenizer_path",
        default="facebook/opt-30b",
        type=str,
        help="path to tokenizer",
    )
    parser.add_argument(
        "--cross_attn_every_n_layers",
        type=int,
        default=1,
        help="how often to add a cross-attention layer after each transformer layer",
    )

    # training args
    parser.add_argument(
        "--run_name",
        type=str,
        default="openflamingo3B",
        help="used to name saving directory and wandb run",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        help="path to checkpoint to resume from, this should contain model, optimizer, and lr_scheduler states. if there exists a checkpoint in the dir named run_name, we will resume from that checkpoint by default",
        default=None,
    )
    parser.add_argument(
        "--delete_previous_checkpoint",
        action="store_true",
        help="delete previous checkpoint when saving new checkpoint",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument(
        "--lr_scheduler",
        default="constant",
        type=str,
        help="constant, linear, or cosine",
    )
    parser.add_argument("--loss_multiplier_mmc4", type=float, default=1.0)
    parser.add_argument("--loss_multiplier_laion", type=float, default=1.0)
    parser.add_argument("--warmup_steps", default=5000, type=int)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument(
        "--precision",
        choices=["amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
        default="fp32",
        help="Floating point precision.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="whether to train with gradient/activation checkpointing",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="we define an 'epoch' as a fixed number of examples (train_num_samples_mmc4, train_num_samples_laion), not a pass through the entire dataset",
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--freeze_lm_embeddings",
        action="store_true",
        help="if True, we freeze the LM embeddings during training. Otherwise, we train the <image> and <|endofchunk|> embeddings.",
    )
    parser.add_argument(
        "--logging_steps", type=int, default=100, help="log loss every n steps"
    )

    # data args
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--train_num_samples", type=int, default=10000)

    # distributed training args
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
    )
    parser.add_argument(
        "--fsdp",
        default=False,
        action="store_true",
        help="Use FullyShardedDataParallel for distributed training.",
    )
    parser.add_argument(
        "--fsdp_use_orig_params",
        default=False,
        action="store_true",
        help="Passed into the FSDP constructor. Enables param_groups and gradient masking for weight_decay. Does not work with OPT.",
    )
    parser.add_argument(
        "--fsdp_sharding_strategy", default="full", type=str, choices=["full", "hybrid"]
    )

    # wandb args
    parser.add_argument("--report_to_wandb", default=False, action="store_true")
    parser.add_argument(
        "--wandb_project",
        type=str,
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
    )
    parser.add_argument(
        "--save_checkpoints_to_wandb",
        default=False,
        action="store_true",
        help="save checkpoints to wandb",
    )

    args = parser.parse_args()

    # Validate args
    if args.save_checkpoints_to_wandb and not args.report_to_wandb:
        raise ValueError("save_checkpoints_to_wandb requires report_to_wandb")

    if args.fsdp and not args.fsdp_use_orig_params:
        print(
            "Warning: FSDP is running without fsdp_use_orig_params flag. "
            + "This is not recommended because it means we will use uniform weight decay"
            + " and train all embeddings, not just the newly added ones. "
            + "Note: OPT models are not compatible with fsdp_use_orig_params flag."
        )

    if args.fsdp and args.fsdp_sharding_strategy == "hybrid":
        print(
            "Warning: As of torch=2.0.1, the FSDP logic for optim_state_dict() is broken for hybrid sharding."
            + "To make this method work, we need to modify torch.distributed.fsdp._optim_utils.py"
            + "Copy and paste the code from the _optim_utils.py in this repo into the torch file."
            + "The main issue was the missing group kwarg on line 1596 in _all_gather_optim_state."
        )

    # Set up distributed training
    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    device_id = init_distributed_device(args)
    random_seed(args.seed)

    # Set up logging
    if not os.path.exists(args.run_name):
        os.makedirs(args.run_name, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    logger = logging.getLogger(__name__)
    file_handler = logging.FileHandler(os.path.join(args.run_name, 'train.log'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(module)s:%(lineno)d] %(message)s'))
    logger.addHandler(file_handler)

    # Initialize model
    model, image_processor, tokenizer = create_model_and_transforms(
        args.vision_encoder_path,
        args.vision_encoder_pretrained,
        args.lm_path,
        args.tokenizer_path if args.tokenizer_path else args.lm_path,
        cross_attn_every_n_layers=args.cross_attn_every_n_layers,
        use_local_files=args.offline,
        gradient_checkpointing=args.gradient_checkpointing,
        freeze_lm_embeddings=args.freeze_lm_embeddings,
    )

    # TODO: add comments to explain loading checkpoints twice
    logger.info('loading checkpoints before PEFT')
    if args.resume_from_checkpoint is not None:
        model, checkpoint, resume_from_epoch, message = resume_from_checkpoints(model, args.resume_from_checkpoint, args, logger)

    model, config = prepare_model_for_tuning(model, args.tuning_config)

    if (config['from_pretrained'] or config['lora']) and args.resume_from_checkpoint is not None:
        logger.info('loading checkpoints after PEFT')
        model, checkpoint, resume_from_epoch, message = resume_from_checkpoints(model, args.resume_from_checkpoint, args, logger)

    if args.rank==0:
        logger.info(get_params_count_summary(model))
        logger.info('args')
        for key, value in args.__dict__.items():
            logger.info("\t{:<30}\t{}".format(key+":", value))
        
        logger.info('Tuning config')
        logger.info(config)

        logger.info('model.lang_encoder.config')
        logger.info(model.lang_encoder.config)

    random_seed(args.seed, args.rank)

    # Initialize logging
    print(f"Start running training on rank {args.rank}.")
    if args.rank == 0 and args.report_to_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
        )
    if args.rank == 0:
        os.makedirs(os.path.join(args.run_name, 'tensorboard'), exist_ok=True)
        tensorboard_writer = SummaryWriter(os.path.join(args.run_name, 'tensorboard'))
    else:
        tensorboard_writer = None


    # Load model checkpoint on CPU
    # if os.path.exists(f"{args.run_name}") and args.resume_from_checkpoint is None:
    #     # if args do not specify a checkpoint to resume from, check if checkpoints exist for this run
    #     # and automatically resume from the latest checkpoint
    #     checkpoint_list = glob.glob(f"{args.run_name}/checkpoint_*.pt")
    #     if len(checkpoint_list) == 0:
    #         print(f"Found no checkpoints for run {args.run_name}.")
    #     else:
    #         args.resume_from_checkpoint = sorted(
    #             checkpoint_list, key=lambda x: int(x.split("_")[-1].split(".")[0])
    #         )[-1]
    #         print(
    #             f"Found checkpoint {args.resume_from_checkpoint} for run {args.run_name}."
    #         )
    # model, checkpoint, resume_from_epoch, message = resume_from_checkpoints(model, args.resume_from_checkpoint, args, logger)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # resume_from_epoch = 0
    # if args.resume_from_checkpoint is not None:
    #     if args.rank == 0:
    #         print(f"Loading checkpoint from {args.resume_from_checkpoint}")
    #     checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
    #     msd = checkpoint["model_state_dict"]
    #     msd = {k.replace("module.", ""): v for k, v in msd.items()}

    #     # for fsdp, only one rank needs to load the state dict
    #     if not args.fsdp or args.rank == 0:
    #         model.load_state_dict(msd, False)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # if args.resume_from_checkpoint is not None:
    #     for this_checkpoint in args.resume_from_checkpoint.split(','):
    #         if args.rank == 0:
    #             logger.info(f"Loading checkpoint from {this_checkpoint}")
    #         checkpoint = torch.load(this_checkpoint, map_location="cpu")
    #         if args.continue_training:
    #             resume_from_epoch = checkpoint["epoch"] + 1
    #         msd = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint.keys() else checkpoint
    #         if 'module' not in list(msd.keys())[0]:
    #             msd = {f"module.{k}": v for k, v in msd.items()}
    #             logger.info('Adding "module." prefix to checkpoints')
    #         # for fsdp, only one rank needs to load the state dict
    #         if not args.fsdp or args.rank == 0:
    #             model.load_state_dict(msd, False)

    # Initialize FSDP / DDP, and ensure the model is on GPU
    print(f"Initializing distributed training with {args.world_size} GPUs.")
    if args.fsdp:
        print(
            f"Before FSDP parameter num: {sum(p.numel() for p in model.parameters())} on rank {args.rank}"
        )

        # init MixedPrecision
        if args.precision != "fp32":
            cast_dtype = get_mp_policy_dtype(args.precision)
            mp_policy = MixedPrecision(
                param_dtype=torch.float32,
                reduce_dtype=cast_dtype,  # gradient communication
                buffer_dtype=cast_dtype,
            )
        else:
            mp_policy = None

        # init process groups
        if args.fsdp_sharding_strategy == "hybrid":
            intra_node_group, inter_node_group = _init_intra_and_inter_node_groups(
                _get_default_group()
            )
            args.my_group = intra_node_group  # for optimizer saving
            process_group = (intra_node_group, inter_node_group)  # for FSDP init
        else:
            args.my_group = None  # for optimizer saving
            process_group = None  # for FSDP init

        # init FSDP
        wrapper_kwargs = dict(
            process_group=process_group,
            cpu_offload=CPUOffload(offload_params=False),
            device_id=device_id,
            sync_module_states=True,  # broadcast loaded ckpt from rank 0 -> all ranks
            sharding_strategy=ShardingStrategy.FULL_SHARD
            if args.fsdp_sharding_strategy == "full"
            else ShardingStrategy.HYBRID_SHARD,
            use_orig_params=args.fsdp_use_orig_params,
            mixed_precision=mp_policy,
            forward_prefetch=True,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            limit_all_gathers=True,
        )
        model.wrap_fsdp(wrapper_kwargs, device_id)
        ddp_model = model

        print(
            f"After FSDP parameter num: {sum(p.numel() for p in model.parameters())} on rank {args.rank}"
        )
        print(
            f"After FSDP {torch.cuda.memory_allocated()/1024**3:.3} GB on rank {args.rank}"
        )

    else:
        if args.precision != "fp32":
            cast_dtype = get_cast_dtype(args.precision)
            model = model.to(cast_dtype)
        model = model.to(device_id)
        ddp_model = DDP(model, device_ids=[device_id])

    # Initialize gradient checkpointing
    if args.gradient_checkpointing:
        non_reentrant_wrapper = functools.partial(
            checkpoint_wrapper,
            offload_to_cpu=True,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        apply_activation_checkpointing(
            ddp_model,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda m: getattr(m, "_use_gradient_checkpointing", False)
            and not isinstance(m, FSDP)
            and not isinstance(m, CheckpointWrapper),
        )

    # Initialize optimizer
    params_to_optimize = ddp_model.named_parameters()
    params_to_optimize = list(
        filter(
            lambda x: x[1].requires_grad
            and not getattr(x[1], "exclude_from_optimizer", False),
            params_to_optimize,
        )
    )
    if not args.fsdp or args.fsdp_use_orig_params:
        # apply weight decay only to params in the xattn layers
        def get_grouped_params(model):
            params_with_wd, params_without_wd = [], []
            for n, p in params_to_optimize:
                if "gated_cross_attn" in n:
                    params_with_wd.append(p)
                else:
                    params_without_wd.append(p)
            return [
                {"params": params_with_wd, "weight_decay": args.weight_decay},
                {"params": params_without_wd, "weight_decay": 0.0},
            ]

        optimizer = torch.optim.AdamW(
            get_grouped_params(params_to_optimize), lr=args.learning_rate
        )
    else:
        # unclear if we should be using no weight decay or small weight decay for all parameters
        optimizer = torch.optim.AdamW(
            (p for _, p in params_to_optimize),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

    # load optimizer checkpoint
    if args.resume_from_checkpoint is not None and "optimizer_state_dict" in checkpoint and args.continue_training:
        osd = checkpoint["optimizer_state_dict"]
        if args.fsdp:
            osd = FSDP.optim_state_dict_to_load(osd, ddp_model, optimizer)
        optimizer.load_state_dict(osd)

    # Initialize data loaders
    instruction_dataset = get_data(args, image_processor, tokenizer, logger=logger, epoch=0)
    train_num_samples = len(instruction_dataset.dataloader.dataset) if args.train_num_samples==-1 else args.train_num_samples
    total_training_steps = (
        (train_num_samples) // (args.batch_size * args.world_size)
    ) * args.num_epochs
    if args.rank == 0:
        print(f"Total training steps: {total_training_steps}")

    # Initialize lr scheduler
    if args.lr_scheduler == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_training_steps,
        )
    elif args.lr_scheduler == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_training_steps,
        )
    else:
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps
        )

    # load lr scheduler checkpoint
    if args.resume_from_checkpoint is not None and "lr_scheduler_state_dict" in checkpoint and args.continue_training:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

    # Start training!
    ddp_model.train()

    for epoch in range(resume_from_epoch, args.num_epochs):
        train_one_epoch(
            args=args,
            model=ddp_model,
            epoch=epoch,
            tokenizer=tokenizer,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            dataloader=instruction_dataset.dataloader,
            logger=logger,    
            device_id=device_id,
            tensorboard_writer=tensorboard_writer,
            wandb=wandb,
        )
        save_checkpoint(ddp_model, optimizer, lr_scheduler, epoch, args)
    
        # reinitialize dataset to sample different images
        if epoch != (args.num_epochs-1):
            instruction_dataset = get_data(args, image_processor, tokenizer, "instruction", epoch=epoch, logger=logger)

    # save final checkpoint
    save_checkpoint(ddp_model, optimizer, lr_scheduler, epoch, args)


if __name__ == "__main__":
    main()
