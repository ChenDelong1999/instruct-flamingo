import time
from contextlib import suppress
import torch
from tqdm import tqdm
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.api import FullOptimStateDictConfig
import torch.distributed as dist
import os
import wandb
from einops import rearrange
import copy
import json
from peft import LoraConfig, get_peft_model, PeftModel


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    return cast_dtype


def get_mp_policy_dtype(precision: str):
    if "bfloat16" in precision or "bf16" in precision:
        return torch.bfloat16
    elif precision == "fp16":
        return torch.float16
    else:
        return torch.float32


def get_autocast(precision, cache_enabled=True):
    if precision == "amp":
        return torch.cuda.amp.autocast(cache_enabled=cache_enabled)
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(
            dtype=torch.bfloat16, cache_enabled=cache_enabled
        )
    else:
        return suppress


def train_one_epoch(
    args,
    model,
    epoch,
    dataloader,
    tokenizer,
    optimizer,
    lr_scheduler,
    device_id,
    logger,
    tensorboard_writer,
    wandb,
):
    # setup loaders
    num_batches_per_epoch = dataloader.num_batches
    total_training_steps = num_batches_per_epoch * args.num_epochs

    autocast = get_autocast(
        args.precision, cache_enabled=(not args.fsdp)
    )  # if fsdp, disable cache to save memory
    cast_dtype = get_cast_dtype(args.precision)

    # setup model
    media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
        "input_ids"
    ][-1]
    model.train()

    # setup logging
    step_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    accumulated_loss = 0.0

    # loop through dataloader
    for num_steps, batch in tqdm(
        enumerate(dataloader),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch),
    ):
        data_time_m.update(time.time() - end)
        global_step = num_steps + epoch * num_batches_per_epoch

        #### FORWARD PASS ####
        images, text, target_mask, dataset_idxs_batch  = batch
        images = images.to(device_id, dtype=cast_dtype, non_blocking=True).unsqueeze(2)
        input_ids = torch.stack([x for x in text['input_ids']]).squeeze(1).to(device_id)
        attention_mask = torch.stack([x for x in text['attention_mask']]).squeeze(1).to(device_id)
        # change attention mask type to bool
        attention_mask = attention_mask.type(torch.bool)       

        # set up labels; language model is expected to handle shifting
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        labels[:, 0] = -100
        # only calculate loss on response tokens
        target_mask = target_mask.squeeze(1)
        labels[target_mask == 0] = -100


        for i in range(labels.shape[0]):
            # # remove loss for any token before the first <image> token
            # label_idx = 0
            # while (
            #     label_idx < labels.shape[1] and labels[i][label_idx] != media_token_id
            # ):
            #     labels[i][label_idx] = -100
            #     label_idx += 1

            # get index of all endofchunk tokens in the sequence
            endofchunk_idxs = torch.where(labels[i] == endofchunk_token_id)[0]
            for endofchunk_idx in endofchunk_idxs:
                token_idx = endofchunk_idx + 1
                while (
                    token_idx < labels.shape[1]
                    and labels[i][token_idx] != media_token_id
                ):
                    labels[i][token_idx] = -100
                    token_idx += 1

        labels[labels == media_token_id] = -100
        labels = labels.to(device_id)

        # gradient accumulation w/ fsdp cpu offloading requires a no_sync context manager
        with autocast():
            output = model(
                vision_x=images,
                lang_x=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss, logits = output[:2]

            # if loss is nan, skip this batch
            # this hack of skipping the batch is not FSDP-compatible
            if torch.isnan(loss):
                print("loss is nan, skipping this batch")
                print("input_ids: ", tokenizer.batch_decode(input_ids))
                print("labels: ", labels)
                print("images: ", images)
                optimizer.zero_grad(set_to_none=True)
                continue

        divided_loss = loss / args.gradient_accumulation_steps
        divided_loss.backward()
        accumulated_loss += divided_loss.item()

        if (not args.freeze_lm_embeddings) and (
            not args.fsdp or args.fsdp_use_orig_params
        ):
            # Mask gradients for input embeddings s.t. we only update the added tokens <image> and <|endofchunk|>
            if args.fsdp:
                embed_grad = model.lang_encoder.get_input_embeddings().weight.grad
            else:
                embed_grad = (
                    model.module.lang_encoder.get_input_embeddings().weight.grad
                )
            zero_mask = torch.zeros_like(embed_grad)
            zero_mask[media_token_id] = torch.ones_like(zero_mask[media_token_id])
            zero_mask[endofchunk_token_id] = torch.ones_like(
                zero_mask[endofchunk_token_id]
            )
            if args.fsdp:
                model.lang_encoder.get_input_embeddings().weight.grad = (
                    embed_grad * zero_mask
                )
            else:
                model.module.lang_encoder.get_input_embeddings().weight.grad = (
                    embed_grad * zero_mask
                )

        # clip gradient norm
        if args.fsdp:
            """
            The way we clip gradients with FSDP is different than the non-FSDP case,
            because during FSDP, gradient norms are computed over certain submodules,
            rather than the entire model.
            At least for OPT-125M, this didn't seem to make a difference in performance.
            """
            grad_norm = model.clip_grad_norm_(1.0)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # step optimizer and log
        if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (
            num_steps == num_batches_per_epoch - 1
        ):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            # step time and reset end outside of rank 0
            step_time_m.update(time.time() - end)
            end = time.time()

            average_loss_tensor = torch.tensor(accumulated_loss, device=args.device)
            dist.all_reduce(average_loss_tensor, op=dist.ReduceOp.SUM)
            average_loss = average_loss_tensor.item() / args.world_size
            accumulated_loss = 0.0

            # rank 0 logging
            if args.rank == 0:# and args.report_to_wandb:
                samples_per_second = (
                    args.gradient_accumulation_steps
                    * args.batch_size
                    * args.world_size
                    / step_time_m.val
                )
                samples_per_second_per_gpu = (
                    args.gradient_accumulation_steps
                    * args.batch_size
                    / step_time_m.val
                )
                tensorboard_writer.add_scalar('time/data_time', data_time_m.avg, global_step)
                tensorboard_writer.add_scalar('time/step_time_m', step_time_m.avg, global_step)
                tensorboard_writer.add_scalar('time/samples_per_second', samples_per_second, global_step)
                tensorboard_writer.add_scalar('time/samples_per_second_per_gpu', samples_per_second_per_gpu, global_step)
                # wandb.log(
                #     {
                #         "data_time": data_time_m.avg,
                #         "step_time": step_time_m.avg,
                #         "laion_samples_per_second": laion_samples_per_second,
                #         "laion_samples_per_second_per_gpu": laion_samples_per_second_per_gpu,
                #         "c4_samples_per_second": c4_samples_per_second,
                #         "c4_samples_per_second_per_gpu": c4_samples_per_second_per_gpu,
                #         "lr": optimizer.param_groups[0]["lr"],
                #     },
                #     commit=False,
                # )
                step_time_m.reset()
                data_time_m.reset()

                # wandb.log(
                #     {
                #         "loss_laion": loss_laion.item(),
                #         "global_step": global_step,
                #     },
                #     commit=False,
                # )
                # wandb.log(
                #     {"loss_mmc4": loss_mmc4.item(), "global_step": global_step},
                #     commit=True,
                # )
                tensorboard_writer.add_scalar('train/loss', average_loss, global_step)
                tensorboard_writer.add_scalar('train/lr', optimizer.param_groups[0]["lr"], global_step)
                tensorboard_writer.add_scalar('train/grad_norm', grad_norm.type(torch.float32), global_step)

        # Log loss to console
        if ((num_steps + 1) % args.logging_steps == 0) and args.rank == 0:
            
            input_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].squeeze().tolist())
            input_text = tokenizer.convert_tokens_to_string(input_tokens)
            print('[input_ids]')
            # print(input_ids[0])
            print(input_text)
            print('-'*128)
            
            labels_ = copy.deepcopy(labels)
            labels_[labels == -100] = 1
            mask_token = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens([1]))
            label_tokens = tokenizer.convert_ids_to_tokens(labels_[0].squeeze().tolist())
            label_text = tokenizer.convert_tokens_to_string(label_tokens).replace(mask_token, ' ')
            print('[labels]')
            # print(labels[0])
            print(label_text)
            print('-'*128)

            probs = torch.softmax(logits, dim=-1)
            predicted_token_indexes = torch.argmax(probs, dim=-1)
            predicted_token_indexes[labels == -100] = 1
            predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_token_indexes[0].squeeze().tolist())
            predicted_text = tokenizer.convert_tokens_to_string(predicted_tokens).replace(mask_token, ' ')
            print('[predicted]')
            # print(predicted_token_indexes[0])
            print(predicted_text)
            logger.info(
                f"Step {num_steps+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete. Loss: {loss.item():.7f}. LR: {optimizer.param_groups[0]['lr']:.7f}"
            )          


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def filter_state_dict_to_trainable(model, state_dict):
    """
    Remove non-trainable parameters from model state dict.
    Exception: Embeddings will not be removed, even if frozen.
    This is because we need the new <image> <|endofchunk|> tokens to
    be consistent across initializations.
    """
    for (
        name,
        p,
    ) in model.named_parameters():  # won't work for fsdp + use_orig_params=False
        if "fsdp" in name:
            continue
        if "embed" in name or isinstance(p, torch.nn.Embedding):
            continue
        if not p.requires_grad:
            name = name.replace("._checkpoint_wrapped_module", "")
            if name in state_dict:
                del state_dict[name]
            else:
                print(f"WARNING: filtering but {name} not in state_dict")

    # also remove the keys in state_dict generated from
    # lang_encoder.old_decoder_blocks and lang_encoder.gated_cross_attn_layers
    # because these are already saved in lang_encoder.model...
    to_delete = [
        n
        for n in state_dict.keys()
        if ("lang_encoder.old_decoder_blocks" in n)
        or ("lang_encoder.gated_cross_attn_layers" in n)
        # PEFT add 'base_model.model' to parameter name
        or ("lang_encoder.base_model.model.old_decoder_blocks" in n) 
        or ("lang_encoder.base_model.model.gated_cross_attn_layers" in n)
        or ("vision_encoder" in n)
    ]
    for name in to_delete:
        del state_dict[name]
    return state_dict


def save_checkpoint(model, optimizer, lr_scheduler, epoch, args):
    """
    Save training checkpoint with model, optimizer, and lr_scheduler state.
    """
    if args.fsdp:
        FSDP.set_state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            FullOptimStateDictConfig(rank0_only=True),
        )
        model_state = model.state_dict()
        optim_state = FSDP.optim_state_dict(model, optimizer, group=args.my_group)

    else:
        model_state = model.state_dict()
        optim_state = optimizer.state_dict()

    if args.rank == 0:
        if not (args.fsdp and not args.fsdp_use_orig_params):
            model_state = filter_state_dict_to_trainable(model, model_state)

        if not os.path.exists(args.run_name):
            os.makedirs(args.run_name)

        checkpoint_dict = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": optim_state,
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        }

        print(f"Saving checkpoint to {args.run_name}/checkpoint_{epoch}.pt")
        torch.save(checkpoint_dict, f"{args.run_name}/checkpoint_{epoch}.pt")
        if args.report_to_wandb and args.save_checkpoints_to_wandb:
            wandb.save(f"{args.run_name}/checkpoint_{epoch}.pt")

        if args.delete_previous_checkpoint:
            if epoch > 0:
                os.remove(f"{args.run_name}/checkpoint_{epoch-1}.pt")


def get_params_count(model, max_name_len: int = 100):
  params = [(name[:max_name_len], p.numel(), str(tuple(p.shape)), p.requires_grad) for name, p in model.named_parameters()]
  total_trainable_params = sum([x[1] for x in params if x[-1]])
  total_nontrainable_params = sum([x[1] for x in params if not x[-1]])
  return params, total_trainable_params, total_nontrainable_params


def get_params_count_summary(model, max_name_len: int = 100):
    padding = 100
    params, total_trainable_params, total_nontrainable_params = get_params_count(model, max_name_len)
    param_counts_text = ''
    param_counts_text += '=' * (max_name_len + padding) + '\n'
    param_counts_text += f'| {"Module":<{max_name_len}} | {"Trainable":<10} | {"Shape":>15} | {"Param Count":>12} |\n'
    param_counts_text += '-' * (max_name_len + padding) + '\n'
    for name, param_count, shape, trainable in params:
        truncated_name = name[:max_name_len]  # Truncate the name if it's too long
        param_counts_text += f'| {truncated_name:<{max_name_len}} | {"True" if trainable else "False":<10} | {shape:>15} | {param_count:>12,} |\n'
    param_counts_text += '-' * (max_name_len + padding) + '\n'
    param_counts_text += f'| {"Total trainable params":<{max_name_len}} | {"":<10} | {"":<15} | {total_trainable_params:>12,} |\n'
    param_counts_text += f'| {"Total non-trainable params":<{max_name_len}} | {"":<10} | {"":<15} | {total_nontrainable_params:>12,} |\n'
    param_counts_text += '=' * (max_name_len + padding) + '\n'
    return param_counts_text


# https://github.com/open-mmlab/Multimodal-GPT/blob/main/mmgpt/models/open_flamingo/builder.py
def prepare_model_for_tuning(model, config):
    # Freeze all parameters
    model.requires_grad_(False)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0

    config = json.load(open(config, 'r'))
    if config['from_pretrained']:
        model.lang_encoder = PeftModel.from_pretrained(
            model.lang_encoder,
            config['from_pretrained']
        )
    elif config['lora']:
        lora_config = LoraConfig(
            r=config['lora_r'],
            lora_alpha=config['lora_alpha'],
            target_modules=config['lora_target_modules'],
            lora_dropout=config['lora_dropout'],
            bias="none",  # won't use bias currently
            modules_to_save=[],  # TODO: might be helpful if save partial model
            task_type="VL",
        )
        model.lang_encoder = get_peft_model(model.lang_encoder, peft_config=lora_config)

    # manually unfreeze modules, we use a `substring` fashion mathcing
    for name, param in model.named_parameters():
        if any(substr in name for substr in config['unfrozen']):
            param.requires_grad = True

    return model, config


def resume_from_checkpoints(model, checkpoints, args=None, logger=None):
    messages = []
    for this_checkpoint in checkpoints.split(','):
        # print('#'*128)
        # print('loading checkpoint from', this_checkpoint)
        if args is not None and logger is not None and args.rank == 0:
            logger.info(f"Loading checkpoint from {this_checkpoint}")
        checkpoint = torch.load(this_checkpoint, map_location="cpu")

        resume_from_epoch = 0
        if args is not None and args.continue_training:
            resume_from_epoch = checkpoint["epoch"] + 1
            
        msd = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint.keys() else checkpoint
        msd = {k.replace("module.", ""): v for k, v in msd.items()}

        # for key in msd.keys():
        #     print('\t', key)

        if args is None:
            message = model.load_state_dict(msd, False)
        # for fsdp, only one rank needs to load the state dict
        elif not args.fsdp or args.rank == 0:
            message = model.load_state_dict(msd, False)
        else:
            message = None
        # print(message)
        messages.append(message)
    return model, checkpoint, resume_from_epoch, messages