import torch
import torchaudio

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.amp

import numpy as np

import yaml
import argparse
import os
import logging
from tqdm import tqdm
from time import time
import datetime
from collections import OrderedDict
from copy import deepcopy

from edm import EDM
from mustdiff import MusTDiff
from dataset import MP3Dataset, get_mp3_file

import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch 1.12.

class Config(object):
    def __init__(self, dic):
        for key in dic:
            setattr(self, key, dic[key])
    def items(self):
        # Return the items of the dictionary used to initialize the class
        return self.__dict__.items()

def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/train.log")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag
        
        
@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
        
        

# ===== training =====

def train(args):
    """
    Trains a new DiT model.
    """
    use_amp = args.use_amp
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    
    # Configuration:
    yaml_path = args.config
    with open(yaml_path, 'r') as f:
        args = yaml.full_load(f)
    args = Config(args)
    
    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    local_rank = dist.get_rank()
    device = local_rank % torch.cuda.device_count()
    local_seed = args.global_seed + local_rank
    torch.cuda.set_device(device)
    
    # Setup an experiment folder:
    model_dir = os.path.join(args.save_dir, "ckpts")
    listen_dir = os.path.join(args.save_dir, "listen")
    if local_rank == 0:
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(listen_dir, exist_ok=True)

    # Log:
    if local_rank == 0:
        logger = create_logger(args.save_dir)
        logger.info(f"Experiment directory created at {args.save_dir}")
    else:
        logger = create_logger(None)
    
    logger.info("########## Configuration ##########")
    for key, value in args.items():
        logger.info(f"{key}: {value}")

    # Seed:
    logger.info("local_rank = {}, seed = {}".format(local_rank, local_seed))
    np.random.seed(seed=local_seed)
    torch.manual_seed(seed=local_seed)
    torch.cuda.manual_seed_all(seed=local_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Model:
    nn_model = MusTDiff(num_heads=8, depth=2)
    if local_rank == 0:
        pytorch_total_grad_params = sum(p.numel() for p in nn_model.parameters() if p.requires_grad)
        logger.info(f'total number of trainable parameters in the Score Model: {pytorch_total_grad_params}')
        pytorch_total_params = sum(p.numel() for p in nn_model.parameters())
        logger.info(f'total number of parameters in the Score Model: {pytorch_total_params}')
        
    model = DDP(nn_model.to(device), device_ids=[local_rank], find_unused_parameters=True, broadcast_buffers=False)
    # diffusion = EDM(model, p_mean=args.diffusion.p_mean, p_std=args.diffusion.p_std, sigma_data=args.diffusion.sigma_data)
    diffusion = EDM(model, **args.diffusion)
    
    # EMA:
    if local_rank == 0:
        ema = deepcopy(nn_model).to(device)  # Create an EMA of the model for use after training
        requires_grad(ema, False)
        # Prepare models for training:
        update_ema(ema, diffusion.model.module, decay=0)  # Ensure EMA is initialized with synced weights  

    # Data:
    train_set=MP3Dataset(args.data_dir)
    logger.info(f"Train dataset:{len(train_set)}")
    
    sampler = DistributedSampler(
        train_set,
        num_replicas=dist.get_world_size(),
        rank=local_rank,
        shuffle=True,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=args.num_workers,
        persistent_workers=True
    )

    # Learning rate and optimizer:
    lr = args.learning_rate
    DDP_multiplier = dist.get_world_size()
    logger.info("Using DDP, lr = %f * %d" % (lr, DDP_multiplier))
    lr *= DDP_multiplier
    optim = torch.optim.AdamW(diffusion.model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # Load checkpoint
    if args.load_epoch != -1:
        checkpoint_path = os.path.join(model_dir, f"model_{args.load_epoch}.pth")
        logger.info("loading model at", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        diffusion.model.load_state_dict(checkpoint['model'])
        if local_rank == 0:
            ema.load_state_dict(checkpoint['ema'])
        optim.load_state_dict(checkpoint['optim'])
        

    # Training
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    
    for current_epoch in range(args.load_epoch + 1, args.n_epoch):
        for g in optim.param_groups:
            g['lr'] = lr * min((current_epoch + 1.0) / args.warm_epoch, 1.0) # warmup
        sampler.set_epoch(current_epoch)
        dist.barrier()
        
        diffusion.model.train()
        train_set.prefetch()
        # logging
        if local_rank == 0:
            current_lr = optim.param_groups[0]['lr']
            logger.info(f'epoch {current_epoch}, lr {current_lr:f}')
            loss_ema = None
            progress_bar = tqdm(train_loader)
        else:
            progress_bar = train_loader
        for x in progress_bar:
            optim.zero_grad()
            x = x.to(device)
            # print(x.shape)
            loss = diffusion.loss(x)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(parameters=diffusion.model.parameters(), max_norm=1.0)
            for param in diffusion.model.parameters():
                if param.grad is not None:
                    torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
            scaler.step(optim)
            scaler.update()
            

            # logging
            dist.barrier()
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss = loss.item() / dist.get_world_size()
            if local_rank == 0:
                update_ema(ema, diffusion.model.module)
                if loss_ema is None:
                    loss_ema = loss
                else:
                    loss_ema = 0.95 * loss_ema + 0.05 * loss
                progress_bar.set_description(f"loss: {loss_ema:.4f}")

            running_loss += loss
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)

                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()
                
        # testing
        if local_rank == 0:
            if current_epoch % 100 == 0 or current_epoch == args.n_epoch - 1:
                pass
            else:
                continue

            temp_edm = EDM(ema)

            noise = torch.randn([args.n_sample,1024,1024]).to(device)
            temp_edm.model.eval()
            with torch.no_grad():
                x_gen = temp_edm.sample(noise)

            save_path = os.path.join(listen_dir, f"audio_ep{current_epoch}_ema")
            os.makedirs(save_path, exist_ok=True)
            for i in range(x_gen.shape[0]):
                torch_tensor = x_gen[i]
                current_audio_save_path = os.path.join(save_path, f"{i}_sample.mp3")
                get_mp3_file(torch_tensor, current_audio_save_path)
            logger.info(f'saved audio at{save_path}')

            # optionally save model
            if args.save_model:
                checkpoint = {
                    'model': diffusion.model.state_dict(),
                    'ema': ema.state_dict(),
                    'optim': optim.state_dict(),
                }
                save_path = os.path.join(model_dir, f"model_{current_epoch}.pth")
                torch.save(checkpoint, save_path)
                logger.info(f'saved model at{save_path}')

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--use_amp", action='store_true', default=False)
    args = parser.parse_args()
    train(args)
