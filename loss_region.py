import argparse
import numpy as np
from PIL import Image
import PIL
import os 
from diffusers import DDIMScheduler, StableDiffusionPipeline, StableDiffusionDiffEditPipeline, DDIMInverseScheduler
from diffusers.utils import BaseOutput
import torch
import json
from typing import Optional, Callable, Union, List, Dict, Any
import os
import torchvision
from dataclasses import dataclass
from torchvision import transforms
from src.pipline_diff_forensics import DiffForensicPipeline, DiffEditInversionPipelineOutput
from tools.image_tools import *
from tools.utils import *
import lpips
from torch.utils.tensorboard import SummaryWriter

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--caption-dir", type=str, default="../datasets/AutoSplice/Caption")
    parser.add_argument("--image-dir", type=str, default="../datasets/AutoSplice/Authentic")
    parser.add_argument("--mask-dir", type=str, default="../datasets/AutoSplice/Mask")
    parser.add_argument("--edited-dir", type=str, default="../datasets/AutoSplice/Forged_JPEG100")
    parser.add_argument("--checkpoint-path", type=str, default="")
    parser.add_argument("--experiment-name", type=str)
    parser.add_argument("--real-image", action="store_true")
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--concat-images", action="store_true")
    
    #   Invert configuration
    parser.add_argument("--inference-steps", type=int, default=50)
    parser.add_argument("--inpaint-strength", type=float, default=0.8)
    parser.add_argument("--guidance-scale", type=float, default=7.5)

    #   Analyze images
    parser.add_argument("--loss-type", type=str, default="l2")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=15)
    return parser.parse_args()

def main(args):

    ######################################  Setup   ##########################################
    DEVICE=torch.device("cuda")    
    print_args(args) 
    #   Setup workspace folder
    exp_dir = f"experiments/{args.experiment_name}"    
    tb_writer = SummaryWriter(f"{exp_dir}/tensorboard") if args.tensorboard else None

    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)
    caption_dir = args.caption_dir
    auth_dir = args.image_dir
    mask_dir = args.mask_dir
    edited_dir = args.edited_dir
    ###########################################################################################

    ############################################### Models/Data  ##############################
    pipe = DiffForensicPipeline.from_pretrained(
        args.checkpoint_path,
        torch_dtype=torch.float32,
        safety_checker=None,
        use_safetensors=True
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    pipe.to(DEVICE)
    
    loss_fn = register_loss(args.loss_type)
    images, masks, edited_images, orig_sizes, _ = get_samples(auth_dir=auth_dir, mask_dir=mask_dir, edited_dir=edited_dir 
                                           ,caption_dir=caption_dir,n_samples = 3, offset=0, dataset="autosplice")
    images = images.to(DEVICE)
    edited_images = edited_images.to(DEVICE)
    ############################################################################################
    #   Create dir to store the inverse latent
    for idx in range(images.shape[0]):
        if not os.path.isdir(f"{exp_dir}/image_{idx}"):
            os.makedirs(f"{exp_dir}/image_{idx}")

    text = ["a photo of a bus"]
    input_images = edited_images
    if args.real_image:
        input_images = images
        
        breakpoint()
    all_inv_latents = pipe.invert(
        prompt =  text * images.shape[0],
        # prompt = list_promtps * images.shape[0],
        image = input_images,
        inpaint_strength = args.inpaint_strength,
        guidance_scale = args.guidance_scale,
        num_inference_steps = args.inference_steps,
        return_noise=False
    )
    all_inv_latents = all_inv_latents.latents.detach().cpu()

    #   Steps to analyze
    start = args.start
    end = args.end

    if args.end > int(args.inference_steps * args.inpaint_strength):
        args.end = int(args.inference_steps * args.inpaint_strength)
    total_steps = all_inv_latents.shape[1] 

    for step in range(start, end):
        inv_step = total_steps - (step + 1)
        z0_inv = all_inv_latents[:, step, :, : ,:].to(DEVICE)
        with torch.no_grad():
            img_torch = pipe.decode_latents(z0_inv, return_type="pt").to(DEVICE)
            img_torch = img_torch.permute(0, 3, 1, 2)
        loss = loss_fn(input_images, img_torch)
        #   If using PSNR score --> normalized the psnr to visualize the heatmap
        if args.loss_type == "psnr":
            batch_max = torch.max(loss.clone().view(loss.shape[0], -1), dim=-1).values
            batch_min = torch.min(loss.clone().view(loss.shape[0], -1), dim=-1).values
            expand_dims = len(loss.shape) - len(batch_max.shape)
            for _ in range(expand_dims):
                batch_max.unsqueeze_(-1)
                batch_min.unsqueeze_(-1)

        #   [B, C, H, W]
        loss = torch.mean(loss, dim = 1)                            

        #   Concat the input + reconstruction with the overlay heatmap
        if args.concat_images:
            img_torch_uint8 = img_torch.permute(0, 2, 3, 1)
            input_img_uint8 = input_images.permute(0, 2, 3, 1)
            img_torch_uint8 = convert2uint8(img_torch_uint8)
            input_img_uint8 = convert2uint8(input_img_uint8)

        for idx, loss_map in enumerate(loss): 
            cam, vis = overlay_heatmap(input_images[idx, :, :, :], loss_map, normalize=True) 
            if args.concat_images:
                vis = np.hstack([
                    input_img_uint8[idx, :, :, :], 
                    img_torch_uint8[idx, :, :, :],
                    vis
                ])
            cv2.imwrite(f"{exp_dir}/image_{idx}/inverse_step_{inv_step}.png", vis)
            #   Write tensorboard
            if tb_writer:
                tb_writer.add_image(f"image_{idx}", vis, inv_step, dataformats="HWC")

if __name__ == '__main__':
    args = get_args()
    main(args)
