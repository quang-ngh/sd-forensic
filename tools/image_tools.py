import cv2
import numpy as np 
from PIL import Image 
import os 
import json 
import torch
from torchvision import transforms
AUTOSPLICE="autosplice"
MAGICBRUSH="magicbrush"

def get_samples(auth_dir, mask_dir, edited_dir, caption_dir, n_samples=4, offset=0, 
                dataset="autosplice", return_orig_size=True):

    tf = transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: x * 2 -1 )
    ])    
    img_list = sorted(os.listdir(auth_dir))
    mask_list = sorted(os.listdir(mask_dir))
    edited_list = sorted(os.listdir(edited_dir))
    caption_list = sorted(os.listdir(caption_dir))

    #   Get image
    images=[]
    masks=[]
    edited=[]
    orig_sizes =  []
    image_paths=[]
    for item in caption_list[n_samples*offset:n_samples*(offset+1)]:
        basename, ext = item.split(".")
        
        img_path = f"{auth_dir}/{basename}.jpg" if dataset == AUTOSPLICE else f"{auth_dir}/{basename}.png"
        

        mask_name = f"{basename}_mask" if dataset == AUTOSPLICE else basename
        mask_path = f"{mask_dir}/{mask_name}.png"

        if dataset == AUTOSPLICE:
            for item in edited_list:
                if item.find(basename) >= 0:
                    edited_name = item.split(".")[0]
        else:
            edited_name = basename
        
        edited_name = edited_name + ".png" if dataset == MAGICBRUSH else edited_name + ".jpg"
        edited_path = f"{edited_dir}/{edited_name}"
        image_paths.append(edited_path)

        H, W = np.array(Image.open(img_path)).shape[:-1]
        img = tf(Image.open(img_path))
        mask = tf(Image.open(mask_path))
        edited_img = tf(Image.open(edited_path))

        images.append(img)
        masks.append(mask)
        edited.append(edited_img)

        orig_sizes.append((H,W)) 
        #   Captions
    images = torch.stack(images, dim=0) 
    masks  = torch.stack(masks, dim=0)
    edited  = torch.stack(edited, dim=0)

    if return_orig_size:
        return images, masks, edited, orig_sizes, image_paths
    return images, masks, edited, image_paths

def create_mask2folder():
    
    mask_folder = "datasets/AutoSplice/Mask"
    image_folder = "datasets/AutoSplice/Authentic"
    result = {}
    for item in sorted(os.listdir(mask_folder)):
        basename = item.split("_")[0]
        if basename not in result:
            result[basename] = []
        for fn in sorted(os.listdir(image_folder)):
            if basename in fn:
                result[basename].append(fn)
    
    with open("datasets/AutoSplice/mask2authen.json", mode="w") as f:
        json.dump(result, f)
    f.close()

def convert2uint8(torch_tensor: torch.Tensor):
    img = torch_tensor.clone().detach().cpu()
    img = img.clamp(0,1)
    img = img.numpy() * 255.0
    return img.astype("uint8")

def save_torch_img(img_tensor: torch.Tensor):

    images = []

    #   CxHxW -> HxWxC
    if len(img_tensor.shape) == 3:
        if img_tensor.shape[0] == 3:
            img_tensor = img_tensor.permute(1,2,0)
        image_uint8 = convert2uint8(img_tensor)
        images.append(
            Image.fromarray(image_uint8)
        )
        return images

    #   BxCxHxW
    B, C, H, W = img_tensor.shape
    if C == 3:
        img_tensor = img_tensor.permute(0, 2, 3, 1)

    image_uint8 = convert2uint8(img_tensor)    
    for image in image_uint8:
        images.append(
            Image.fromarray(image)
        )
    return images 

def overlay_heatmap(images, heatmap, normalize=False):
    """
    args: 
        images: torch.Tensor: single image
                [C,H,W] or [B,C,H,W]
        heatmap: torch.Tensor
                [H,W]
    """
    if isinstance(images, torch.Tensor):
        #   CxHxW -> 1xCxHxW
        if len(images.shape) == 3:            
            images = images.unsqueeze(0) 
        if images.shape[1] == 3:
            images = images.permute(0, 2, 3, 1) 
        images = images.clone().detach().cpu().clamp(0,1).numpy()

    # breakpoint() 
    #   Overlay heatmap
    heatmap = heatmap.clone().cpu().numpy()
    if normalize:
        min_value = heatmap.min()
        max_value = heatmap.max()
        heatmap = (heatmap - min_value) / (max_value - min_value)

    heatmap = (heatmap*255.0).astype("uint8")
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = heatmap.astype("float32") / 255.0

    #   Class activation map
    cam = images.astype("float32") + heatmap
    cam = cam / np.max(cam)

    #   Convert to save format
    vis = (cam * 255.0).astype("uint8")
    vis  = cv2.cvtColor(vis[0], cv2.COLOR_RGB2BGR)

    return cam, vis
            
