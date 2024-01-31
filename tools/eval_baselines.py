import torch 
import numpy as np
from PIL import Image 
import os 
import json 
import argparse
from metrics import *
import cv2 
from tqdm import tqdm 

def get_opt():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument("--gt-dir", type=str, help="Path to the image list") 
    parser.add_argument("--pred-dir", type=str, default="")
    parser.add_argument("--save-name", type=str, default="")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--baseline-name", type=str, default="")
    parser.add_argument("--authentic", action='store_true', default="")
    opt = parser.parse_args()
    return opt 

def repeat_mask(mask_arr):
    mask_new = np.reshape(mask_arr.copy(), (mask_arr.shape[0], mask_arr.shape[1],1))
    mask_new = np.repeat(mask_new, 3, axis=-1)
    return mask_new

def create_examples():
    dataset = "magicbrush"
    baseline_name = "mvss"

    
    if dataset == "autosplice":
        #   AutoSplice
        caption_dir = "datasets/AutoSplice/Caption"
        image_dir = "datasets/AutoSplice/Authentic"
        forge_dir = "datasets/AutoSplice/Forged_JPEG100"
        mask_gt = "datasets/AutoSplice/Mask" 
        pscc_mask_pred = "experiments/reproduce_baselines/pscc_inference_forged100/predictions"
        mvss_mask_pred =  "experiments/reproduce_baselines/mvss_inference_forged100/predictions"
        trufor_mask_pred  = "experiments/reproduce_baselines/trufor_inference_forged100/predictions"

    elif dataset == "magicbrush":
        #   MagicBrush
        image_dir = "datasets/MagicBrush/images"
        forge_dir = "datasets/MagicBrush/target_images"
        mask_gt = "datasets/MagicBrush/bin_mask" 
        pscc_mask_pred = f"experiments/reproduce_baselines/pscc_magicbrush_inference/predictions"
        mvss_mask_pred =  "experiments/reproduce_baselines/mvss_magicbrush_inference/predictions"
        trufor_mask_pred  = "experiments/reproduce_baselines/trufor_magicbrush_inference/predictions"
 
    save_dir = f"experiments/reproduce_baselines/{baseline_name}_magicbrush_inference/examples"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    if baseline_name == "mvss":
        pred_mask_dir = mvss_mask_pred
    elif baseline_name == "pscc":
        pred_mask_dir = pscc_mask_pred
    elif baseline_name == "trufor":
        pred_mask_dir = trufor_mask_pred

    mask_gt_list = sorted(os.listdir(mask_gt))
    forge_img_list = sorted(os.listdir(forge_dir))
    auth_img_list = sorted(os.listdir(image_dir))
    pscc_mask_list = sorted(os.listdir(pscc_mask_pred))

    if dataset == "autosplice":
        for item in sorted(os.listdir(caption_dir))[:100]:
            fn = open(f"{caption_dir}/{item}", mode="r")
            objs = json.load(fn)
            fn.close()

            orig_caption = objs["ori_caption"][:50]
            basename = str(objs["id"])
            orig_term = objs["oriCaptionTerm"]
            user_term = objs["userInputTerm"]

            img_arr = np.array(Image.open(f"{image_dir}/{basename}.jpg"))
            mask_arr = np.array(Image.open(f"{mask_gt}/{basename}_mask.png"))
            mask_arr_save = repeat_mask(mask_arr)

            found_forge_list = []
            for name in forge_img_list:
                if name.find(basename) >= 0:
                    found_forge_list.append(name)
                    
            print(found_forge_list)
            for idx, obj in enumerate(found_forge_list):
                name_tmp, _ = obj.split(".")
                forge_arr = np.array(Image.open(f"{forge_dir}/{name_tmp}.jpg"))
                mask_pred = np.array(Image.open(f"{pred_mask_dir}/{name_tmp}.png"))
                mask_pred_save = repeat_mask(mask_pred)
            
                vis = np.hstack([img_arr, forge_arr, mask_arr_save, mask_pred_save])
                Image.fromarray(vis).save(f"{save_dir}/{save_name}.png")

    elif dataset == "magicbrush":
        print("Magicbrush")
        for item in forge_img_list[:100]:
            basename = item.split(".")[0]
            img_arr = np.array(Image.open(f"{image_dir}/{basename}.png"))
            forge_arr = np.array(Image.open(f"{forge_dir}/{basename}.png"))
            mask_arr = np.array(Image.open(f"{mask_gt}/{basename}.png"))
            mask_arr_save = repeat_mask(mask_arr)
            mask_pred_save = repeat_mask(np.array(Image.open(f"{pred_mask_dir}/{basename}.png")))
            vis = np.hstack([img_arr, forge_arr, mask_arr_save, mask_pred_save])
            Image.fromarray(vis).save(f"{save_dir}/{basename}.png")

def evaluate(args):
    gt_dir = args.gt_dir
    pred_dir = args.pred_dir 
    save_name = args.save_name
    baseline_name = args.baseline_name
    dataset = args.dataset

    f1_scores =  []
    iou_scores = []
    acc_scores = []
    precision_scores = []
    recall_scores = []

    #   Get a mapping between the ground truth and prediction path
    mask2forge_dict = None
    if not args.authentic:
        fn = open(f"datasets/{dataset}/mask2forge.json")
    else:
        fn = open(f"datasets/{dataset}/mask2authen.json")

    mask2forge_dict = json.load(fn)
    fn.close()
    
    for gt_basename, pred_list in tqdm(mask2forge_dict.items()):
        if dataset == "AutoSplice":
            mask_gt = np.array(Image.open(f"{gt_dir}/{gt_basename}_mask.png"))
        elif dataset == "MagicBrush":
            mask_gt = np.array(Image.open(f"{gt_dir}/{gt_basename}.png"))
        else:
            raise ValueError("Not implemented")

        mask_H, mask_W = mask_gt.shape 
        for pred_fn in pred_list:
            pred_basename = pred_fn.split(".")[0]
            pred = np.array(Image.open(f"{pred_dir}/{pred_basename}.png"))

            if pred.shape[0] != mask_H or pred.shape[1] != mask_W:
                pred = cv2.resize(pred, (mask_H, mask_W), interpolation=cv2.INTER_CUBIC)
            accuracy, f1, precision, recall = calculate_pixel_f1(pred, mask_gt)
            iou = calculate_iou(mask_gt, pred)

            acc_scores.append(accuracy)
            f1_scores.append(f1)
            precision_scores.append(precision)
            recall_scores.append(recall)
            iou_scores.append(iou)
    

    res = {
        "mIoU": np.mean(iou_scores),
        "accuracy": np.mean(acc_scores),
        "precision": np.mean(precision_scores),
        "recall": np.mean(recall_scores),
        "f1": np.mean(f1_scores)
    } 

    with open(f"experiments/reproduce_baselines/{baseline_name}_{save_name}.json",  mode="w") as f:
        json.dump(res, f)
    f.close()



if __name__ == '__main__':
    
    args = get_opt() 
    # create_examples()
    evaluate(args)


    # image_dir = "datasets/AutoSplice/Authentic"
    # forge_dir = "datasets/AutoSplice/Forged_JPEG100"
    # mask_gt = "datasets/AutoSplice/Mask"
    
    # forge_dir = "datasets/MagicBrush/bin_mask"
    # mask_gt = "datasets/MagicBrush/target_images"

    # mask_gt_list = sorted(os.listdir(mask_gt))
    # forge_img_list = sorted(os.listdir(forge_dir))
    # objs = {}
    # for item in mask_gt_list:
    #     filename = item.split(".")[0]
    #     basename = filename.split("_")[0]
    #     if basename not in objs:
    #         objs[basename] = []
    #     for name in forge_img_list:
    #         if name.find(basename) >= 0:
    #             objs[basename].append(name)
    
    # save_name = "mask2forge.json"
    # fn = open(f"datasets/MagicBrush/{save_name}", mode="w")
    # json.dump(objs, fn)
    # fn.close()