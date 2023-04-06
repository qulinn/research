import time
start = time.perf_counter()

import os
import logging
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
logging.getLogger('tensorflow').setLevel(logging.FATAL)
warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import glob
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from utils.CRF import apply_crf
from utils.make_filter import patch_segmentation, show_segmentation_result, show_CRF_result


def main():

    # Load PNULoss model
    model = load_model(opt.model, compile=False)

    # Get image paths
    Imgs_paths = sorted(glob.glob(opt.dataset+"/*.*"))
    Outputs = []

    for img_path in tqdm(Imgs_paths):

        # Load image
        x = np.array(Image.open(img_path).convert('RGB'))

        # Segmentation
        prob_map, binary_filter, color_filter, superimpose = patch_segmentation(
            img=x,
            model=model, 
            patch_size=opt.patch_size,
            stride=opt.stride)
        Outputs.append(prob_map)

        # CRF
        binary_filter_after_crf, binary_superimpose_before_crf, binary_superimpose_after_crf = \
            apply_crf(
                x=x, 
                y=binary_filter
                )
    
        prob_map = np.squeeze(prob_map)
        color_filter = np.squeeze(color_filter)
        binary_filter = np.squeeze(binary_filter)
        binary_filter_after_crf = np.squeeze(binary_filter_after_crf)

        # Save
        f_name = os.path.basename(img_path)
        f_base_name, ext = os.path.splitext(f_name)
        if ext.lower() not in [".jpg",".png"]:
            ext = ".jpg"
        f_name = f_base_name+ext
        
        plt.imsave(os.path.join(opt.save_dir+"/original",f"{f_name}"), x)
        plt.imsave(os.path.join(opt.save_dir+"/color_filter",f"{f_name}"), color_filter, cmap="jet")
        plt.imsave(os.path.join(opt.save_dir+"/color_superimpose",f"{f_name}"), superimpose)
        
        plt.imsave(os.path.join(opt.save_dir+"/binary_filter",f"{f_name}"), binary_filter, cmap="gray")
        plt.imsave(os.path.join(opt.save_dir+"/binary_filter_after_crf",f"{f_name}"), binary_filter_after_crf, cmap="gray")
        
        plt.imsave(os.path.join(opt.save_dir+"/binary_superimpose",f"{f_name}"), binary_superimpose_before_crf)
        plt.imsave(os.path.join(opt.save_dir+"/binary_superimpose_after_crf",f"{f_name}"), binary_superimpose_after_crf)
        
        show_segmentation_result(x, color_filter, superimpose)
        plt.savefig(os.path.join(opt.save_dir+"/color_filter_summary",f"{f_name}"), bbox_inches='tight', pad_inches=0.1)

        show_CRF_result(x, binary_filter, binary_filter_after_crf)
        plt.savefig(os.path.join(opt.save_dir+"/binary_filter_summary",f"{f_name}"), bbox_inches='tight', pad_inches=0.1)

    # Save outputs (probability maps)
    np.save(f'{opt.save_dir}/prob_output',np.array(Outputs))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            prog='segmentation.py',
            usage='Image segmentation with PU learned model.',
            description='---'
            )
    parser.add_argument('--dataset', type=str, required=True, help='Image data path.')
    parser.add_argument('--patch_size', type=int, default=32, help='Size of patch image for prediction of segmentation values.')
    parser.add_argument('--stride', type=int, default=1, help='Stride of raster scan. Stride^2 area of a image is sequentially filled by the single predicted value of a patch image.')
    parser.add_argument('--model', type=str, required=True, help='Trained model path.')
    parser.add_argument('--save_dir', type=str, default="Results", help='path of the save results folder.')
    opt = parser.parse_args()

    os.makedirs(os.path.join(opt.save_dir+"/original"),exist_ok=True)
    os.makedirs(os.path.join(opt.save_dir+"/color_filter"),exist_ok=True)
    os.makedirs(os.path.join(opt.save_dir+"/color_superimpose"),exist_ok=True)
    os.makedirs(os.path.join(opt.save_dir+"/binary_filter"),exist_ok=True)
    os.makedirs(os.path.join(opt.save_dir+"/binary_filter_after_crf"),exist_ok=True)
    os.makedirs(os.path.join(opt.save_dir+"/binary_superimpose"),exist_ok=True)
    os.makedirs(os.path.join(opt.save_dir+"/binary_superimpose_after_crf"),exist_ok=True)
    os.makedirs(os.path.join(opt.save_dir+"/color_filter_summary"),exist_ok=True)
    os.makedirs(os.path.join(opt.save_dir+"/binary_filter_summary"),exist_ok=True)
    
    main()

print("segmentation time: ", time.perf_counter() - start)