# last update : 2023-11-03
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd



def get_image_path(gt_dir, prediction_dir):
    image_names = os.listdir(gt_dir)    
    gt_paths = []
    prediction_paths = []

    for name in image_names:
        gt_paths.append(os.path.join(gt_dir, name))
        prediction_paths.append(os.path.join(prediction_dir, name))
    
    return gt_paths, prediction_paths, image_names



def import_images(gt_paths, prediction_paths):
    gt_images = [cv2.imread(path, 0) for path in gt_paths]
    model_predictions = [cv2.imread(path, 0) for path in prediction_paths]

    return gt_images, model_predictions


def calculate_iou(label, prediction):
    intersection = np.logical_and(label, prediction)
    union = np.logical_or(label, prediction)
    iou = np.sum(intersection) / np.sum(union)
    # print(np.sum(intersection))
    # print(np.sum(union))
    # print('---')
    return iou


def calculate_dice(label, prediction):
    intersection = np.logical_and(label, prediction)
    dice = (2.0 * np.sum(intersection)) / (np.sum(label) + np.sum(prediction))
    return dice


def call_calculate_iou_dice(gt_images, model_predictions):
    iou_scores = []
    dice_scores = []

    for prediction, gt in zip(model_predictions, gt_images):
        iou = calculate_iou(gt, prediction)
        dice = calculate_dice(gt, prediction)
        iou_scores.append(iou)
        dice_scores.append(dice)
    
    return iou_scores, dice_scores



def output_to_csv(iou_scores, dice_scores, image_names, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    avg_iou = np.mean(iou_scores)
    avg_dice = np.mean(dice_scores)
    results = pd.DataFrame({'Image Name': image_names, 'IOU': iou_scores, 'Dice': dice_scores, 'mIoU': avg_iou, 'mDice': avg_dice})
    results.to_csv(os.path.join(save_dir, 'iou_dice_scores.csv'), index=False)


def make_and_save_graph(iou_scores, dice_scores, num_of_models, save_dir, sub_dirs) -> None:
    x = [i for i in range(num_of_models)]#range(len(predict_path))
    avg_iou = [np.mean(iou) for iou in iou_scores]
    avg_dice = [np.mean(dice) for dice in dice_scores]

    plt.plot(x, avg_iou, linestyle="solid", marker="o")
    plt.plot(x, avg_dice, linestyle="dashed", marker="x")
    plt.xticks(x, sub_dirs)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=18)
    plt.savefig(os.path.join(save_dir, 'iou.png'))
# ['p1e1','p1e3','p1e5','default', 'p2e3', 'p2e5','p3e1', 'p3e3','p3e5','p4e1','p4e3','p4e5']


def launcher(gt_dir, prediction_dirs, save_dirs, num_of_models, sub_dirs):
    iou_scores_per_dataset, dice_scores_per_dataset = [], []

    for prediction_dir, save_dir in zip(prediction_dirs, save_dirs):
        print(prediction_dir)
        gt_paths, prediction_paths, image_names = get_image_path(gt_dir, prediction_dir)
        gt_images, model_predictions = import_images(gt_paths, prediction_paths)
        iou_scores, dice_scores = call_calculate_iou_dice(gt_images, model_predictions)
        iou_scores_per_dataset.append(iou_scores)
        dice_scores_per_dataset.append(dice_scores)
        output_to_csv(iou_scores, dice_scores, image_names, save_dir)
    
    make_and_save_graph(iou_scores_per_dataset, dice_scores_per_dataset, num_of_models, src_dir, sub_dirs)


if __name__ == '__main__':
    
    gt_dir = '/data/Users/izumi/cloud-images/1ch-original-grayscale/2009/masks'
    src_dir = '/data/Users/izumi/cospa/result/2008-2010/2009-testdata/resnet18/patchsize255/epoch900/seg'
    predict_result_dir = 'binary_filter_after_crf'
        
    sub_dirs = os.listdir(src_dir)
    sub_dirs.remove('iou.png')
    sub_dirs.sort()

    model_dirs = [os.path.join(src_dir, sub_dir) for sub_dir in sub_dirs]
    prediction_dirs = [os.path.join(model_dir, predict_result_dir) for model_dir in model_dirs]

    save_dirs = [os.path.join(model_dir, 'iou') for model_dir in model_dirs]

    num_of_models = 12
    
    launcher(gt_dir, prediction_dirs, save_dirs, num_of_models, sub_dirs)


    # prediction_dir = os.path.join(model_dirs, predict_result_dir)  
    # src_dir = '/data/Users/izumi/cospa/result/2008-2010/2009-testdata/resnet18/patchsize255/epoch900/seg/p4e5'
