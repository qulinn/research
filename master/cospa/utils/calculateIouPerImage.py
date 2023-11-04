# gt_path = "/data/Users/izumi/cloud-images/2006/masks/*.jpg"
# predict_data = "/data/Users/izumi/bs-module-test/result/binary_filter_after_crf/*.jpg"
# pre-trained model: 2008-2010, patch_size=255, epoch=900,
# test setting: test data: 2006, patch_size=255, stride=43, gt_prob = 0.7


gt_path = "/data/Users/izumi/cloud-images/2006/masks/*.jpg"
predict_data_path = "/data/Users/izumi/bs-module-test/result/gt06/binary_filter_after_crf/*.jpg"
# predict_data_path = "/data/Users/izumi/cospa/result/2008-2010/2006-testdata/additional_data/resnet18/patch_255/epoch900/default-gt-prob-08/seg/default/binary_filter_after_crf/*.jpg"
# predict_data_path = "/data/Users/izumi/bs-module-test/result/binary_filter_after_crf/*.jpg"
# save_dir_path = "/data/Users/izumi/bs-module-test/result"



import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

def getImagesPath(gt_path, predict_data_path):
    gt_path = glob.glob(gt_path)
    predict_data_path = glob.glob(predict_data_path)
    gt_path.sort()
    predict_data_path.sort()

    return gt_path, predict_data_path

def importData(gt_path, predict_data_path):
    gt_images = [cv2.imread(path, 0) for path in gt_path]
    predict_images = [cv2.imread(path, 0) for path in predict_data_path]

    return gt_images, predict_images

def calculateIouDice(gt_images, prediction_images):
    height, width = len(gt_images[0]), len(gt_images[0])
    iou, dice = [], []
    for i in range(len(gt_images)):
        ground_truth = gt_images[i]
        prediction = prediction_images[i]

        # tp: true positive
        # fp: false positive
        # fn: false negative 
        tp, fp, fn = 0, 0, 0
        for j in range(height):
            for k in range(width):
                # when prediction = 1
                if all(prediction[j][k] != np.array([0,0,0])):
                    # when gt = 1
                    if all(ground_truth[j][k] != np.array([0,0,0])):
                        tp += 1
                    # when gt = 0
                    else:
                        fp += 1
                # when gt = 1
                elif all(ground_truth[j][k] != np.array([0,0,0])):
                    # when prediction = 0
                    if all(prediction[j][k] == np.array([0,0,0])):
                        fn += 1
        if tp == 0 and fp == 0 and fn == 0:
            raise Exception("ERROR: tp, fp, fn = 0, 0, 0")


        iou_per_images = tp / (tp + fp + fn)
        dice_per_images = 2 * tp / (2 * tp + fp + fn)
        iou.append(iou_per_images)
        dice.append(dice_per_images)

    # iou_mean = np.mean(iou)
    # dice_mean = np.mean(dice)
    return iou, dice


def test(gt_path, predict_data_path):
    gt_images_path, prediction_images_path = getImagesPath(gt_path, predict_data_path)
    gt_images, predict_images = importData(gt_images_path, prediction_images_path)
    # print(gt_images_path)
    # print(prediction_images_path)
    # print(gt_images)
    # print(predict_images)
    iou, dice = calculateIouDice(gt_images, predict_images)
    return iou, dice

def main():
    iou, dice = test(gt_path, predict_data_path)
    print("iou:", iou)
    print("dice:", dice)

if __name__ == '__main__':
        main()