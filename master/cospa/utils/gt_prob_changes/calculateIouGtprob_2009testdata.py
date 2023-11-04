
# gt data
# 2006
# /data/Users/izumi/cospa/dataset-all/test/2006-testdata/images
# /data/Users/izumi/cospa/dataset-all/test/2006-testdata/masks
# 2008
# /data/Users/izumi/cospa/dataset-all/test/2008-testdata/images
# /data/Users/izumi/cospa/dataset-all/test/2008-testdata/masks
# 2009
# /data/Users/izumi/cospa/dataset-all/test/2009-testdata/images
# /data/Users/izumi/cospa/dataset-all/test/2009-testdata/masks

# model prediction
# 2008, patch_size=127, prior=0.4, eta=0.3
# /data/Users/izumi/cospa/result/2008/2008-testdata/undergraduate-thesis/resnet18/patchsize127/epoch800/seg/gt_prob08/p4e3/binary_filter_after_crf/*.jpg
# /data/Users/izumi/cospa/result/2008/2008-testdata/undergraduate-thesis/resnet18/patchsize127/epoch800/seg/gt_prob07/p4e3/binary_filter_after_crf/*.jpg
# /data/Users/izumi/cospa/result/2008/2008-testdata/undergraduate-thesis/resnet18/patchsize127/epoch800/seg/gt_prob06/p4e3/binary_filter_after_crf/*.jpg
# /data/Users/izumi/cospa/result/2008/2008-testdata/undergraduate-thesis/resnet18/patchsize127/epoch800/seg/gt_prob05/p4e3/binary_filter_after_crf/*.jpg
# /data/Users/izumi/cospa/result/2008/2008-testdata/undergraduate-thesis/resnet18/patchsize127/epoch800/seg/gt_prob04/p4e3/binary_filter_after_crf/*.jpg
# /data/Users/izumi/cospa/result/2008/2008-testdata/undergraduate-thesis/resnet18/patchsize127/epoch800/seg/gt_prob03/p4e3/binary_filter_after_crf/*.jpg
# /data/Users/izumi/cospa/result/2008/2008-testdata/undergraduate-thesis/resnet18/patchsize127/epoch800/seg/gt_prob02/p4e3/binary_filter_after_crf/*.jpg
# /data/Users/izumi/cospa/result/2008/2008-testdata/undergraduate-thesis/resnet18/patchsize127/epoch800/seg/gt_prob01/p4e3/binary_filter_after_crf/*.jpg

# 2006 test data prediction
# /data/Users/izumi/cospa/result/2008/2006-testdata/resnet18/patchsize127/epoch800/seg/gt_prob08/p4e3/binary_filter_after_crf
# /data/Users/izumi/cospa/result/2008/2006-testdata/resnet18/patchsize127/epoch800/seg/gt_prob07/p4e3/binary_filter_after_crf
# /data/Users/izumi/cospa/result/2008/2006-testdata/resnet18/patchsize127/epoch800/seg/gt_prob06/p4e3/binary_filter_after_crf
# /data/Users/izumi/cospa/result/2008/2006-testdata/resnet18/patchsize127/epoch800/seg/gt_prob05/p4e3/binary_filter_after_crf
# /data/Users/izumi/cospa/result/2008/2006-testdata/resnet18/patchsize127/epoch800/seg/gt_prob04/p4e3/binary_filter_after_crf
# /data/Users/izumi/cospa/result/2008/2006-testdata/resnet18/patchsize127/epoch800/seg/gt_prob03/p4e3/binary_filter_after_crf
# /data/Users/izumi/cospa/result/2008/2006-testdata/resnet18/patchsize127/epoch800/seg/gt_prob02/p4e3/binary_filter_after_crf
# /data/Users/izumi/cospa/result/2008/2006-testdata/resnet18/patchsize127/epoch800/seg/gt_prob01/p4e3/binary_filter_after_crf

# 2009 test data prediction
# /data/Users/izumi/cospa/result/2008/2009-testdata/resnet18/patchsize127/epoch800/seg/gt_prob08/p4e3/binary_filter_after_crf/*.jpg
# /data/Users/izumi/cospa/result/2008/2009-testdata/resnet18/patchsize127/epoch800/seg/gt_prob07/p4e3/binary_filter_after_crf/*.jpg
# /data/Users/izumi/cospa/result/2008/2009-testdata/resnet18/patchsize127/epoch800/seg/gt_prob06/p4e3/binary_filter_after_crf/*.jpg
# /data/Users/izumi/cospa/result/2008/2009-testdata/resnet18/patchsize127/epoch800/seg/gt_prob05/p4e3/binary_filter_after_crf/*.jpg
# /data/Users/izumi/cospa/result/2008/2009-testdata/resnet18/patchsize127/epoch800/seg/gt_prob04/p4e3/binary_filter_after_crf/*.jpg
# /data/Users/izumi/cospa/result/2008/2009-testdata/resnet18/patchsize127/epoch800/seg/gt_prob03/p4e3/binary_filter_after_crf/*.jpg
# /data/Users/izumi/cospa/result/2008/2009-testdata/resnet18/patchsize127/epoch800/seg/gt_prob02/p4e3/binary_filter_after_crf/*.jpg
# /data/Users/izumi/cospa/result/2008/2009-testdata/resnet18/patchsize127/epoch800/seg/gt_prob01/p4e3/binary_filter_after_crf/*.jpg



# gt_path = "/data/Users/izumi/cospa/dataset-all/test/2008-testdata/masks/*.jpg"
# predict_data_path = "/data/Users/izumi/cospa/result/2008/2008-testdata/undergraduate-thesis/resnet18/patchsize127/epoch800/seg/gt_prob08/p4e3/binary_filter_after_crf/*.jpg"
# save_dir = "/data/Users/izumi/cospa/result/2008/2008-testdata/undergraduate-thesis/resnet18/patchsize127/epoch800/seg/gt_prob08/p4e3/binary_filter_after_crf"

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

    mIou = np.mean(iou)
    mDice = np.mean(dice)
    return iou, dice, mIou, mDice


def test(gt_path, predict_data_path):
    gt_images_path, prediction_images_path = getImagesPath(gt_path, predict_data_path)
    gt_images, predict_images = importData(gt_images_path, prediction_images_path)
    iou, dice, mIou, mDice = calculateIouDice(gt_images, predict_images)
    
    return iou, dice, mIou, mDice

def main():
    gt_path = "/data/Users/izumi/cospa/dataset-all/test/2008-testdata/masks/*.jpg"
    # predict_path = [
    #             '/data/Users/izumi/cospa/result/2008/2008-testdata/undergraduate-thesis/resnet18/patchsize127/epoch800/seg/gt_prob08/p4e3/binary_filter_after_crf/*.jpg',
    #             '/data/Users/izumi/cospa/result/2008/2008-testdata/undergraduate-thesis/resnet18/patchsize127/epoch800/seg/gt_prob07/p4e3/binary_filter_after_crf/*.jpg',
    #             '/data/Users/izumi/cospa/result/2008/2008-testdata/undergraduate-thesis/resnet18/patchsize127/epoch800/seg/gt_prob06/p4e3/binary_filter_after_crf/*.jpg',
    #             '/data/Users/izumi/cospa/result/2008/2008-testdata/undergraduate-thesis/resnet18/patchsize127/epoch800/seg/gt_prob05/p4e3/binary_filter_after_crf/*.jpg',
    #             '/data/Users/izumi/cospa/result/2008/2008-testdata/undergraduate-thesis/resnet18/patchsize127/epoch800/seg/gt_prob04/p4e3/binary_filter_after_crf/*.jpg',
    #             '/data/Users/izumi/cospa/result/2008/2008-testdata/undergraduate-thesis/resnet18/patchsize127/epoch800/seg/gt_prob03/p4e3/binary_filter_after_crf/*.jpg',
    #             '/data/Users/izumi/cospa/result/2008/2008-testdata/undergraduate-thesis/resnet18/patchsize127/epoch800/seg/gt_prob02/p4e3/binary_filter_after_crf/*.jpg',
    #             '/data/Users/izumi/cospa/result/2008/2008-testdata/undergraduate-thesis/resnet18/patchsize127/epoch800/seg/gt_prob01/p4e3/binary_filter_after_crf/*.jpg'
    #             ]
    # predict_path = [
    #     '/data/Users/izumi/cospa/result/2008/2006-testdata/resnet18/patchsize127/epoch800/seg/gt_prob08/p4e3/binary_filter_after_crf/*.jpg',
    #     '/data/Users/izumi/cospa/result/2008/2006-testdata/resnet18/patchsize127/epoch800/seg/gt_prob07/p4e3/binary_filter_after_crf/*.jpg',
    #     '/data/Users/izumi/cospa/result/2008/2006-testdata/resnet18/patchsize127/epoch800/seg/gt_prob06/p4e3/binary_filter_after_crf/*.jpg',
    #     '/data/Users/izumi/cospa/result/2008/2006-testdata/resnet18/patchsize127/epoch800/seg/gt_prob05/p4e3/binary_filter_after_crf/*.jpg',
    #     '/data/Users/izumi/cospa/result/2008/2006-testdata/resnet18/patchsize127/epoch800/seg/gt_prob04/p4e3/binary_filter_after_crf/*.jpg',
    #     '/data/Users/izumi/cospa/result/2008/2006-testdata/resnet18/patchsize127/epoch800/seg/gt_prob03/p4e3/binary_filter_after_crf/*.jpg',
    #     '/data/Users/izumi/cospa/result/2008/2006-testdata/resnet18/patchsize127/epoch800/seg/gt_prob02/p4e3/binary_filter_after_crf/*.jpg',
    #     '/data/Users/izumi/cospa/result/2008/2006-testdata/resnet18/patchsize127/epoch800/seg/gt_prob01/p4e3/binary_filter_after_crf/*.jpg'
    # ]
    predict_path = [
        '/data/Users/izumi/cospa/result/2008/2009-testdata/resnet18/patchsize127/epoch800/seg/gt_prob08/p4e3/binary_filter_after_crf/*.jpg',
        '/data/Users/izumi/cospa/result/2008/2009-testdata/resnet18/patchsize127/epoch800/seg/gt_prob07/p4e3/binary_filter_after_crf/*.jpg',
        '/data/Users/izumi/cospa/result/2008/2009-testdata/resnet18/patchsize127/epoch800/seg/gt_prob06/p4e3/binary_filter_after_crf/*.jpg',
        '/data/Users/izumi/cospa/result/2008/2009-testdata/resnet18/patchsize127/epoch800/seg/gt_prob05/p4e3/binary_filter_after_crf/*.jpg',
        '/data/Users/izumi/cospa/result/2008/2009-testdata/resnet18/patchsize127/epoch800/seg/gt_prob04/p4e3/binary_filter_after_crf/*.jpg',
        '/data/Users/izumi/cospa/result/2008/2009-testdata/resnet18/patchsize127/epoch800/seg/gt_prob03/p4e3/binary_filter_after_crf/*.jpg',
        '/data/Users/izumi/cospa/result/2008/2009-testdata/resnet18/patchsize127/epoch800/seg/gt_prob02/p4e3/binary_filter_after_crf/*.jpg',
        '/data/Users/izumi/cospa/result/2008/2009-testdata/resnet18/patchsize127/epoch800/seg/gt_prob01/p4e3/binary_filter_after_crf/*.jpg'
    ]
    

    for path in predict_path:
        print(path)
        iou, dice, mIou, mDice = test(gt_path, path)
        print("IoU: ",iou)
        print("Dice: ", dice)
        print("mean_Iou: ", mIou)
        print("mean_Dice: ", mDice)

if __name__ == '__main__':
        main()


# def makeAndSaveGraph(iou, dice, save_dir_path) -> None:
#     x = [i for i in range(12)]#range(len(predict_path))
#     plt.plot(x, iou, linestyle="solid", marker="o")
#     plt.plot(x, dice, linestyle="dashed", marker="x")
#     plt.xticks(x, ['p1e1','p1e3','p1e5','default', 'p2e3', 'p2e5','p3e1', 'p3e3','p3e5','p4e1','p4e3','p4e5'])
#     plt.savefig(save_dir_path + "/iou.jpg")
#     output_data = [x, iou, dice]
#     np.savetxt(save_dir_path + "/iou_each_image.csv", output_data, delimiter=",", fmt='%s')