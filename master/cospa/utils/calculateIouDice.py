import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt



def getImagesPath(CORRECT_PATH, PREDICT_PATH):
    correct_path = glob.glob(CORRECT_PATH)
    predict_path = glob.glob(PREDICT_PATH)
    correct_path.sort()
    predict_path.sort()

    return correct_path, predict_path



def importData(correct_path, predict_path):
    correct_images = [cv2.imread(path, 0) for path in correct_path]
    predict_images = [cv2.imread(path, 0) for path in predict_path]

    return correct_images, predict_images



def culcIouDice(correct_images, predict_images):
    height, width = len(correct_images[0]), len(correct_images[0])
    iou = list()
    dice = list()

    for i in range(len(correct_images)):
        correct = correct_images[i]
        predict = predict_images[i]

        tp, fp, fn = 0, 0, 0
        for j in range(height):
            for k in range(width):
                #predict == 1
                if all(predict[j][k] != np.array([0, 0, 0])):
                    #correct == 1
                    if all(correct[j][k] != np.array([0, 0, 0])):
                        tp += 1
                    else:
                        #correct == 0
                        fp += 1
                #correct == 1
                elif all(correct[j][k] != np.array([0, 0, 0])):
                    #predict == 0
                    if all(predict[j][k] == np.array([0, 0, 0])):
                        fn += 1

        if tp == 0 and fp == 0 and fn == 0:
            raise Exception("error!")

        temp_iou = tp / (tp + fp + fn)
        temp_dice = 2 * tp / (2 * tp + fp + fn)
        iou.append(temp_iou)
        dice.append(temp_dice)
    iou_mean = np.mean(iou)
    dice_mean = np.mean(dice)
    return iou_mean, dice_mean


def makeAndSaveGraph(iou, dice, save_dir_path) -> None:
    x = [i for i in range(12)]#range(len(predict_path))
    plt.plot(x, iou, linestyle="solid", marker="o")
    plt.plot(x, dice, linestyle="dashed", marker="x")
    plt.xticks(x, ['p1e1','p1e3','p1e5','default', 'p2e3', 'p2e5','p3e1', 'p3e3','p3e5','p4e1','p4e3','p4e5'])
    plt.savefig(save_dir_path + "/iou.jpg")
    output_data = [x, iou, dice]
    np.savetxt(save_dir_path + "/iou.csv", output_data, delimiter=",", fmt='%s')

def test(correct_path, predict_path) -> None:
    correct_iamges_path, predict_images_path = getImagesPath(correct_path, predict_path)
    correct_images, predict_images = importData(correct_iamges_path, predict_images_path)
    iou_mean_per_model, dice_mean_per_model = culcIouDice(correct_images, predict_images)
    return iou_mean_per_model, dice_mean_per_model




def main():
    correct_path = 'GT_PATH'
    predict_path = ["MODEL'S PREDICTION"]
    save_dir_path = "SAVE_DIR_PATH"

    
    
    iou_all = []
    dice_all = []
    for i in range(len(predict_path)):
        iou_per_model, dice_per_model = test(correct_path, predict_path[i])
        iou_all.append(iou_per_model)
        dice_all.append(dice_per_model)
    makeAndSaveGraph(iou_all, dice_all, save_dir_path)


    

    
if __name__ == '__main__':
    main()