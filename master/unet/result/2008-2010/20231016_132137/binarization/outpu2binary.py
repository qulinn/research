# test.jpg : /data/Users/izumi/unet/2008-2010/data/membrane/results/20231016_132137/0_predict.png


import cv2
import numpy as np
import glob
import os


SAVE_DIR = "/data/Users/izumi/unet/2008-2010/data/membrane/results/20231016_132137/binarization"
DIR_PATH = "/data/Users/izumi/unet/2008-2010/data/membrane/results/20231016_132137"


def getFiles(DIR_PATH):
    # TODO: get pathes
    # /data/Users/izumi/unet/2008-2010/data/membrane/results/20231016_132137
    #
    image_paths = glob.glob(DIR_PATH + "/*.png")
    image_paths.sort()

    # filenames = []
    # for path in image_paths:
    #     filenames.append(os.path.basename(path))

    paths = []
    for path in image_paths:
        if not (path.endswith("acc.png") or path.endswith("loss.png")):
            paths.append(path)

    return paths


def get_brightness(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("画像を読み込めません。")
        return

    height, width = image.shape

    print(image[0, 0])

    max_brightness = 0
    for y in range(height):
        for x in range(width):
            pixel = image[y, x]
            # 輝度値の計算方法は、RGBの平均値を使うことができます
            brightness = pixel  # RGBの平均値
            max_brightness = max(max_brightness, pixel)
            # print(f"Pixel at ({x}, {y}) - Brightness: {brightness}")

    print("Max Brightness: ", max_brightness)
    return max_brightness, height, width


def drawBinaryImage(image_path):
    # 画像を白黒で読み込む
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("画像を読み込めません。")
        return

    height, width = image.shape

    # 最大輝度値を見つける
    max_brightness = np.max(image)

    # 輝度値が130を超えるピクセルを1に設定
    binary_image = np.where(image >= 130, 1, 0)

    # バイナリ画像を保存
    cv2.imwrite(
        SAVE_DIR + "/" + os.path.basename(image_path) + ".png", binary_image * 255
    )  # バイナリ画像を保存するときに0と1を255と0に変換


if __name__ == "__main__":
    # image_path = "/data/Users/izumi/unet/test.png"  # 画像ファイルのパスを指定してください
    image_paths = getFiles(DIR_PATH)

    for image_path in image_paths:
        drawBinaryImage(image_path)
        print(image_path)

    # max_brightness, height, width = get_brightness(image_path)
