import cv2
import numpy as np


def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)


# img_path = '/home/zhaoxiang/experiments_zzx/reconstruction/outputs/nonrecon_imagenet/hist_DIY_crop_blur/1e-3/liver_4_457/a_map_102.png'
img_path = '/home/zhaoxiang/experiments_zzx/reconstruction/outputs/nonrecon_imagenet/hist_DIY_crop_blur/1e-3/liver_10_379/a_map_162.png'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

img = min_max_norm(img)

cv2.imwrite('min_max_norm_imgNet_4.png', img * 255)