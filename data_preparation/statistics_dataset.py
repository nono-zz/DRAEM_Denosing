"""[summary]
    this script is created to compute the mean and std of my own dataset
"""

import os
import cv2
import numpy as np

root = '/home/zhaoxiang/dataset'
phases = ['train', 'test']


def find_mean_std(phase):
    
    pixel_count = []
    sum = []
    sum_sq = []
    mean_img = []
    
    if phase == 'train':
        folder_path = os.path.join(root, phase, 'good')
    else:
        folder_path = os.path.join(root, phase)
    files = os.listdir(folder_path)
    files.sort()
    # first have a look at the cv2.imwrite funcion, check what's the value range of an image
    for file in files:
        file_path = os.path.join(folder_path, file)
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        pixel_count.append(cv2.countNonZero(img))
        
        sum_img = np.sum(img/255)           # normalize to 0~1
        sum.append(sum_img)
        
        img_sq = np.square(img/255)
        sum_sq.append(np.sum(img_sq))
        
        mean_img.append(sum_img/cv2.countNonZero(img))
    
    total_value = np.sum(sum)
    total_pixel = np.sum(pixel_count)
    total_sq = np.sum(sum_sq)
    
    # E[x]
    mean = total_value/total_pixel
    mean_sq = total_sq/total_pixel
    
    # V[x]
    std = mean_sq - mean**2
    std = np.sqrt(std)
    
    # save the results
    lines = phase + '   mean: ' + str(mean) + ",   std: " + str(std) + '\n'
    with open('statistics.txt', 'a') as f:
        f.writelines(lines)
        
    print(lines)
    
    return total_value, total_pixel, total_sq

total_value_train, total_pixel_train, total_sq_train = find_mean_std('train')        
total_value_test, total_pixel_test, total_sq_test = find_mean_std('test') 

def compute_mean_std(total_value, total_pixel, total_sq):
    # E[x]
    mean = total_value/total_pixel
    mean_sq = total_sq/total_pixel
    
    # V[x]
    std = np.sqrt(mean_sq - mean**2)
    
    return mean, std

# compute the total mean and std

total_value = total_value_test + total_value_train
total_pixel = total_pixel_test + total_pixel_train
total_sq = total_sq_test + total_sq_train

mean, std = compute_mean_std(total_value, total_pixel, total_sq)

lines = 'Total' + '   mean: ' + str(mean) + ",   std: " + str(std) + '\n'

print(lines)

with open('statistics.txt', 'a') as f:
    f.writelines(lines)
