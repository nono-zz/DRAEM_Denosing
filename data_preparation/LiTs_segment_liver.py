import os
import cv2
import numpy as np


# delete non liver slices
gt_dir = '/home/zhaoxiang/LiTs/slices/gt'
raw_dir = '/home/zhaoxiang/LiTs/slices/raw'
seg_dir = '/home/zhaoxiang/LiTs/slices/raw_segment'
label_dir = '/home/zhaoxiang/LiTs/slices/tumor_labels'

gt_files = os.listdir(gt_dir)
gt_files.sort()

for gt_file in gt_files:
    gt_path = os.path.join(gt_dir, gt_file)
    gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    # if np.max(gt_img) == 0:
    #     os.remove(gt_path)
    
    # save tumor label
    if 128 in gt_img:
        label = np.where(gt_img == 255, 255, 0).astype(np.uint8)
    else:
        label = gt_img * 0
        
    label_name = gt_file
    label_path = os.path.join(label_dir, label_name)
    
    cv2.imwrite(label_path, label)
    
    # save segment raw img
    img_name = gt_file.replace('gt', 'raw')
    img_path = os.path.join(raw_dir, img_name)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    seg_img = np.where(gt_img > 0, img, 0)
    
    seg_name = img_name.replace('raw', 'segmentation')
    seg_path = os.path.join(seg_dir, seg_name)
    cv2.imwrite(seg_path, seg_img)
    
        
    