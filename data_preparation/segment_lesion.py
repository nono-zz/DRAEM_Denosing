# implement based on 'load_nii.py' file
# should prepare the sliced the 2D images in advance

# test if pixel-wise product works for image segmentation

import cv2
import os
import numpy as np

root_path = '/home/zhaoxiang/dataset/test'            # for MSD
# root_path = '/home/zhaoxiang/dataset/data_normal'            # for Altas

folders = os.listdir(root_path)
folders.sort()
for folder in folders:
    files_path = os.path.join(root_path, folder, 'raw')
    files_gt_path = os.path.join(root_path, folder, 'gt')
    # read the files in the folder
    files = os.listdir(files_path)
    files.sort()

    for file in files:
        if file.startswith('.'):
            continue
        if file.startswith('61'):
            jls_extract_var = 'found'
            print(jls_extract_var)
        
        number = file.split('_')[0]
        
        # read the raw image
        raw_path = os.path.join(files_path, file)
        raw_img = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE) # img.dtype('uint8')
        
        # read ground truth
        gt_name = number + '_gt.png'
        gt_path = os.path.join(files_gt_path, gt_name)
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        if np.max(gt_img) != 0:
            if 128 in gt_img:           # means that image has lesion
                mask = np.where(gt_img > 250, 255, 0)
            else:
                mask = np.full_like(gt_img, 0)      # ll fill out with 0, all black
            
            # save the image
            new_folder = os.path.join(root_path, folder, 'label')
            if not os.path.exists(new_folder):
                os.mkdir(new_folder)
                
            new_img_name = number + '_label.png'
            new_img_path = os.path.join(new_folder, new_img_name)
            
            
            # cv2.imwrite(new_img_path, liver_img)
            cv2.imwrite(new_img_path, mask)

        else:
            continue

    print(folder, '  done')