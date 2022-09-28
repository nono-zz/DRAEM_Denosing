# implement based on 'load_nii.py' file
# should prepare the sliced the 2D images in advance

# test if pixel-wise product works for image segmentation

import cv2
import os
import numpy as np

root_path = '/home/zhaoxiang/dataset/abdnomial_images/data_lesion'            # for MSD
# root_path = '/home/zhaoxiang/dataset/abdnomial_images/data_normal'            # for Altas

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
        
        number = file.split('_')[0]
        
        # read the raw image
        raw_path = os.path.join(files_path, file)
        raw_img = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE) # img.dtype('uint8')
        
        # read ground truth
        gt_name = number + '_gt.png'
        gt_path = os.path.join(files_gt_path, gt_name)
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        
        # raw_img histogram equalization
        hist_img = cv2.equalizeHist(raw_img)
        
        
        if np.max(gt_img) != 0:
            # generate the mask and produce the liver image
            mask = np.where(gt_img > 0 , 1, 0)
            liver_img = np.multiply(mask, hist_img)
            
            # save the image
            new_folder = os.path.join(root_path, folder, 'hist_segmentation_flip')
            if not os.path.exists(new_folder):
                os.mkdir(new_folder)
                
            new_img_name = number + '_hist_segmentation_flip.png'
            new_img_path = os.path.join(new_folder, new_img_name)
            
            # flip the image (vertically)
            if root_path.endswith('normal'):
                flipped = cv2.flip(liver_img,0)
                
                # 
                cv2.imwrite(new_img_path, flipped)
            elif root_path.endswith('lesion'):
                cv2.imwrite(new_img_path, liver_img)
        else:
            continue
    print(folder, '  done')