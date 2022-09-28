# requirement: install nibabel
import os
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import numpy as np

def load_nii(dir_path, mode):
    
    folder_repository = {
        'train_1': 'Training Batch 1',
        'train_2': 'Training Batch 2',
        'train_raw': 'imagesTr',
        'train_gt': 'labelsTr'
        }
    
    folder = folder_repository[mode]
    
    dir_path = os.path.join(dir_path, folder)
    filenames = os.listdir(dir_path)
    filenames.sort()
    
    # delete the element 'DS_Store'
    for j, x in enumerate(filenames):
        if x.startswith('.'):
            filenames.pop(j)
        
    for file in filenames:
        file_path = os.path.join(dir_path, file)
        img = nib.load(file_path)          # read nii
        img_fdata = img.get_fdata()
        fname = file.replace('.nii', '')

        
        # img_f_path = os.path.join('/home/zhaoxiang/LiTs/slices', fname)
        img_f_path = '/home/zhaoxiang/LiTs/slices'
        
        if not os.path.exists(img_f_path):
            os.mkdir(img_f_path)
        
        if 'volume' in file:
            save_path = os.path.join(img_f_path, 'raw')
            a = 'raw'
            number = int(fname.replace('volume-', ''))
            
        elif 'segmentation' in file:
            save_path = os.path.join(img_f_path, 'gt')
            a = 'gt'
            number = int(fname.replace('segmentation-', ''))
            
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            
            
        (x,y,z) = img.shape
        for i in range(z):
            gray = img_fdata[:,:,i]
            matplotlib.image.imsave(os.path.join(save_path, 'liver_{}_{}_{}'.format(number, i, a) + '.png'), gray, cmap='gray')
        print(number, '    done')
            
            
dir_path = '/home/zhaoxiang/LiTs'

mode = 'train_1'
# mode = 'train_2'

load_nii(dir_path, mode)

