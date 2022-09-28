# requirement: install nibabel
import os
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import numpy as np

def load_nii(dir_path, mode):
    
    folder_repository = {
        'test_raw': 'imagesTs',
        'test_gt': 'labelsTs',
        'train_raw': 'imagesTr',
        'train_gt': 'labelsTr'
        }
    
    folder = folder_repository[mode]
    
    dir_path = '/home/zhaoxiang/MSD_test/' + folder
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
        fname = file.replace('.nii.gz', '')
        
        img_f_path = os.path.join('/home/zhaoxiang/MSD_test/data_lesion', fname)
        
        if not os.path.exists(img_f_path):
            os.mkdir(img_f_path)
        
        if mode.endswith('raw'): 
            save_path = os.path.join(img_f_path, 'raw')
            a = 'raw'
        else:
            save_path = os.path.join(img_f_path, 'gt')
            a = 'gt'
            
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            
            
        (x,y,z) = img.shape
        for i in range(z):
            gray = img_fdata[:,:,i]
            matplotlib.image.imsave(os.path.join(save_path, '{}_'.format(i) + a + '.png'), gray, cmap='gray')
        print(save_path, '    done')
            
            
dir_path = '/home/zhaoxiang/MSD_test'

# mode = 'train_raw'
mode = 'train_gt'
# mode = 'test_raw'
# mode = 'test_gt'

load_nii(dir_path, mode)

