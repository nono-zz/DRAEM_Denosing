# requirement: install nibabel
from asyncio.constants import DEBUG_STACK_DEPTH
import os
from re import I
from symbol import del_stmt
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import numpy as np
import cv2

def load_nii(dir_path):

    filenames = os.listdir(dir_path)
    filenames.sort()
    
    # delete the element 'DS_Store'
    for j, x in enumerate(filenames):
        if x.startswith('.'):
            filenames.pop(j)
        
    # randomly split the training, validation, test set
    # 938 training, 62 validation, and 251 test patients
    training_filenames = filenames[:938]
    validation_filenames = filenames[938:938+62]
    testing_filenames = filenames[938+62:938+62+251]
    tot_filenames = [training_filenames, validation_filenames, testing_filenames]
    mode = ['train', 'val', 'test']
    
    # tot_filenames = [validation_filenames, testing_filenames]
    # mode = ['val', 'test']
    
    for index, filenames in enumerate(tot_filenames):
        for filename in filenames:
            file_path = os.path.join(dir_path, filename)
            
            flair_name = filename + '_flair.nii.gz'
            
            flair_path = os.path.join(file_path, flair_name)
            flair_img = nib.load(flair_path)          # read nii
            img_fdata = flair_img.get_fdata()
            
            img_f_path = os.path.join('/home/zhaoxiang/dataset/BraTs/{}'.format(mode[index]))
            if not os.path.exists(img_f_path):
                os.mkdir(img_f_path)
                
            # (x,y,z) = flair_img.shape
            # for i in range(z):
            #     gray = img_fdata[:,:,i]
            #     matplotlib.image.imsave(os.path.join(img_f_path, '{}_{}_flair'.format(filename, i) + '.png'), gray, cmap='gray')
            # print(i, '    done')
            
            
            """deal with ground truth"""
            gt_name = filename + '_seg.nii.gz'
            
            gt_path = os.path.join(file_path, gt_name)
            gt_img = nib.load(gt_path)          # read nii
            img_gt_data = gt_img.get_fdata()
            
            img_gt_path = os.path.join('/home/zhaoxiang/dataset/BraTs/{}_gt'.format(mode[index]))
            if not os.path.exists(img_gt_path):
                os.mkdir(img_gt_path)
               
               
            """save the flair image and the ground truth image"""
            if mode[index] == 'train':
                
                (x,y,z) = gt_img.shape
                for i in range(z):
                    gray_gt = img_gt_data[:,:,i]
                    
                    if np.sum(gray_gt) > 0:
                        label = 'anomaly'
                        continue
                    else:
                        label = 'good'
                        # matplotlib.image.imsave(os.path.join(img_gt_path, '{}_{}_gt'.format(filename, i) + '.png'), gray_gt, cmap='gray')
                        gray = img_fdata[:,:,i]
                        matplotlib.image.imsave(os.path.join(img_f_path, '{}_{}_flair'.format(filename, i) + '.png'), gray, cmap='gray')
            else:
                (x,y,z) = gt_img.shape
                for i in range(z):
                    gray_gt = img_gt_data[:,:,i]
                    matplotlib.image.imsave(os.path.join(img_gt_path, '{}_{}_gt'.format(filename, i) + '.png'), gray_gt, cmap='gray')
                    
                    gray = img_fdata[:,:,i]
                    matplotlib.image.imsave(os.path.join(img_f_path, '{}_{}_flair'.format(filename, i) + '.png'), gray, cmap='gray')
            
            print(filename, '    done')


def learning(img_array):

	#flatten image array and calculate histogram via binning
	histogram_array = np.bincount(img_array.flatten(), minlength=256)
 
	histogram_array[0] = 0
	histogram_array = histogram_array/np.linalg.norm(histogram_array)

	#normalize
	num_pixels = np.sum(histogram_array)
	histogram_array = histogram_array/num_pixels

	#normalized cumulative histogram
	chistogram_array = np.cumsum(histogram_array)


	"""
	STEP 2: Pixel mapping lookup table
	"""
	transform_map = np.floor(255 * chistogram_array).astype(np.uint8)
 
	# 把transform_map里的 0 到 150 拉伸到 30 到 150 之间
	for i in range(len(transform_map)):
		if transform_map[i] > 0 and transform_map[i] < 150:

			transform_map[i] = np.floor((transform_map[i]/150*120 + 30))
			# transform_map[i] = np.floor((transform_map[i]/100*50 + 50))

		# elif transform_map[i] > 200:
		# 	transform_map[i] = 200
   

	"""
	STEP 3: Transformation
	"""
	# flatten image array into 1D list
	img_list = list(img_array.flatten())

	# transform pixel values to equalize
	eq_img_list = [transform_map[p] for p in img_list]

	# reshape and write back into img_array
	eq_img_array = np.reshape(np.asarray(eq_img_list), img_array.shape)
 
	return eq_img_array
            
            
def preprocess(source_path, dst_path, gt_source_path, gt_dst_path):
    names = os.listdir(source_path)
    names.sort()
    for old_name in names:
        old_path = os.path.join(source_path, old_name)
        gray = cv2.imread(old_path, cv2.IMREAD_GRAYSCALE)
        
        # abandon empty images
        if np.sum(gray)==0:
            continue
        else:
            hist_DIY = learning(gray)
            name = old_path.split('/')[-1].replace('.png', '_DIY_hist.png')
            new_path = os.path.join(dst_path, name)
            cv2.imwrite(new_path, hist_DIY)
            
            if not 'train' in old_path:
        
                # Unify the ground truth
                gt_name = old_path.split('/')[-1].replace('flair', 'gt')
                gt_old_path = os.path.join(gt_source_path, gt_name)
                gt_new_path = os.path.join(gt_dst_path, gt_name)
                gtUnify(gt_old_path, gt_new_path)

import os
def deleteEmpty(source_path):
    names = os.listdir(source_path)
    names.sort()

    for old_name in names:
        old_path = os.path.join(source_path, old_name)
        gray = cv2.imread(old_path, cv2.IMREAD_GRAYSCALE)
        
        # abandon empty images
        if np.sum(gray)==0:
            os.remove(old_path) 
            if 'val' in source_path:
                gt_path = old_path.replace('val', 'val_label').replace('flair', 'gt')
                os.remove(gt_path)
            if 'test' in source_path:
                gt_path = old_path.replace('test', 'test_label').replace('flair', 'gt')
                os.remove(gt_path)
        else:
            continue

# deleteEmpty('/home/zhaoxiang/dataset/BraTs/train/good')
# deleteEmpty('/home/zhaoxiang/dataset/BraTs/test')

from random import sample
import shutil
def subsampleTest(source_path, dst_path):
    

    names = os.listdir(source_path)
    names.sort()
    print('test queue length before subsampling', len(names))           # 34800

    ## subsampling to 1/30 length of the orginal test set

    subsampleNames = sample(names, int(len(names)/30))
    for name in subsampleNames:
        img_name = name.replace('_flair.png', '') 
        img_path = os.path.join(source_path, name)
        gt_path = os.path.join(source_path.replace('rest', 'rest_label'), img_name+'_gt.png')

        # move the files to the dst_path\
        new_img_path = os.path.join(dst_path, name)
        new_gt_path = os.path.join(dst_path.replace('test', 'test_label'), img_name+'_gt.png')

        shutil.copyfile(img_path, new_img_path)
        shutil.copyfile(gt_path, new_gt_path)
        

subsampleTest('/home/zhaoxiang/dataset/BraTs/rest', '/home/zhaoxiang/dataset/BraTs/test')
        
def gtUnify(gt_old_path, gt_new_path):
    gray = cv2.imread(gt_old_path, cv2.IMREAD_GRAYSCALE)
    grayUnified = np.where(gray>0, 255, 0)
    cv2.imwrite(gt_new_path, grayUnified)


def gt_process(source_path, dst_path):
    dirs = ['val_gt', 'test_gt']
    for dirname in dirs:
        dir_path = os.path.join(source_path, dirname)
        new_dir_path = os.path.join(dst_path, dirname)
        if not os.path.exists(new_dir_path):
            os.makedirs(new_dir_path, exist_ok=True)

        filenames = os.listdir(dir_path)
        filenames.sort()
        for filename in filenames:
            old_path = os.path.join(dir_path, filename)
            gray = cv2.imread(old_path, cv2.IMREAD_GRAYSCALE)
            new_gray = np.where(gray>0, 255, 0)
            new_path = os.path.join(new_dir_path, filename)

            cv2.imwrite(new_path, new_gray)


# gt_process('/home/zhaoxiang/dataset/BraTs_multi_Gt', '/home/zhaoxiang/dataset/BraTs')
    
    
            
## slice the 2Ds          
# dir_path = '/home/datasets/BraTs/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021'
# dir_path = '/home/zhaoxiang/dataset/brain/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021'
# load_nii(dir_path)


# ### hist_DIY images
# modes = ['train', 'val', 'test']
# # modes = ['val', 'test']
# for mode in modes:
#     source_path = '/home/zhaoxiang/dataset/BraTs/{}'.format(mode)
#     dst_path = '/home/zhaoxiang/dataset/BraTs_histDIY/{}'.format(mode)
    
#     gt_source_path = '/home/zhaoxiang/dataset/BraTs/{}_gt'.format(mode)
#     gt_dst_path = '/home/zhaoxiang/dataset/BraTs_histDIY/{}_gt'.format(mode)
#     if not os.path.exists(dst_path):
#         os.makedirs(dst_path, exist_ok=True)
        
#     if not os.path.exists(gt_dst_path):
#         os.makedirs(gt_dst_path, exist_ok=True)
    
#     preprocess(source_path, dst_path, gt_source_path, gt_dst_path)
        
# ### unify all the gt colors
# for mode in modes:
#     source_path = '/home/zhaoxiang/dataset/BraTs/{}_gt'.format(mode)
#     dst_path = '/home/zhaoxiang/dataset/BraTs_histDIY/{}_gt'.format(mode)
#     if not os.path.exists(dst_path):
#         os.makedirs(dst_path, exist_ok=True)
        
#     gtUnify(source_path, dst_path)
        
        




