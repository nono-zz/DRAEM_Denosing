import pydicom as dicom
import os
import numpy as np
import cv2
from matplotlib import pyplot, cm
from pathlib import Path


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
 
	# 把transform_map里的 0 到 150 拉伸到 30 到 255 之间
	for i in range(len(transform_map)):
		if transform_map[i] > 0 and transform_map[i] < 150:
    
			transform_map[i] = np.floor((transform_map[i]/150*120 + 30))
   




  

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



PathDicom = "/home/zhaoxiang/datasets_raw/CHAOs/Train_Sets/"
lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName,filename))

# print(lstFilesDCM)

# Get ref file
lstFilesDCM.sort()
for i, image in enumerate(lstFilesDCM):
    ds = dicom.read_file(lstFilesDCM[i])

    pixel_array_numpy = ds.pixel_array
    
    # transfer from uint 16 to 8
    # pixel_array_numpy = (pixel_array_numpy/256).astype(numpy.uint8)
    # min_max_norm
    # pixel_array_numpy = ((pixel_array_numpy - pixel_array_numpy.min())/(pixel_array_numpy.max() - pixel_array_numpy.min()) * 256).astype(numpy.uint8)
    
    pixel_array_numpy = ((pixel_array_numpy - pixel_array_numpy.min())/(pixel_array_numpy.max() - pixel_array_numpy.min()) * 256)
    pixel_array_numpy = pixel_array_numpy.astype(np.uint8)
    
    if 'MR' in image:
        continue
    
    # if '23' in image:
        # print('found')
    
    image = image.replace('.dcm', '.png')
    image_name = image.split('/')[-1]
    dir_number = image.split('/')[-3]
    
    image_number = image_name[2:5]
    if image_name.startswith('IMG'):
        image_number = image_name[-7:-4]
    gt_image_name = 'liver_GT_' + image_number + '.png'
    
    gt_path = image.replace('DICOM_anon', 'Ground')
    gt_path = gt_path.replace(image_name, gt_image_name)
    # image = image.replace('Train_Sets', 'Train_slice')
    

    # segmen liver
    # /home/zhaoxiang/datasets_raw/CHAOs/Train_Sets/CT/1/Ground/liver_GT_000.png
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    if not os.path.exists(gt_path):
        print(image, 'gt path doent exist')
        continue
    
    if np.max(gt) != 0:
        mask = np.where(gt > 0 , 1, 0)
        liver_img = np.multiply(mask, pixel_array_numpy)
        
        # flip horizontaly
        liver_img = cv2.flip(liver_img, 1)
        
        # histogram equalization
        hist_DIY = learning(liver_img)
        
        # crop
        # x, y = np.nonzero(hist_DIY)
        # x_min, x_max = x.min(), x.max()
        # y_min, y_max = y.min(), y.max()
        # img_crop = hist_DIY[x_min:x_max+1, y_min:y_max+1]
        
    
    
    
        save_dir = os.path.join('/home/zhaoxiang/datasets_raw/CHAOs/Train_segment', dir_number)
        save_path = os.path.join('/home/zhaoxiang/datasets_raw/CHAOs/Train_segment', dir_number, image_name)
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(save_path, hist_DIY)
        print('done')
    
    


