import os
import cv2
import numpy as np

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
			# transform_map[i] = np.floor((transform_map[i]/150*100 + 50))

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


raw_dir = '/home/zhaoxiang/datasets_raw/LiTs/slices/raw_segment'
label_dir = '/home/zhaoxiang/LiTs/slices/tumor_labels'



# dst_img_dir = '/home/zhaoxiang/LiTs/slices/hist_crop'
# dst_gt_dir = '/home/zhaoxiang/LiTs/slices/hist_crop_gt'

dst_img_dir = '/home/zhaoxiang/datasets_raw/LiTs/slices/hist_DIY'
# dst_gt_dir = '/home/zhaoxiang/LiTs/slices/hist_crop_gt'

names = os.listdir(raw_dir)
names.sort()

for old_name in names:
    old_path = os.path.join(raw_dir, old_name)
    gray = cv2.imread(old_path, cv2.IMREAD_GRAYSCALE)
    hist_DIY = learning(gray)
    # x, y = np.nonzero(hist_DIY)
    # x_min, x_max = x.min(), x.max()
    # y_min, y_max = y.min(), y.max()
    # img_crop = hist_DIY[x_min:x_max+1, y_min:y_max+1]
    
    # gt_name = old_name.replace('segmentation', 'gt')
    # gt_path = os.path.join(label_dir, gt_name)
    
    # gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    # gt_img_crop = gt_img[x_min:x_max+1, y_min:y_max+1]
    
    # # save crop img
    # name = old_path.split('/')[-1].replace('.png', '_DIY_hist_crop.png')
    # new_path = os.path.join(dst_img_dir, name)
    # cv2.imwrite(new_path, img_crop)
    
    # # save cropped gt
    # gt_new_name = gt_name.replace('.png', '_DIY_hist_crop.png')
    # gt_new_path = os.path.join(dst_gt_dir, gt_new_name)
    # cv2.imwrite(gt_new_path, gt_img_crop)
    
    
    # save hist_DIT image
    name = old_path.split('/')[-1].replace('.png', '_DIY_hist.png')
    new_path = os.path.join(dst_img_dir, name)
    cv2.imwrite(new_path, hist_DIY)
print('done')
