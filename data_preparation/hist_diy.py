import numpy as np 

def imhist(im):
  # calculates normalized histogram of an image
	m, n = im.shape
	h = [0.0] * 256
	for i in range(m):
		for j in range(n):
			h[im[i, j]]+=1
	return np.array(h)/(m*n)

def cumsum(h):
	# finds cumulative sum of a numpy array, list
	return [sum(h[:i+1]) for i in range(len(h))]

def histeq(im):
	#calculate Histogram
	h = imhist(im)
	cdf = np.array(cumsum(h)) #cumulative distribution function
	sk = np.uint8(255 * cdf) #finding transfer function values
	s1, s2 = im.shape
	Y = np.zeros_like(im)
	# applying transfered values for each pixels
	for i in range(0, s1):
		for j in range(0, s2):
			Y[i, j] = sk[im[i, j]]
	H = imhist(Y)
	#return transformed image, original and new istogram, 
	# and transform function
	return Y , h, H, sk


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


import cv2
import os

source_path = '/home/zhaoxiang/dataset/Atlas_train+LiTs_test/train/good'
dst_path = '/home/zhaoxiang/dataset/Atlas_train+LiTs_test_hist_DIT_crop/train/good'
# source_path = '/home/zhaoxiang/dataset/initial_version/test'
# dst_path = '/home/zhaoxiang/dataset/hist_DIY/test'
names = os.listdir(source_path)


for old_name in names:
    old_path = os.path.join(source_path, old_name)
    gray = cv2.imread(old_path, cv2.IMREAD_GRAYSCALE)
    hist_DIY = learning(gray)
    name = old_path.split('/')[-1].replace('.png', '_DIY_hist.png')
    new_path = os.path.join(dst_path, name)
    cv2.imwrite(new_path, hist_DIY)
    
print('done')
