import cv2
import numpy as np
import os
from scipy.ndimage import gaussian_filter


# methods

# blur
def blur(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    kernel = np.ones((15,15),np.float32)/225
    dst = cv2.filter2D(img,-1,kernel)
    return dst

def blur_best(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    kernel = np.ones((7,7),np.float32)/25   
    dst = cv2.filter2D(img,-1,kernel)
    return dst


# averaging
def average(img_path):
    img = cv2.imread(img_path)
    blur = cv2.blur(img,(5,5))
    
    return blur

def median(img_path):
    img = cv2.imread(img_path)
    median = cv2.medianBlur(img,3)
    return median

def bilateral(img_path):
    img = cv2.imread(img_path)
    blur = cv2.bilateralFilter(img,31,75,75)
    return blur

def clip(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    MIN_BOUND = 0
    MAX_BOUND = 150
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    
    # image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1
    image[image < 0] = 0
    image = image*0.7
    image = (image*255).astype(np.uint8)
    
    # kernel = np.ones((15,15),np.float32)/225
    # dst = cv2.filter2D(img,-1,kernel)
    
    # blur = cv2.medianBlur(image,7)
    image = gaussian_filter(image, sigma=2)
    return image
        

# method = 'blur'
# method = 'blur'
method = 'gaussian'
# method = 'clip'


root = '/home/zhaoxiang/data_preparation/sample_img/raw'
save_dir = os.path.join('/home/zhaoxiang/data_preparation/sample_img', method)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)


for file in os.listdir(root):
    img_path = os.path.join(root, file)
    save_path = os.path.join(save_dir, file)
    
    # dst = median(img_path)
    dst = clip(img_path)
    # dst = bilateral(img_path)
    
    cv2.imwrite(save_path, dst)
    




