import os
import cv2
import numpy as np

# dir_path = '/home/zhaoxiang/dataset/hist_DIY_2/train/good'
# des_path = '/home/zhaoxiang/dataset/hist_DIY_2_crop/train/good'

dir_path = '/home/zhaoxiang/dataset/hist_DIY_2/test'
des_path = '/home/zhaoxiang/dataset/hist_DIY_2_crop/test'

files = os.listdir(dir_path)
files.sort()
for file in files:
    new_name = file.replace('.png', '_crop.png')
    new_path = os.path.join(des_path, new_name)
    file_path = os.path.join(dir_path, file)
    
    # crop out zero region
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    x, y = np.nonzero(img)
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    img_crop = img[x_min:x_max+1, y_min:y_max+1]
    
    cv2.imwrite(new_path, img_crop)
    
print('done')