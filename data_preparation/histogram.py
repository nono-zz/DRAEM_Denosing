import numpy as np
import os
import cv2
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image


def histequal(flair_img=None):
    N,edges = np.histogram(flair_img,100)
    minimum = edges[np.where(edges>np.percentile(flair_img[flair_img!=0],2))[0][0]]
    diffN = np.zeros(N.shape)
    for i in range(1,N.shape[0]):
        if N[i-1]!=0:
            diffN[i] = N[i]/N[i-1]
    s = np.where(edges >= np.percentile(flair_img,50))[0][0]
    if np.where(diffN[s:]>1.0)[0].shape[0]<5:
        return flair_img
    f = np.where(diffN[s:]>1.0)[0][4]
    start=s+f
    ind = np.argmax(N[start:])
    peak_val = edges[ind + start-1]
    maximum = minimum + ((peak_val - minimum) * 2.55)
    flair_img[flair_img<minimum] = minimum
    flair_img[flair_img>maximum] = maximum
    flair_img = (flair_img-minimum)/(maximum-minimum)
    if maximum == minimum:
        print('found')
    return flair_img


# root = '/home/zhaoxiang/dataset/initial_version/good'
# files = os.listdir(root)
# files.sort()
# count = 0
# for file in files:
    
#     # file_path = os.path.join(root, file)
#     file_path = '/home/zhaoxiang/dataset/initial_version/test/liver_1_63_segmentation.png'
#     img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
#     count += 1

    
#     img_hist = histequal(img)
    
#     cv2.imwrite('hist.png', img_hist * 255)    
#     if count > 0:
#         break



# file_path = '/home/zhaoxiang/liver_3.nii.gz'
file_path = '/home/zhaoxiang/66_raw.png'

# img = nib.load(file_path)          # read nii
# img_fdata = img.get_fdata()
# (x,y,z) = img_fdata.shape

# i= 350
# gray = img_fdata[:,:,i]

gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

# gray_norm = (gray - gray.min())/(gray.max() - gray.min()) * 255
# gray_norm = gray_norm.astype(np.uint8)

median = cv2.medianBlur(gray,5)
# hist = histequal(gray)
hist = cv2.equalizeHist(median)

# matplotlib.image.imsave('/home/zhaoxiang/2D_cut.png', hist, cmap='gray')

# hist_median = cv2.medianBlur(hist,3)
cv2.imwrite('/home/zhaoxiang/2D_cut_median_5.png', hist)






# img_h = histequal(img_fdata)
# print('hey')

# raw_path = '/home/zhaoxiang/dataset/3D_hist'

# # img_h = img_fdata
        
# (x,y,z) = img_h.shape
# for i in range(z):
#     gray = img_h[:,:,i]
#     matplotlib.image.imsave(os.path.join(raw_path, '{}_'.format(i) + '.png'), gray, cmap='gray')
# print(fname, '   done')
