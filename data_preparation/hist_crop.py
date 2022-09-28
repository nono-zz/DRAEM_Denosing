import os
import cv2
import nibabel as nib
import numpy as np


# img = nib.load(file_path)          # read nii
# img_fdata = img.get_fdata()
# (x,y,z) = img_fdata.shape


img = cv2.imread('/home/zhaoxiang/dataset/abdnomial_images/data_lesion/liver_1/raw/57_raw.png', cv2.IMREAD_GRAYSCALE)
hist = cv2.equalizeHist(img)

cv2.imwrite('hist_raw.png', hist)


gt = cv2.imread('/home/zhaoxiang/dataset/abdnomial_images/data_lesion/liver_1/gt/57_gt.png', cv2.IMREAD_GRAYSCALE)
mask = np.where(gt>0, 1, 0)
liver_img = np.multiply(mask, hist)

cv2.imwrite('liver_hist.png', liver_img)

# hist_img = histequal(img_fdata)
hist_img = img_fdata
for i in range(z):
    gray = hist_img[:,:,i]
    matplotlib.image.imsave(os.path.join(save_path, '{}_'.format(i) + 'raw' + '.png'), gray, cmap='gray')
print(save_path, '    done')
