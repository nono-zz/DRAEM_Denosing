import os
import cv2
import numpy as np


# img_dir = '/home/zhaoxiang/dataset/Atlas_train+LiTs_test/test'
# label_dir = '/home/zhaoxiang/dataset/Atlas_train+LiTs_test/test_label'
img_paths = []
root_dir = '/home/zhaoxiang/dataset/Atlas_train+LiTs_test'
for (dirpath, dirnames, filenames) in os.walk(root_dir):
        for filename in filenames:
            img_paths.append(os.path.join(dirpath, filename))

# imgs = os.listdir(label_dir)
for img_path in img_paths:
    # img_path = os.path.join(label_dir, img)
    frame = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    frame = cv2.flip(frame, 0)
    
    cv2.imwrite(img_path, frame)
    
