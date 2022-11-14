import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import cv2

def get_img_labels(files):
    label_list = []
    root_dir = '/home/zhaoxiang/dataset/LiTs_with_labels/'
    for img_name in files:
        gt_name = img_name.replace('liver', 'liver_gt')
        gt_path = os.path.join(root_dir, 'image_labels', gt_name)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt.sum() > 0:
            label = 1
        else:
            label = 0
            
        label_list.append(label)
        
    return label_list
        
    
        





root_dir = '/home/zhaoxiang/dataset/LiTs_with_labels/image'
files = os.listdir(root_dir)

files.sort()

files_np = np.array(files)

# X_train, Y_train = train_test_split(files, test_size = 0.2)

# print(Y_train.shape)

kf = KFold(n_splits=5, shuffle=True)
kf.get_n_splits(files)

# train_labels = []
# test_labels = []
for i, (train_index, test_index) in enumerate(kf.split(files)):
    print('TRAIN:', train_index, 'TEST:', test_index)
    X_train, X_test = files_np[train_index], files_np[test_index]
    
    train_labels = get_img_labels(X_train)
    test_labels = get_img_labels(X_test)
    
    
    data = {'Train_images': X_train,
            'Train_labels': train_labels,
            'Test_images': X_test,
            'Test_labels': test_labels}
    
    with open('/home/zhaoxiang/dataset/LiTs_with_labels/fold_{}.npy'.format(i), 'wb') as f:
        np.save(f, data)
    

for i in range(5):
    file_path = '/home/zhaoxiang/dataset/LiTs_with_labels/fold_{}.npy'.format(i)
    data = np.load(file_path, allow_pickle=True)
    
    train_files = data.item()['Train']
    test_files = data.item()['Test']
    
    
