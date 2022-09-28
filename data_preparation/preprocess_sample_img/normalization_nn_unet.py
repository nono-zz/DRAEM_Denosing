import os
import numpy as np
import cv2


# def nnunet(img, mean, std, min, max):
#     img = np.clip(img, min, max)
#     img = ((img - mean)/std + 1) * 122.5
#     return img

# def xiaoman(img):
#     img = np.clip(img, -17, 201)
#     img = (img-99.40)/39.36
#     return 


def min_max_norm(img):
    img = (img - img.min())/(img.max() - img.min())
    return img




# root = '/home/zhaoxiang/data_preparation/preprocess_sample_img/raw'
# train_root = '/home/zhaoxiang/dataset/initial_version/train/good'
# test_root = '/home/zhaoxiang/dataset/initial_version/test'
# mode = 'nnunet'
# # mode = 'xiaoman'
# new_dir = test_root.replace('test', mode)
# if not os.path.exists(new_dir):
#     os.mkdir(new_dir)

# # collect the mean and std
# data = []
# for file in os.listdir(train_root):
#     file_path = os.path.join(train_root, file)    
#     img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
#     # img = img.ravel
#     data.extend(img.ravel())
    
# for file in os.listdir(test_root):
#     file_path = os.path.join(test_root, file)    
#     img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
#     # img = img.ravel
#     data.extend(img.ravel())
    

# data = np.array(data)
# data = data[data!=0]
# mean = np.mean(data)
# std = np.std(data)

# # data = np.array(data)
# min_boundary = np.percentile(data, 0.5)
# max_boundary = np.percentile(data, 99.5)


# print('mean is {}, std is {}, min is {}, max is {}'.format(mean, std, min_boundary, max_boundary))

# mean = 144.331
# std = 30.4866
# min_boundary = 78.0
# max_boundary = 195.0


# mean is 136.42001041687985, std is 25.314454084740706, min is 70.0, max is 200.0




# # normalize the image by the mean and std collected
# for file in os.listdir(test_root):
#     file_path = os.path.join(test_root, file)    
#     img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    
#     img = np.clip(img, min_boundary, max_boundary)
#     img = (img - mean)/std
    
#     img = min_max_norm(img) * 255
    
    
           
#     # img = (xiaoman(img) + 1) * 122.5
    
#     new_path = os.path.join(new_dir, file)
#     cv2.imwrite(new_path, img)
    


def clip(img):
    x, y =  img.shape
    for i in range(x):
        for j in range(y):
            if img[i,j] != 0:
                if img[i,j] < 70:
                    img[i,j] = 70
                elif img[i,j] > 200:
                    img[i,j] = 200
    
            
    # img = np.clip(img, 113, 200)
    return img
    

# root = '/home/zhaoxiang/data_preparation/preprocess_sample_img/raw'
# train_root = '/home/zhaoxiang/dataset/initial_version/train/good'
test_root = '/home/zhaoxiang/dataset/initial_version/test'

new_dir = os.path.join(test_root.replace('initial_version', 'nnUnet_normalization'))
    
# normalize the image by the mean and std collected
for i, file in enumerate(os.listdir(test_root)):
    file_path = os.path.join(test_root, file)    
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    

    img = clip(img)
    img = min_max_norm(img) * 255
    
    
           
    # img = (xiaoman(img) + 1) * 122.5
    
    new_path = os.path.join(new_dir, file)
    cv2.imwrite(new_path, img)
    