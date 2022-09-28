import os

count = 0
# cound test set
# root_dir = '/home/zhaoxiang/dataset/data_lesion'

# count training set
root_dir = '/home/zhaoxiang/dataset/hist_DIY_crop_blur/test'

subjects = os.listdir(root_dir)

print(len(subjects))

# for subject in subjects:
#     image_path = os.path.join(root_dir, subject, 'segmentation')
#     images = os.listdir(image_path)
#     n = len(images)
#     count += n
    
# print(' the total number is:   ', count)


# the total number for
# test set: 19810
# training set: 1542
# test: 1659 是MSD的前10个project

# 训练集数据太少了


# count test 集里多少