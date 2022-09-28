import os
import shutil

# extract images from their folders, and change names

root = '/home/zhaoxiang/dataset/full_hist'
dataset_path = '/home/zhaoxiang/dataset/abdnomial_images'

version = 'hist_segmentation_flip'

# phase = 'train'
phase = 'test'

destination_dir = os.path.join(root, phase)
if not os.path.exists(destination_dir):
    os.mkdir(destination_dir)
    
# access the images
if phase == 'train':
    folder = os.path.join(dataset_path, 'data_normal')
    subjects = os.listdir(folder)
    subjects.sort()
    for subject in subjects:
        subject_path = os.path.join(folder, subject, version)
        files = os.listdir(subject_path)
        files.sort()
        # images for each subject
        for file in files:
            file_path = os.path.join(subject_path, file)
            old_path = file_path
            new_name = subject + '_' + file
            new_path = os.path.join(destination_dir, new_name)
            shutil.copy(old_path, new_path)
            print(new_path, '   done')
            
            
elif phase == 'test':
    # 先把处理好的图片挪到liver里面
    numbers = [i for i in range(1,11)]
    names = ['liver_{}'.format(i) for i in numbers]
    for name in names:
        folder_path = os.path.join(dataset_path, 'data_lesion', name, 'hist_segmentation_flip')
        new_folder_path = os.path.join(dataset_path, 'liver1-10', name, 'hist_segmentation_flip')
        
        # move the folder to liver1_10
        if not os.path.exists(new_folder_path):
            shutil.copytree(folder_path, new_folder_path)
    
    
    
    
    folder = os.path.join(dataset_path, 'liver1-10')
    subjects = os.listdir(folder)
    subjects.sort()
    for subject in subjects:
        subject_img_path = os.path.join(folder, subject, 'hist_segmentation_flip')
        subject_label_path = os.path.join(folder, subject, 'label')
        imgs = os.listdir(subject_img_path)
        # labels = os.listdir(subject_label_path)
        imgs.sort()
        for img in imgs:
            img_path = os.path.join(subject_img_path, img)
            number = img.split('_')[0]
            label = number + '_label.png'
            label_path = os.path.join(subject_label_path, label)
            
            new_img_name = subject + '_' + img
            new_label_name = subject + '_' + label
            
            new_img_path = os.path.join(destination_dir, new_img_name)
            new_label_path = os.path.join(os.path.join(root, 'test_label'), new_label_name)
            
            shutil.copy(img_path, new_img_path)
            shutil.copy(label_path, new_label_path)
            
            print(new_label_path,  '   done')
        
        
