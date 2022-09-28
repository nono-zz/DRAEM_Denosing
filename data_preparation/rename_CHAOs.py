import os
import shutil

dst_folder = '/home/zhaoxiang/datasets_raw/CHAOs/train/good'
root = '/home/zhaoxiang/datasets_raw/CHAOs/Train_segment'

prefix = 'liver_'

folders = os.listdir(root)
folders.sort()
for i, folder in enumerate(folders):
    files = os.listdir(os.path.join(root, folder))
    files.sort()
    for file in files:
        file_path = os.path.join(root, folder, file)
        
        if file.startswith('IMG'):
            number = file[-7:-4]
        else:
            number = file[2:5]
        new_name = prefix + str(i) + '_' + number + '.png'
        
        new_path = os.path.join(dst_folder, new_name)
        
        shutil.copy(file_path, new_path)
        