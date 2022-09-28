import csv
import os
import shutil

rows = []
previous_number  = 0

root_path = 'dataset/head_CT/imgs'

img_names = os.listdir(root_path)
img_names.sort()
# print(img_paths)
for img_name in img_names:
    if not img_name.endswith('.png'):
        continue
    number = img_name.replace('.png', '')
    number = int(number)
    if number == 0:
        continue
    else:
        assert number == previous_number+1
        previous_number = number

print('continuous test pass')



train_path = os.path.join(root_path, 'train')
test_path = os.path.join(root_path, 'test')

if not os.path.exists(train_path):
    os.mkdir(train_path)
    
if not os.path.exists(test_path):
    os.mkdir(test_path)
    
    
# read csv add divide img into 'no' and 'yes' folder
def recognize(root_path, train_path, test_path):
    with open('/home/zhaoxiang/dataset/head_CT/labels.csv') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == 'id':
                continue
            
            
            idx = int(row[0])
            label = row[1]
            img_name = img_names[idx]
            img_path = os.path.join(root_path, img_name)
            
            if label == '0':
                new_path = os.path.join(train_path, img_name)
            else:   
                new_path = os.path.join(test_path, img_name) 

            shutil.copy(img_path, new_path)
            print("File {} copied successfully to {}\n".format(img_name, new_path))

# recognize(root_path, train_path, test_path)
        
        
# add label on image name
no_path = os.path.join(root_path, 'no')
yes_path = os.path.join(root_path, 'yes')

def add_suffix(folder_path):
    folder_name = folder_path.split('/')[-1]
    if 'no' in folder_name:
        label = 'N'
    elif 'yes' in folder_name:
        label = 'Y'
    
    sample_names = os.listdir(folder_path)
    for name in sample_names:
        
        new_name = label + name
        old_path = os.path.join(folder_path, name )
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)

# add_suffix(no_path)
add_suffix(yes_path)

