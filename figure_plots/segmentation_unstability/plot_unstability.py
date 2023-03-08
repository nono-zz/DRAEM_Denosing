import os
import numpy as np
import shutil
import matplotlib.pyplot as plt

#%% first copy all files to results

# root_dir = '/home/zhaoxiang/output'
# i = 0
# for dir_name in os.listdir(root_dir):
#     if dir_name.startswith('Segmentation_un'):
#         i += 1
#         src_path = os.path.join(root_dir, dir_name, 'results.txt')
#         dst_path = os.path.join('/home/zhaoxiang/DRAEM_Denosing/figure_plots/segmentation_unstability/results', 'results_{}.txt'.format(i))
#         shutil.copy(src_path, dst_path)


#%% read all the data

def retrive_number(string):
    return ''.join([n for n in string if n.isdigit()])

def read_results(file_path):
    epochs, aucs, dices = [], [], []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('Epoch'):
                data = line.split(',')
                # epochs.append(int(retrive_number(data[0])))
                # aucs.append(float(retrive_number(data[1])))
                # dices.append(float(retrive_number(data[2])))
                epochs.append(int(data[0].replace('Epoch:', '')))
                aucs.append(float(data[1].replace(' Sample Auroc', '')))
                dices.append(float(data[2][5:10]))
    epochs[0] = 1
    return epochs, aucs, dices
                

plt.figure(figsize=(8,6), dpi=80)                

root_dir = '/home/zhaoxiang/DRAEM_Denosing/figure_plots/segmentation_unstability/results'
filenames = os.listdir(root_dir)
filenames.sort()
for file_name in filenames:
    file_path = os.path.join(root_dir, file_name)
    
    
    if '6' in file_path:
        epochs, aucs, dices = read_results(file_path)
        epochs_1 = epochs[::2]
        aucs_1 = aucs[::2]
        dices_1 = dices[::2]
        
        # plt.plot(epochs_1, dices_1)
        plt.plot(epochs_1, aucs_1)
        
        epochs_2 = epochs[1::2]
        aucs_2 = aucs[1::2]
        dices_2 = dices[1::2]
        
        # plt.plot(epochs_2, dices_2)
        plt.plot(epochs_2, aucs_2)
        
        
    else:
        epochs, aucs, dices = read_results(file_path)
        # plt.plot(epochs, dices, marker='s', markersize = 2, linewidth=1)
        plt.plot(epochs, aucs, marker='s', markersize = 2, linewidth=1)
    # plt.plot(epochs, dices)


plt.xlabel('Epochs', fontsize=14)
plt.ylabel('DICE', fontsize=14)

x_ticks = [x for x in range(0, 200, 20)]
# x_ticks.insert(1, 1)
x_ticks[0] = 1
# plt.xticks([x for x in range(0,200,20)])
plt.xticks(x_ticks)
# plt.ylim(0.25, 0.6)
plt.ylim(0.6, 0.82)
plt.xlim(0, 200)
plt.savefig('/home/zhaoxiang/DRAEM_Denosing/figure_plots/segmentation_unstability/unstability_auc.png')
    
    
    
