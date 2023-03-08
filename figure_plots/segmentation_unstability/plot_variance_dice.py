import os
import numpy as np
import shutil
import matplotlib.pyplot as plt

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

data = []
for file_name in filenames:
    file_path = os.path.join(root_dir, file_name)
    
    
    if '6' in file_path or '4' in file_path:
        epochs, aucs, dices = read_results(file_path)
        epochs_1 = epochs[::2]
        aucs_1 = aucs[::2]
        dices_1 = dices[::2]
        
        data.append(dices_1)
        
        epochs_2 = epochs[1::2]
        aucs_2 = aucs[1::2]
        dices_2 = dices[1::2]
        
        data.append(dices_2)

        
        
    else:
        epochs, aucs, dices = read_results(file_path)
        data.append(dices)
        
        x_labels = epochs
        
x_labels[0] = 1    



data_numpy = np.array(data)
#%% compute the mean and variance
data_mean = np.mean(data_numpy, axis=0)
data_std = np.std(data_numpy, axis=0)


# plt.figure(figsize=(20,6), dpi=80)   
fig, ax1 = plt.subplots()
fig.set_figheight(6)
fig.set_figwidth(8)
ax2 = ax1.twinx()
mean_line = ax1.plot(x_labels, data_mean, marker='s', markersize = 2, linewidth=2, color= 'royalblue', label='mean')
std_line = ax2.plot(x_labels, data_std, marker='s', markersize = 2, linewidth=2, color = 'darkorange', label='std')

for line in data_numpy:
    ax1.plot(x_labels, line, linewidth=1, linestyle='--', alpha=.5)
    

ax1.set_xlabel('Epochs', fontsize=14)
ax1.set_ylabel('Mean', fontsize=14)
ax2.set_ylabel('Standard Deviation', fontsize=14)

x_ticks = [x for x in range(0, 200, 20)]
# x_ticks.insert(1, 1)
x_ticks[0] = 1
x_ticks.append(200)
# plt.xticks([x for x in range(0,200,20)])
# ax1.set_xticks(x_ticks)
plt.xticks(x_ticks)
# plt.ylim(0.25, 0.6)
# plt.ylim(0.7, 0.8)

ax1.set_ylim(0.30, 0.57)
ax2.set_ylim(0, 0.2)

# plt.legend('mean', 'std')
# ax1.legend(loc=0)
# ax2.legend(loc=0)
lns = mean_line + std_line
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc = 'center right' )

plt.xlim(0, 200)
plt.savefig('/home/zhaoxiang/DRAEM_Denosing/figure_plots/segmentation_unstability/unstability_dice_variance_assemble.png')
    
    
    
