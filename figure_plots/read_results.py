import numpy as np
import pandas as pd


#%% read the results.txt


def read_results(results_path, list_len = 90):
    with open(results_path, 'r') as f:
        lines = f.readlines()
        
    # lines = lines.split('\n')
    losses = []
    for line in lines:
        if 'loss' in line:
            x = line.split('loss:')[-1]
            if 'learning_rate' in x:
                x = x.split(', learning_rate')[0]
            else:
                x = x.split(' \n')[0]
            x = float(x)
            
        losses.append(x)
        
        
        if len(losses) == list_len:
            break
    return losses

WS_results_path = '/home/zhaoxiang/DRAEM_Denosing/figure_plots/results/WS_results.txt'
WS_loss = read_results(WS_results_path)

DRAEM_results_path = '/home/zhaoxiang/DRAEM_Denosing/figure_plots/results/DRAEM_results.txt'
DRAEM_loss = read_results(DRAEM_results_path)
epochs = np.arange(1,91)

marker_DRAEM = DRAEM_loss[0::10]
marker_WS = WS_loss[0::10]
marker_epochs = epochs[0::10]


#%% plot the lines
import matplotlib.pyplot as plt


# plt.plot(epochs, WS_loss, 'bo')
# plt.plot(epochs, DRAEM_loss, 'go')
plt.figure(figsize=(8,6), dpi=80)


# line
plt.plot(epochs, WS_loss, linestyle='--', linewidth = 3, label='U-net', color='m')
plt.plot(epochs, DRAEM_loss, linestyle='--', linewidth = 3, label='Autoencoder', color='c')

# # marker
# plt.scatter(marker_epochs, marker_WS, marker='^', c='r')
# plt.scatter(marker_epochs, marker_DRAEM, marker='^', c = 'r')
# # marker
plt.scatter(marker_epochs, marker_WS, marker='o', c='m', s=60)
plt.scatter(marker_epochs, marker_DRAEM, marker='o', c = 'c', s=60)


plt.xlabel('epochs', fontsize=14)
plt.ylabel('loss', fontsize=14)

plt.legend(fontsize=14)

plt.ylim(0, 0.3)
plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])

plt.savefig('/home/zhaoxiang/DRAEM_Denosing/figure_plots/figures/DRAEMvsUnet')