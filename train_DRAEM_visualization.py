from pickle import FALSE
from matplotlib.pyplot import gray
import torch
from torch.utils.data import DataLoader
from torch import optim
from tensorboard_visualizer import TensorboardVisualizer
from loss import FocalLoss, SSIM, DiceLoss, DiceBCELoss
import os
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
import numpy as np

import torch.nn.functional as F
import random

from dataloader_zzx import MVTecDataset, Medical_dataset, MVTecDataset_cross_validation
from evaluation_mood import evaluation, evaluation_DRAEM, evaluation_DRAEM_with_device, evaluation_DRAEM_half
from cutpaste import CutPaste3Way, CutPasteUnion

from model import ReconstructiveSubNetwork, DiscriminativeSubNetwork

from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def mean(list_x):
    return sum(list_x)/len(list_x)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
def get_data_transforms(size, isize):
    # mean_train = [0.485]         # how do you set the mean_train and std_train in the get_data_transforms function?
    # mean_train = [-0.1]
    # std_train = [0.229]
    data_transforms = transforms.Compose([
        # transforms.Resize((size, size)),
        # transforms.CenterCrop(isize),
        
        #transforms.CenterCrop(args.input_size),
        transforms.ToTensor()
        # transforms.Normalize(mean=mean_train,
        #                      std=std_train)
    ])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor()])

    return data_transforms, gt_transforms

        
        
def add_Gaussian_noise(x, noise_res, noise_std, img_size):
    ns = torch.normal(mean=torch.zeros(x.shape[0], x.shape[1], noise_res, noise_res), std=noise_std).to(x.device)

    ns = F.upsample_bilinear(ns, size=[img_size, img_size])

    # Roll to randomly translate the generated noise.
    roll_x = random.choice(range(128))
    roll_y = random.choice(range(128))
    ns = torch.roll(ns, shifts=[roll_x, roll_y], dims=[-2, -1])

    mask = x.sum(dim=1, keepdim=True) > 0.01
    ns *= mask # Only apply the noise in the foreground.
    res = x + ns
    
    return res
        

def train_on_device(args):

    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    # run_name = args.experiment_name + '_' +str(args.lr)+'_'+str(args.epochs)+'_bs'+str(args.bs)+"_"+"Guassian_blur"
    run_name = args.experiment_name + '_' +str(args.lr)+'_'+str(args.epochs)+'_colorRange'+'_'+str(args.colorRange)+'_threshold'+'_'+str(args.threshold)+"_" + args.model + "_" + args.process_method
    # run_name = args.experiment_name + '_' +str(args.lr)+'_'+str(700)+'_colorRange'+'_'+str(args.colorRange)+'_threshold'+'_'+str(args.threshold)+"_" + args.model + "_" + args.process_method

    main_path = '/home/zhaoxiang/dataset/{}'.format(args.dataset_name)
    
    data_transform, gt_transform = get_data_transforms(args.img_size, args.img_size)
    test_transform, _ = get_data_transforms(args.img_size, args.img_size)
    train_transform = transforms.Compose([])
    train_transform.transforms.append(CutPaste3Way(transform = test_transform))
    # test_transform, _ = get_data_transforms(args.img_size, args.img_size)

    dirs = os.listdir(main_path)
    
    for dir_name in dirs:
        if 'train' in dir_name:
            train_dir = dir_name
        elif 'test' in dir_name:
            if 'label' in dir_name:
                label_dir = dir_name
            else:
                test_dir = dir_name
    if 'label_dir' in locals():
        dirs = [train_dir, test_dir, label_dir]                


    from model_noise import UNet
    
    device = torch.device('cuda:{}'.format(args.gpu_id))
    n_input = 1
    n_classes = 1           # the target is the reconstructed image
    depth = 4
    wf = 6
    
    if args.model == 'ws_skip_connection':
        model = UNet(in_channels=n_input, n_classes=n_classes, norm="group", up_mode="upconv", depth=depth, wf=wf, padding=True).to(device)
    elif args.model == 'DRAEM_reconstruction':
        model = ReconstructiveSubNetwork(in_channels=n_input, out_channels=n_input).to(device)
    elif args.model == 'DRAEM_discriminitive':
        model = DiscriminativeSubNetwork(in_channels=n_input, out_channels=n_input).to(device)
    elif args.model == 'DRAEM':
        model_denoise = UNet(in_channels=n_input, n_classes=n_classes, norm="group", up_mode="upconv", depth=depth, wf=wf, padding=True).to(device)
        # model_denoise = DiscriminativeSubNetwork(in_channels=n_input, out_channels=n_input).to(device)
        model_segment = DiscriminativeSubNetwork(in_channels=2, out_channels=2, out_latent_features=True).to(device)
        
        model_denoise.to(device)
        model_segment.to(device)
        model_denoise = torch.nn.DataParallel(model_denoise, device_ids=[1])
        model_segment = torch.nn.DataParallel(model_segment, device_ids=[1])
        
        # model_denoise = torch.nn.DataParallel(model_denoise, device_ids=[0, 1])
        # model_segment = torch.nn.DataParallel(model_segment, device_ids=[0, 1])
        base_path= '/home/zhaoxiang'
        output_path = os.path.join(base_path, 'output')

        experiment_path = os.path.join(output_path, run_name)
        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path, exist_ok=True)
        ckp_path = os.path.join(experiment_path, 'last.pth')
        result_path = os.path.join(experiment_path, 'results.txt')
        
        # load the pretrained 
        # model_denoise.load_state_dict(torch.load(os.path.join(experiment_path, 'reconstruction_last.pth'))['model'])
        model_denoise.load_state_dict(torch.load(os.path.join(experiment_path, 'experiement_3_epoch_200.pth'))['model'])
        
        
    last_epoch = 0
    if args.resume_training:
        # model_denoise.load_state_dict(torch.load(ckp_path)['model_denoise'])
        model_segment.load_state_dict(torch.load(ckp_path)['model'])
        last_epoch = torch.load(ckp_path)['epoch']
        
    train_data = MVTecDataset(root=main_path, transform = test_transform, gt_transform=gt_transform, phase='train', dirs = dirs, data_source=args.experiment_name, args = args)
    # test_data = MVTecDataset_cross_validation(root='/home/zhaoxiang/dataset/LiTs_with_labels', transform = test_transform, gt_transform=gt_transform, phase='test', data_source=args.experiment_name, args = args)
    # test_data = MVTecDataset(root=main_path, transform = test_transform, gt_transform=gt_transform, phase='test', dirs = dirs, data_source=args.experiment_name, args = args)
    test_data = MVTecDataset_cross_validation(root='/home/zhaoxiang/dataset/LiTs_with_labels', transform = test_transform, gt_transform=gt_transform, phase='test', data_source=args.experiment_name, args = args)
    # train_data = MVTecDataset(root=main_path, transform = test_transform, gt_transform=gt_transform, phase='train', dirs = dirs, data_source=args.experiment_name, args = args)
    
        
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size = args.bs, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size = 1, shuffle = True)
        
    model_denoise.eval()
    model_segment.eval()
    
    data = []
    target = []
    
    i = 0
    iteration = 0
    
    with torch.no_grad():
        for img, gt, label, img_path, save in tqdm(test_dataloader):
            if iteration == 80:
                break
            iteration += 1
            
            img = img.to(device)
            gt[gt > 0.1] = 1
            gt[gt <= 0.1] = 0
            # check if img is RGB
            with torch.no_grad():
                rec = model_denoise(img)
                rec = rec.detach().cpu()
            rec = rec.to(device)

            features = torch.cat((rec, img), dim=1)
            content = features.detach().cpu().numpy()
            
            
            data.append(content[0].ravel())
            target.append(torch.max(gt[0]).detach().cpu().numpy() + 2)
    
    

    for img, aug, anomaly_mask in tqdm(train_dataloader):
        img = torch.reshape(img, (-1, 1, args.img_size, args.img_size))
        aug = torch.reshape(aug, (-1, 1, args.img_size, args.img_size))
        anomaly_mask = torch.reshape(anomaly_mask, (-1, 1, args.img_size, args.img_size))
        
        img = img.to(device)
        aug = aug.to(device)
        anomaly_mask = anomaly_mask.to(device)

        with torch.no_grad():
            rec = model_denoise(aug)
            rec = rec.detach().cpu()
        rec = rec.to(device)
        
        features = torch.cat((rec, aug), dim=1)
        content = features.detach().cpu().numpy()
        
        for j in range(args.bs):
            data.append(content[j].ravel())
            target.append(torch.max(anomaly_mask[j]).detach().cpu().numpy())
                
        i += 1
        
        if i == 80:
            break
        
        
    target = [0 if x == 2 else x for x in target]
    target = [2 if x == 3 else x for x in target]
    
    # remove 0
        
    data = np.array(data)
    target = np.array(target)

    data = TSNE(n_components=2, perplexity=15, learning_rate=10, verbose=2).fit_transform(data)
    
    df = pd.DataFrame({"x":data[:, 0], "y":data[:, 1], "hue":target})

    sns.scatterplot(x="x", y="y", data=df, hue="hue", legend="full")
    plt.savefig("mnist_conv_autoencoder_data_visualization.png")
        

if __name__=="__main__":
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_id', default=1,  action='store', type=int)
    parser.add_argument('--lr', default=0.0001, action='store', type=float)
    parser.add_argument('--epochs', default=200, action='store', type=int)
    parser.add_argument('--checkpoint_path', default='./checkpoints/', action='store', type=str)
    parser.add_argument('--log_path', default='./logs/', action='store', type=str)
    parser.add_argument('--visualize', default=True, action='store_true')

    parser.add_argument('--backbone', default='noise', action='store')
    
    # for noise autoencoder
    parser.add_argument("-nr", "--noise_res", type=float, default=16,  help="noise resolution.")
    parser.add_argument("-ns", "--noise_std", type=float, default=0.2, help="noise magnitude.")
    parser.add_argument("-img_size", "--img_size", type=float, default=256, help="noise magnitude.")
    
    # need to be changed/checked every time
    parser.add_argument('--bs', default = 1, action='store', type=int)
    # parser.add_argument('--gpu_id', default=['0','1'], action='store', type=str, required=False)
    parser.add_argument('--gpu_id', default='1', action='store', type=str, required=False)
    parser.add_argument('--experiment_name', default='Segmentation_unstable_exp_4', choices=['DRAEM_Denoising_reconstruction, liver, brain, head'], action='store')
    parser.add_argument('--colorRange', default=100, action='store')
    parser.add_argument('--threshold', default=200, action='store')
    parser.add_argument('--dataset_name', default='hist_DIY', choices=['hist_DIY', 'Brain_MRI', 'CovidX', 'RESC_average'], action='store')
    parser.add_argument('--model', default='DRAEM', choices=['ws_skip_connectiosn', 'DRAEM_reconstruction', 'DRAEM_discriminitive'], action='store')
    parser.add_argument('--process_method', default='Gaussian_noise', choices=['none', 'Guassian_noise', 'DRAEM', 'Simplex_noise'], action='store')
    parser.add_argument('--multi_layer', default=False, action='store')
    parser.add_argument('--rejection', default=False, action='store')
    parser.add_argument('--number_iterations', default=1, action='store')
    parser.add_argument('--control_texture', default=False, action='store')
    parser.add_argument('--cutout', default=False, action='store')
    parser.add_argument('--resume_training', default=True, action='store')
    
    args = parser.parse_args()
   
    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    # if args.gpu_id is None:
    #     gpus = "0"
    #     os.environ["CUDA_VISIBLE_DEVICES"]= gpus
    # else:
    #     gpus = ""
    #     for i in range(len(args.gpu_id)):
    #         gpus = gpus + args.gpu_id[i] + ","
    #     os.environ["CUDA_VISIBLE_DEVICES"]= gpus[:-1]

    # torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

    # with torch.cuda.device(args.gpu_id):
    train_on_device(args)
