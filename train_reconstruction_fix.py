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
from tqdm import tqdm

import torch.nn.functional as F
import random

from dataloader_zzx import MVTecDataset, MVTecDataset_fixed
from evaluation_mood import evaluation, evaluation_DRAEM, evaluation_reconstruction
from cutpaste import CutPaste3Way, CutPasteUnion

from model import ReconstructiveSubNetwork, DiscriminativeSubNetwork

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
    run_name = args.experiment_name + '_' + args.dataset_name + '_' +str(args.lr)+'_'+str(args.epochs)+'_colorRange'+'_'+str(args.colorRange)+'_threshold'+'_'+str(args.threshold)+"_" + args.model + "_" + args.process_method

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
    
    # device = torch.device('cuda:{}'.format(args.gpu_id))
    n_input = 1
    n_classes = 1           # the target is the reconstructed image
    depth = 4
    wf = 6
    
    if args.model == 'ws_skip_connection':
        model = UNet(in_channels=n_input, n_classes=n_classes, norm="group", up_mode="upconv", depth=depth, wf=wf, padding=True).cuda()
    elif args.model == 'DRAEM_reconstruction':
        model = ReconstructiveSubNetwork(in_channels=n_input, out_channels=n_input).cuda()
    elif args.model == 'DRAEM_discriminitive':
        model = DiscriminativeSubNetwork(in_channels=n_input, out_channels=n_input).cuda()
        
    base_path= '/home/zhaoxiang'
    output_path = os.path.join(base_path, 'output')

    experiment_path = os.path.join(output_path, run_name)
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path, exist_ok=True)
    ckp_path = os.path.join(experiment_path, 'last.pth')
    
    model = torch.nn.DataParallel(model, device_ids=[0, 1])

    result_path = os.path.join(experiment_path, 'results.txt')
        
    last_epoch = 0
    if args.resume_training:
        model.load_state_dict(torch.load(ckp_path)['model'])
        last_epoch = torch.load(ckp_path)['epoch']
        
    train_data = MVTecDataset_fixed(root=main_path, transform = test_transform, gt_transform=gt_transform, phase='train', data_source=args.experiment_name, args = args)
    val_data = MVTecDataset_fixed(root=main_path, transform = test_transform, gt_transform=gt_transform, phase='test', data_source=args.experiment_name, args = args)
    # test_data = MVTecDataset_fixed(root=main_path, transform = test_transform, gt_transform=gt_transform, phase='test', data_source=args.experiment_name, args = args)
    test_data = MVTecDataset_fixed(root='/home/zhaoxiang/dataset/LiTs_with_labels', transform = test_transform, gt_transform=gt_transform, phase='eval', data_source=args.experiment_name, args = args)
        
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size = args.bs, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size = args.bs, shuffle = False)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size = 1, shuffle = False)
        
    loss_l1 = torch.nn.L1Loss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    
    loss_l2 = torch.nn.modules.loss.MSELoss()
    loss_ssim = SSIM()
    loss_focal = FocalLoss()
    loss_dice = DiceLoss()
    loss_diceBCE = DiceBCELoss()
    
    for epoch in range(last_epoch, args.epochs):
        # evaluation(args, model_denoise, model_segment, test_dataloader, epoch, loss_l1, visualizer, run_name)
        # evaluation_reconstruction(args, model, test_dataloader, epoch, loss_l1, run_name)
        
        model.train()
        loss_list = []
        # dice_value, auroc_px, auroc_sp = evaluation_reconstruction(args, model, test_dataloader, epoch, loss_l1, run_name)
        for img, aug, anomaly_mask in tqdm(train_dataloader):
            img = torch.reshape(img, (-1, 1, args.img_size, args.img_size))
            aug = torch.reshape(aug, (-1, 1, args.img_size, args.img_size))
            anomaly_mask = torch.reshape(anomaly_mask, (-1, 1, args.img_size, args.img_size))
            
            img = img.cuda()
            aug = aug.cuda()
            anomaly_mask = anomaly_mask.cuda()

            rec = model(aug)
            
            l1_loss = loss_l1(rec,img)
            
            loss = l1_loss
            
            save_image(aug, 'aug.png')
            save_image(rec, 'rec_output.png')
            save_image(img, 'rec_target.png')
            save_image(anomaly_mask, 'mask_target.png')
            # loss = loss_l1(img, output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            loss_list.append(loss.item())
            
        print('epoch [{}/{}], loss:{:.4f}'.format(args.epochs, epoch, mean(loss_list)))
        
        with open(result_path, 'a') as f:
            f.writelines('epoch [{}/{}], loss:{:.4f}'.format(args.epochs, epoch, mean(loss_list)))   
                
        if (epoch) % 10 == 0:
            model.eval()
            
            dice_value, auroc_sp = evaluation_reconstruction(args, model, test_dataloader, epoch, loss_l1, run_name)
            result_path = os.path.join('/home/zhaoxiang/output', run_name, 'results.txt')
            print('Sample Auroc{:.3f}, Dice{:3f}'.format(auroc_sp, dice_value))
            
            with open(result_path, 'a') as f:
                # f.writelines('Epoch:{}, Pixel Auroc:{:.3f}, Sample Auroc{:.3f}, Dice:{:3f} \n'.format(epoch, auroc_px, auroc_sp, dice_value))   
                f.writelines('Epoch:{}, Sample Auroc{:.3f}, Dice:{:3f} \n'.format(epoch, auroc_sp, dice_value))   
            
            torch.save({'model': model.state_dict(),
                        'epoch': epoch}, ckp_path)
        
        

if __name__=="__main__":
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_id', default=1,  action='store', type=int)
    parser.add_argument('--lr', default=0.0001, action='store', type=float)
    parser.add_argument('--epochs', default=201, action='store', type=int)
    parser.add_argument('--checkpoint_path', default='./checkpoints/', action='store', type=str)
    parser.add_argument('--log_path', default='./logs/', action='store', type=str)
    parser.add_argument('--visualize', default=True, action='store_true')

    parser.add_argument('--backbone', default='noise', action='store')
    
    # for noise autoencoder
    parser.add_argument("-nr", "--noise_res", type=float, default=16,  help="noise resolution.")
    parser.add_argument("-ns", "--noise_std", type=float, default=0.2, help="noise magnitude.")
    parser.add_argument("-img_size", "--img_size", type=float, default=256, help="noise magnitude.")
    
    # need to be changed/checked every time
    parser.add_argument('--bs', default = 8, action='store', type=int)
    parser.add_argument('--gpu_id', default=['0','1'], action='store', type=str, required=False)
    parser.add_argument('--experiment_name', default='ColorJitter_reconstruction_woRejection_fix_1_exp_1', choices=['DRAEM_Denoising_reconstruction, RandomShape_reconstruction, brain, head'], action='store')
    parser.add_argument('--colorRange', default=100, action='store')
    parser.add_argument('--threshold', default=200, action='store')
    parser.add_argument('--dataset_name', default='hist_DIY_pseudo', choices=['hist_DIY', 'Brain_MRI', 'CovidX', 'RESC_average'], action='store')
    parser.add_argument('--model', default='ws_skip_connection', choices=['ws_skip_connection', 'DRAEM_reconstruction', 'DRAEM_discriminitive'], action='store')
    parser.add_argument('--process_method', default='ColorJitter', choices=['none', 'Guassian_noise', 'DRAEM', 'Simplex_noise'], action='store')
    parser.add_argument('--multi_layer', default=False, action='store')
    parser.add_argument('--rejection', default=False, action='store')
    parser.add_argument('--number_iterations', default=1, action='store')
    parser.add_argument('--control_texture', default=False, action='store')
    parser.add_argument('--cutout', default=False, action='store')
    parser.add_argument('--resume_training', default=False, action='store')
    
    args = parser.parse_args()
   
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    if args.gpu_id is None:
        gpus = "0"
        os.environ["CUDA_VISIBLE_DEVICES"]= gpus
    else:
        gpus = ""
        for i in range(len(args.gpu_id)):
            gpus = gpus + args.gpu_id[i] + ","
        os.environ["CUDA_VISIBLE_DEVICES"]= gpus[:-1]

    torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

    # with torch.cuda.device(args.gpu_id):
    train_on_device(args)

