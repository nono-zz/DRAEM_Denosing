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

import wandb


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

        
        
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
    
    device = torch.device('cuda:{}'.format(0))
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
    elif args.model == 'DRAEM':
        model_denoise = UNet(in_channels=n_input, n_classes=n_classes, norm="group", up_mode="upconv", depth=depth, wf=wf, padding=True).cuda()
        model_segment = DiscriminativeSubNetwork(in_channels=2, out_channels=2).cuda()
        
        model_denoise.cuda()
        model_segment.cuda()
        # model_denoise = torch.nn.DataParallel(model_denoise, device_ids=[0])
        # model_segment = torch.nn.DataParallel(model_segment, device_ids=[0])
        
        model_denoise = torch.nn.DataParallel(model_denoise, device_ids=[0, 1])
        model_segment = torch.nn.DataParallel(model_segment, device_ids=[0, 1])
        base_path= '/home/zhaoxiang'
        output_path = os.path.join(base_path, 'output')

        experiment_path = os.path.join(output_path, run_name)
        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path, exist_ok=True)
        ckp_path = os.path.join(experiment_path, 'last.pth')
        # ckp_path = os.path.join(experiment_path, 'best.pth')
        # ckp_path = os.path.join(experiment_path, 'last_unshuffle.pth')
        # ckp_path = os.path.join(experiment_path, 'best_0.859_0.43_Dice_370_epoch.pth')
        
        
        # model_denoise = torch.nn.DataParallel(model_denoise, device_ids=[0, 1])
        # model_segment = torch.nn.DataParallel(model_segment, device_ids=[0, 1])
        # model_denoise = torch.nn.DataParallel(model_denoise, device_ids=[1])
        # model_segment = torch.nn.DataParallel(model_segment, device_ids=[1])

        result_path = os.path.join(experiment_path, 'results.txt')
        
    last_epoch = 0
    if args.resume_training:
        model_denoise.load_state_dict(torch.load(ckp_path)['model_denoise'])
        model_segment.load_state_dict(torch.load(ckp_path)['model'])
        last_epoch = torch.load(ckp_path)['epoch']
        
    train_data = MVTecDataset(root=main_path, transform = test_transform, gt_transform=gt_transform, phase='train', dirs = dirs, data_source=args.experiment_name, args = args)
    val_data = MVTecDataset(root=main_path, transform = test_transform, gt_transform=gt_transform, phase='test', dirs = dirs, data_source=args.experiment_name, args = args)
    test_data = MVTecDataset_cross_validation(root='/home/zhaoxiang/dataset/LiTs_with_labels', transform = test_transform, gt_transform=gt_transform, phase='test', data_source=args.experiment_name, args = args)
    # test_data = MVTecDataset(root=main_path, transform = test_transform, gt_transform=gt_transform, phase='test', dirs = dirs, data_source=args.experiment_name, args = args)
    # test_data = MVTecDataset_cross_validation(root='/home/zhaoxiang/dataset/LiTs_with_labels', transform = test_transform, gt_transform=gt_transform, phase='test', data_source=args.experiment_name, args = args)
        
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size = args.bs, shuffle=True)
    # val_dataloader = torch.utils.data.DataLoader(val_data, batch_size = args.bs, shuffle = False)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size = 1, shuffle = False)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size = 1, shuffle = False)
        
    loss_l1 = torch.nn.L1Loss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.Adam(list(model_segment.parameters()) + list(model_denoise.parameters()), lr = args.lr)
    # optimizer = torch.optim.SGD(list(model_segment.parameters()) + list(model_denoise.parameters()), lr = args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=50, gamma=0.1)
    
    loss_l2 = torch.nn.modules.loss.MSELoss()
    loss_ssim = SSIM(device=device)
    loss_focal = FocalLoss()
    loss_dice = DiceLoss()
    loss_diceBCE = DiceBCELoss()
    
    
    best_SP = 0
    best_dice = 0
    auroc_sp, dice_value = 0, 0
    
    for epoch in range(last_epoch, args.epochs):
        model_segment.train()
        model_denoise.train()
        loss_list = []
        
        # auroc_sp, dice_value = evaluation_DRAEM_half(args, model_denoise, model_segment, test_dataloader, epoch, loss_l1, run_name, device)
        
        
        for img, aug, anomaly_mask in tqdm(train_dataloader):
            img = torch.reshape(img, (-1, 1, args.img_size, args.img_size))
            aug = torch.reshape(aug, (-1, 1, args.img_size, args.img_size))
            anomaly_mask = torch.reshape(anomaly_mask, (-1, 1, args.img_size, args.img_size))
            
            img = img.cuda()
            aug = aug.cuda()
            anomaly_mask = anomaly_mask.cuda()

            rec = model_denoise(aug)
            joined_in = torch.cat((rec, aug), dim=1)
            
            out_mask = model_segment(joined_in)
            out_mask_sm = torch.softmax(out_mask, dim=1)

            l2_loss = loss_l2(rec,img)
            ssim_loss = loss_ssim(rec, img)
            
            if anomaly_mask.max() != 0:
                segment_loss = loss_focal(out_mask_sm, anomaly_mask)
                loss = segment_loss + l2_loss + ssim_loss
            else:
                loss = l2_loss + ssim_loss
            
            save_image(aug, 'aug.png')
            save_image(rec, 'rec_output.png')
            save_image(img, 'rec_target.png')
            save_image(anomaly_mask, 'mask_target.png')
            save_image(out_mask_sm[:,1:,:,:], 'mask_output.png')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            

        scheduler.step()
        current_learning_rate = get_lr(optimizer)
        print("current learning rate is:   ", current_learning_rate)
        
        print('epoch [{}/{}], loss:{:.6f} \n'.format(args.epochs, epoch, mean(loss_list)))
        with open(result_path, 'a') as f:
                f.writelines('epoch [{}/{}], loss:{:.4f}, learning_rate:{:.6f}, \n'.format(args.epochs, epoch, mean(loss_list), current_learning_rate))

        
        if (epoch+1) % 10 == 0:
            model_segment.eval()
            model_denoise.eval()
            # dice_value, auroc_px, auroc_sp = evaluation_DRAEM(args, model_denoise, model_segment, test_dataloader, epoch, loss_l1, run_name)
            # result_path = os.path.join('/home/zhaoxiang/output', run_name, 'results.txt')
            # print('Pixel Auroc:{:.3f}, Sample Auroc{:.3f}, Dice{:3f}'.format(auroc_px, auroc_sp, dice_value))
            
            # with open(result_path, 'a') as f:
            #     f.writelines('Epoch:{}, Pixel Auroc:{:.3f}, Sample Auroc{:.3f}, Dice:{:3f} \n'.format(epoch, auroc_px, auroc_sp, dice_value))   
            
            
            # auroc_sp = evaluation_DRAEM(args, model_denoise, model_segment, test_dataloader, epoch, loss_l1, run_name)
            # auroc_sp = evaluation_DRAEM_with_device(args, model_denoise, model_segment, test_dataloader, epoch, loss_l1, run_name, device)
            auroc_sp, dice_value = evaluation_DRAEM_half(args, model_denoise, model_segment, test_dataloader, epoch, loss_l1, run_name, device)
            # auroc_sp = 0.5ee_value))
            
            with open(result_path, 'a') as f:
                f.writelines('Epoch:{}, Sample Auroc{:.3f}, Dice{:.3f} \n'.format(epoch, auroc_sp, dice_value)) 
            
            # torch.save(model_segment.state_dict(), ckp_path.replace('last', 'segment'))
            torch.save({'model_denoise': model_denoise.state_dict(),
                        'model': model_segment.state_dict(),
                        'epoch': epoch}, ckp_path)
            
            if auroc_sp > best_SP:
                best_SP = auroc_sp
                torch.save({'model_denoise': model_denoise.state_dict(),
                        'model': model_segment.state_dict(),
                        'epoch': epoch,
                        'SP': best_SP,
                        'dice': dice_value}, ckp_path.replace('last', 'bestSP_{}_DICE_{}'.format(best_SP, dice_value)))
            
            if dice_value > best_dice:
                best_dice = dice_value
                torch.save({'model_denoise': model_denoise.state_dict(),
                        'model': model_segment.state_dict(),
                        'epoch': epoch,
                        'SP': best_SP,
                        'dice': dice_value}, ckp_path.replace('last', 'SP_{}_bestDICE_{}'.format(auroc_sp,best_dice)))
                
                
        wandb.log({"loss": np.mean(loss_list),
            "input_image": wandb.Image(aug),
            "reconstruction": wandb.Image(rec),
            "anomaly_mask": wandb.Image(anomaly_mask),
            "anomaly_output": wandb.Image(out_mask_sm[:,1:,:,:]),
            "auroc_sp": auroc_sp, "dice": dice_value})
    
    
    wandb.finish()
    

                
        
        

if __name__=="__main__":
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_id', default=1,  action='store', type=int)
    parser.add_argument('--lr', default=0.0001, action='store', type=float)
    # parser.add_argument('--lr', default=0.001, action='store', type=float)
    parser.add_argument('--epochs', default=150, action='store', type=int)
    parser.add_argument('--checkpoint_path', default='./checkpoints/', action='store', type=str)
    parser.add_argument('--log_path', default='./logs/', action='store', type=str)
    parser.add_argument('--visualize', default=True, action='store_true')

    parser.add_argument('--backbone', default='noise', action='store')
    
    # for noise autoencoder
    parser.add_argument("-nr", "--noise_res", type=float, default=16,  help="noise resolution.")
    parser.add_argument("-ns", "--noise_std", type=float, default=0.2, help="noise magnitude.")
    parser.add_argument("-img_size", "--img_size", type=float, default=256, help="noise magnitude.")
    
    # need to be changed/checked every time
    parser.add_argument('--bs', default = 16, action='store', type=int)
    parser.add_argument('--gpu_id', default=['0','1'], action='store', type=str, required=False)
    # parser.add_argument('--gpu_id', default='1', action='store', type=str, required=False)
    parser.add_argument('--experiment_name', default='DRAEM_Denoising_reject_weightDecay_bs_16_experiment_1', choices=['DRAEM_Denoising_reconstruction, liver, brain, head'], action='store')
    parser.add_argument('--colorRange', default=100, action='store')
    parser.add_argument('--threshold', default=200, action='store')
    parser.add_argument('--dataset_name', default='hist_DIY', choices=['hist_DIY', 'Brain_MRI', 'CovidX', 'RESC_average'], action='store')
    parser.add_argument('--model', default='DRAEM', choices=['ws_skip_connection', 'DRAEM_reconstruction', 'DRAEM_discriminitive'], action='store')
    parser.add_argument('--process_method', default='Gaussian_noise', choices=['none', 'Guassian_noise', 'DRAEM', 'Simplex_noise'], action='store')
    parser.add_argument('--multi_layer', default=False, action='store')
    parser.add_argument('--rejection', default=True, action='store')
    parser.add_argument('--number_iterations', default=1, action='store')
    parser.add_argument('--control_texture', default=False, action='store')
    parser.add_argument('--cutout', default=False, action='store')
    parser.add_argument('--resume_training', default=False, action='store')
    
    args = parser.parse_args()
    
    
    wandb.init(project='Liver',
           name = args.experiment_name,
           config={
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "batch_size": args.bs
    })
   
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

    with torch.cuda.device(args.gpu_id):
        setup_seed(928)
        # setup_seed(1234)
    # setup_seed(928)
    # setup_seed(1226)

        train_on_device(args)

