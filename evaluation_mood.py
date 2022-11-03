import torch
import os
import cv2
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.ndimage import gaussian_filter
import pandas as pd
from sklearn.metrics import auc
from skimage import measure
import pandas as pd
from numpy import ndarray
from statistics import mean
from scipy.ndimage import gaussian_filter
from torch.nn import functional as F
import pathlib
import csv

import os

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


def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)


def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        #fs_norm = F.normalize(fs, p=2)
        #ft_norm = F.normalize(ft, p=2)
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list


def mean(list_x):
    return sum(list_x)/len(list_x)

def cal_distance_map(input, target):
    # input = np.squeeze(input, axis=0)
    # target = np.squeeze(target, axis=0)
    d_map = np.full_like(input, 0)
    d_map = np.square(input - target)
    return d_map

def dice(pred, gt):
    intersection = (pred*gt).sum()
    return (2. * intersection)/(pred.sum() + gt.sum())

        
def evaluation(args, classifier, decoder, bn, test_dataloader, epoch, loss_function, run_name, threshold = 0.5):
    
    classifier.eval()
    decoder.eval()
    bn.eval()
    
    dice_error = []
        
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    aupro_list = []
    pr_binary_list_px = []
    
    with torch.no_grad():
        for img, gt, label, img_path, save in test_dataloader:

            img = img.cuda()
            gt[gt > 0.1] = 1
            gt[gt <= 0.1] = 0
            # check if img is RGB
            question = classifier(img)
            answer = decoder(bn(question))       # inputs and outputs are both 'list' here
            
            initial_feature = img

            anomaly_map, anomaly_list = cal_anomaly_map(question, answer, img.shape[-1], amap_mode='a')
            # anomaly_map, anomaly_list = cal_anomaly_map(question[1:], answer[1:], img.shape[-1], amap_mode='a')
            # distance_map = cal_distance_map(answer[0].to('cpu').detach().numpy(), img[:,0,:,:].to('cpu').detach().numpy())
            
            # names = ['liver_1_60', 'liver_2_452', 'liver_3_394', 'liver_4_457', 'liver_6_396', 'liver_7_487', 'liver_10_379'] 
            names = ['liver_1_59', 'liver_1_60', 'liver_1_66', 'liver_1_67', 'liver_2_415', 'liver_2_452', 'liver_3_394', 'liver_4_566', 'liver_4_457', 'liver_6_396', 'liver_8_448', 'liver_7_487', 'liver_10_328', 
                            'liver_10_335' 'liver_10_356', 'liver_10_379',  'liver_10_295', 'liver_10_422' , 'liver_16_374', 'liver_16_409', 'liver_18_413', 'liver_19_440', 'liver_20_487', 'liver_21_388', 'liver_23_335',
                            'liver_26_317', 'liver_27_414', 'liver_27_559', 'liver_13_393', 'liver_12_412', 'liver_11_413']
            gt = gt[0,0,:,:].to('cpu').detach().numpy()  
            
            for name in names:
                if name in img_path[0]:
                    # save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs', run_name, name)
                    save_dir = os.path.join('/home/zhaoxiang', 'output', run_name, name)
                    
                    
                    a_map_path = os.path.join(save_dir, 'a_map_{}.png'.format(epoch))
                    p_map_path = os.path.join(save_dir, 'p_map_{}.png'.format(epoch))
                    d_map_path = os.path.join(save_dir, 'd_map_{}.png'.format(epoch))
                    initial_feature_path = os.path.join(save_dir, 'raw.png')
                    reconstruction_path = os.path.join(save_dir, 'reconstruction_{}.png'.format(epoch))
                    gt_path = os.path.join(save_dir, 'gt.png')
                    anomaly_map_rgb_path = os.path.join(save_dir, 'combine.png')
                    
                    if not os.path.exists(save_dir):
                        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
                        
                    # save inital_feature and reconstruction
                    initial_feature = min_max_norm(initial_feature[0,:,:,:].to('cpu').detach().numpy())
                    initial_feature = np.transpose(initial_feature, (1,2,0))
                    cv2.imwrite(initial_feature_path, initial_feature*255)
                    
                    rec = min_max_norm(rec[0,:,:,:].to('cpu').detach().numpy())
                    rec = np.transpose(rec, (1,2,0))
                    cv2.imwrite(initial_feature_path, rec*255)
                    
                    
                    cv2.imwrite(gt_path, gt * 255)
                    
                    # save reconstructed image
                    # img_reconstruction, _, _, _ = answer
                    # img_reconstruction = min_max_norm(img_reconstruction[0,:,:,:].to('cpu').detach().numpy())
                    # img_reconstruction = np.transpose(img_reconstruction, (1,2,0))
                    # cv2.imwrite(reconstruction_path, img_reconstruction * 255)
                    
                    # distance_map = np.transpose(distance_map, (1,2,0))
                    # distance_map = min_max_norm(distance_map)
                    # cv2.imwrite(d_map_path, distance_map*255)
                    
                        
                    # if config['multi_layer']:
                    #     mode = ['a', 'b', 'c']
                    #     for i, a_map_layer in enumerate(anomaly_list):
                    #         a_map_layer_dir = os.path.join('/home/zhaoxiang/experiments_zzx/reconstruction/outputs', 
                    #                 config['experiment_name'], config['dataset_name'], 
                    #                 config['learning_rate'], name, mode[i])
                    #         if not os.path.exists(a_map_layer_dir):
                    #             os.mkdir(a_map_layer_dir)
                                
                    #         a_map_layer_path = os.path.join(a_map_layer_dir, 'a_map_{}_{}.png'.format(epoch, mode[i]))
                    #         a_map_layer = gaussian_filter(a_map_layer, sigma=4)
                    #         a_map_layer = min_max_norm(a_map_layer)
                    #         cv2.imwrite(a_map_layer_path, a_map_layer*255)
                            
                    #         # set threshold to convert a_map to a binary map
                    #         # threshold = 0.8
                    #         p_map_layer = np.where(a_map_layer > threshold, 255, 0)
                    #         p_map_layer_path = os.path.join('/home/zhaoxiang/experiments_zzx/reconstruction/outputs', 
                    #                 config['experiment_name'], config['dataset_name'], 
                    #                 config['learning_rate'], name, 'p_map_{}_{}.png'.format(epoch, mode[i]))
                    #         cv2.imwrite(p_map_layer_path, p_map_layer)
                    label = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                    anomaly_map_rgb = cv2.imread(a_map_path)
                    label = cv2.resize(label, [224, 224])                    
                    anomaly_map_rgb[:,:,1] = label
                    cv2.imwrite(anomaly_map_rgb_path, anomaly_map_rgb)
                            
                            
                    # anomaly_map = gaussian_filter(anomaly_map, sigma=4)
                    # anomaly_map = min_max_norm(anomaly_map)
                    prediction_map = np.where(anomaly_map > threshold, 255, 0)
                    cv2.imwrite(a_map_path, anomaly_map*255)
                    cv2.imwrite(p_map_path, prediction_map)
            

            # anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            # prediction_map = np.where(min_max_norm(anomaly_map) > threshold, 1, 0)
            prediction_map = np.where(anomaly_map > threshold, 1, 0)

            
            # if label.item()!=0:
                # aupro_list.append(compute_pro(gt.squeeze(0).cpu().numpy().astype(int),
                #                             anomaly_map[np.newaxis,:,:]))
                # dice_error.append(dice(prediction_map, gt.squeeze(0).cpu().numpy().astype(int)))  
            
            gt_list_px.extend(gt.astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())
            pr_binary_list_px.extend(prediction_map.ravel())
            gt_list_sp.append(np.max(gt.astype(int)))
            pr_list_sp.append(np.max(anomaly_map))
        
        # dice_value = mean(dice_error)
        dice_value = dice(np.array(gt_list_px), np.array(pr_binary_list_px))
        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 3)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3)
        
        
    return dice_value, auroc_px, auroc_sp, round(np.mean(aupro_list), 3)


def evaluation_DRAEM(args, model_denoise, model_segment, test_dataloader, epoch, loss_function, run_name, threshold = 0.5):
    
    model_denoise.eval()
    model_segment.eval()
    
    dice_error = []
        
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    aupro_list = []
    pr_binary_list_px = []
    
    with torch.no_grad():
        for img, gt, label, img_path, save in test_dataloader:

            img = img.cuda()
            gt[gt > 0.1] = 1
            gt[gt <= 0.1] = 0
            # check if img is RGB
            rec = model_denoise(img)
                    
            joined_in = torch.cat((rec, img), dim=1)
            
            out_mask = model_segment(joined_in)
            out_mask_sm = torch.softmax(out_mask, dim=1)
            
            initial_feature = img

            anomaly_map = out_mask_sm
            
            # names = ['liver_1_60', 'liver_2_452', 'liver_3_394', 'liver_4_457', 'liver_6_396', 'liver_7_487', 'liver_10_379'] 
            names = ['liver_1_59', 'liver_1_60', 'liver_1_66', 'liver_1_67', 'liver_2_415', 'liver_2_452', 'liver_3_394', 'liver_4_566', 'liver_4_457', 'liver_6_396', 'liver_8_448', 'liver_7_487', 'liver_10_328', 
                            'liver_10_335' 'liver_10_356', 'liver_10_379',  'liver_10_295', 'liver_10_422' , 'liver_16_374', 'liver_16_409', 'liver_18_413', 'liver_19_440', 'liver_20_487', 'liver_21_388', 'liver_23_335',
                            'liver_26_317', 'liver_27_414', 'liver_27_559', 'liver_13_393', 'liver_12_412', 'liver_11_413']
            gt = gt[0,0,:,:].to('cpu').detach().numpy()  
            anomaly_map = anomaly_map[0,1,:,:].to('cpu').detach().numpy()  
            
            for name in names:
                if name in img_path[0]:
                    save_dir = os.path.join('/home/zhaoxiang', 'output', run_name, name)
                    
                    
                    a_map_path = os.path.join(save_dir, 'a_map_{}.png'.format(epoch))
                    p_map_path = os.path.join(save_dir, 'p_map_{}.png'.format(epoch))
                    d_map_path = os.path.join(save_dir, 'd_map_{}.png'.format(epoch))
                    initial_feature_path = os.path.join(save_dir, 'raw.png')
                    reconstruction_path = os.path.join(save_dir, 'reconstruction_{}.png'.format(epoch))
                    gt_path = os.path.join(save_dir, 'gt.png')
                    anomaly_map_rgb_path = os.path.join(save_dir, 'combine.png')
                    
                    if not os.path.exists(save_dir):
                        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
                        
                    # save inital_feature and reconstruction
                    initial_feature = min_max_norm(initial_feature[0,:,:,:].to('cpu').detach().numpy())
                    initial_feature = np.transpose(initial_feature, (1,2,0))
                    cv2.imwrite(initial_feature_path, initial_feature*255)
                    
                    rec = min_max_norm(rec[0,:,:,:].to('cpu').detach().numpy())
                    rec = np.transpose(rec, (1,2,0))
                    cv2.imwrite(reconstruction_path, rec*255)
                    
                    cv2.imwrite(gt_path, gt * 255)
                    
                    # anomaly_map = gaussian_filter(anomaly_map, sigma=4)
                    cv2.imwrite(a_map_path, anomaly_map * 255)
                    
                    # combine
                    label = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                    anomaly_map_rgb = cv2.imread(a_map_path)
                    label = cv2.resize(label, [256, 256])                    
                    anomaly_map_rgb[:,:,1] = label
                    cv2.imwrite(anomaly_map_rgb_path, anomaly_map_rgb)
                            
                            
                    prediction_map = np.where(anomaly_map > threshold, 255, 0)
                    cv2.imwrite(a_map_path, anomaly_map*255)
                    cv2.imwrite(p_map_path, prediction_map)
            

            # anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            # prediction_map = np.where(min_max_norm(anomaly_map) > threshold, 1, 0)
            prediction_map = np.where(anomaly_map > threshold, 1, 0)

        
            gt_list_px.extend(gt.astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())
            pr_binary_list_px.extend(prediction_map.ravel())
            gt_list_sp.append(np.max(gt.astype(int)))
            pr_list_sp.append(np.max(anomaly_map))
        
        # dice_value = mean(dice_error)
        dice_value = dice(np.array(gt_list_px), np.array(pr_binary_list_px))
        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 3)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3)
        
        
    return dice_value, auroc_px, auroc_sp

def evaluation_reconstruction(args, model, test_dataloader, epoch, loss_function, run_name, threshold = 0.1):
    
    model.eval()
    
    dice_error = []
        
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    aupro_list = []
    pr_binary_list_px = []
    img_paths, preds, gts, intersections, dices, a_map_max, losses, losses_feature, losses_reconstruction = [], [], [], [], [], [], [], [], []

    
    with torch.no_grad():
        for img, gt, label, img_path, save in test_dataloader:

            img = img.cuda()
            gt[gt > 0.1] = 1
            gt[gt <= 0.1] = 0
            # check if img is RGB
            rec = model(img)
            
            initial_feature = img.to('cpu').detach().numpy()

            difference = cal_distance_map(rec[0,0,:,:].to('cpu').detach().numpy(), img[0,0,:,:].to('cpu').detach().numpy())            
            # names = ['liver_1_60', 'liver_2_452', 'liver_3_394', 'liver_4_457', 'liver_6_396', 'liver_7_487', 'liver_10_379'] 
            names = ['liver_1_59', 'liver_1_60', 'liver_1_66', 'liver_1_67', 'liver_2_415', 'liver_2_452', 'liver_3_394', 'liver_4_566', 'liver_4_457', 'liver_6_396', 'liver_8_448', 'liver_7_487', 'liver_10_328', 
                            'liver_10_335' 'liver_10_356', 'liver_10_379',  'liver_10_295', 'liver_10_422' , 'liver_16_374', 'liver_16_409', 'liver_18_413', 'liver_19_440', 'liver_20_487', 'liver_21_388', 'liver_23_335',
                            '01446_84','01449_97', '01448_80', '01453_83', '01454_102', '01458_96', '01465_113', '01467_78',
                            '01474_78', '01473_90', '01476_71', '01484_85', '01487_73', '-1501_57','01517_81', '01554_65',
                            '01573_103', '01575_90', '01633_62']
            
            gt = gt[0,0,:,:].to('cpu').detach().numpy()  
            
            for name in names:
                if name in img_path[0]:
                    save_dir = os.path.join('/home/zhaoxiang', 'output', run_name, name)
                    
                    
                    a_map_path = os.path.join(save_dir, 'a_map_{}.png'.format(epoch))
                    p_map_path = os.path.join(save_dir, 'p_map_{}.png'.format(epoch))
                    d_map_path = os.path.join(save_dir, 'd_map_{}.png'.format(epoch))
                    initial_feature_path = os.path.join(save_dir, 'raw.png')
                    reconstruction_path = os.path.join(save_dir, 'reconstruction_{}.png'.format(epoch))
                    gt_path = os.path.join(save_dir, 'gt.png')
                    anomaly_map_rgb_path = os.path.join(save_dir, 'combine.png')
                    # evaluation_reconstructionfeature = np.transpose(initial_feature, (1,2,0))
                    cv2.imwrite(initial_feature_path, initial_feature*255)
                    
                    rec = min_max_norm(rec[0,:,:,:].to('cpu').detach().numpy())
                    rec = np.transpose(rec, (1,2,0))
                    cv2.imwrite(reconstruction_path, rec*255)
                    
                    cv2.imwrite(gt_path, gt * 255)
                    # cv2.imwrite(d_map_path, difference*255)
                                        
                    # combine
                    # label = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                    # anomaly_map_rgb = cv2.imread(a_map_path)
                    # label = cv2.resize(label, [256, 256])                    
                    # anomaly_map_rgb[:,:,1] = label
                    # cv2.imwrite(anomaly_map_rgb_path, anomaly_map_rgb)
                            
                            
                    prediction_map = np.where(difference > threshold, 255, 0)
                    cv2.imwrite(a_map_path, difference*255)
                    cv2.imwrite(p_map_path, prediction_map)
            

            # anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            # prediction_map = np.where(min_max_norm(anomaly_map) > threshold, 1, 0)
            prediction_map = np.where(difference > threshold, 1, 0)

        
            gt_list_px.extend(gt.astype(int).ravel())
            pr_list_px.extend(difference.ravel())
            pr_binary_list_px.extend(prediction_map.ravel())
            gt_list_sp.append(np.max(gt.astype(int)))
            pr_list_sp.append(np.max(difference))
            
            intersection = (prediction_map.ravel() * gt.astype(int).ravel()).sum()
            
            img_paths.append(img_path[0].split('/')[-1])
            preds.append(prediction_map.sum())
            gts.append(gt.sum())
            intersections.append(intersection)
            dice_sample_value = dice(prediction_map, gt)
            # dices.append(dice(prediction_map, gt.squeeze(0).squeeze(0).cpu().numpy().astype(int)))
            dices.append(dice_sample_value)
            a_map_max.append(difference.max())
        
        # dice_value = mean(dice_error)
        dice_value = dice(np.array(gt_list_px), np.array(pr_binary_list_px))
        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 3)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3)
        
        csv_path = os.path.join('/home/zhaoxiang/output', run_name, 'dice_results.csv')
        df = pd.DataFrame({'img_path': img_paths, 'pred': preds, 'gt': gts, 'intersection': intersections, 'dice': dices, 'a_map_max': a_map_max})
        df.to_csv(csv_path, index=False)
        
    return dice_value, auroc_px, auroc_sp


def evaluation_reconstruction_AP(args, model, test_dataloader, epoch, loss_function, run_name, threshold = 0.1):
    
    model.eval()
    
    dice_error = []
        
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    aupro_list = []
    pr_binary_list_px = []
    img_paths, preds, gts, intersections, dices, a_map_max, losses, losses_feature, losses_reconstruction = [], [], [], [], [], [], [], [], []

    with torch.no_grad():
        for img, gt, label, img_path, save in test_dataloader:

            img = img.cuda()
            gt[gt > 0.1] = 1
            gt[gt <= 0.1] = 0
            # check if img is RGB
            rec = model(img)
            
            initial_feature = img

            difference = cal_distance_map(rec[0,0,:,:].to('cpu').detach().numpy(), img[0,0,:,:].to('cpu').detach().numpy())            
            # names = ['liver_1_60', 'liver_2_452', 'liver_3_394', 'liver_4_457', 'liver_6_396', 'liver_7_487', 'liver_10_379'] 
            names = ['liver_1_59', 'liver_1_60', 'liver_1_66', 'liver_1_67', 'liver_2_415', 'liver_2_452', 'liver_3_394', 'liver_4_566', 'liver_4_457', 'liver_6_396', 'liver_8_448', 'liver_7_487', 'liver_10_328', 
                            'liver_10_335' 'liver_10_356', 'liver_10_379',  'liver_10_295', 'liver_10_422' , 'liver_16_374', 'liver_16_409', 'liver_18_413', 'liver_19_440', 'liver_20_487', 'liver_21_388', 'liver_23_335',
                            'liver_26_317', 'liver_27_414', 'liver_27_559', 'liver_13_393', 'liver_12_412', 'liver_11_413']
            if args.dataset_name == 'BraTs':
                names = ['01418_88', '01419_68', '01425_67', '01427_72', '01429_65', '01433_95', '01400_95', '01444_52',
                            '01446_84','01449_97', '01448_80', '01453_83', '01454_102', '01458_96', '01465_113', '01467_78',
                            '01474_78', '01473_90', '01476_71', '01484_85', '01487_73', '-1501_57','01517_81', '01554_65',
                            '01573_103', '01575_90', '01633_62']
            
            gt = gt[0,0,:,:].to('cpu').detach().numpy()  
            
            for name in names:
                if name in img_path[0]:
                    save_dir = os.path.join('/home/zhaoxiang', 'output', run_name, name)
                    
                    
                    a_map_path = os.path.join(save_dir, 'a_map_{}.png'.format(epoch))
                    p_map_path = os.path.join(save_dir, 'p_map_{}.png'.format(epoch))
                    d_map_path = os.path.join(save_dir, 'd_map_{}.png'.format(epoch))
                    initial_feature_path = os.path.join(save_dir, 'raw.png')
                    reconstruction_path = os.path.join(save_dir, 'reconstruction_{}.png'.format(epoch))
                    gt_path = os.path.join(save_dir, 'gt.png')
                    anomaly_map_rgb_path = os.path.join(save_dir, 'combine.png')
                    
                    if not os.path.exists(save_dir):
                        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
                        
                    # save inital_feature and reconstruction
                    initial_feature = min_max_norm(initial_feature[0,:,:,:].to('cpu').detach().numpy())
                    initial_feature = np.transpose(initial_feature, (1,2,0))
                    cv2.imwrite(initial_feature_path, initial_feature*255)
                    
                    rec = min_max_norm(rec[0,:,:,:].to('cpu').detach().numpy())
                    rec = np.transpose(rec, (1,2,0))
                    cv2.imwrite(reconstruction_path, rec*255)
                    
                    cv2.imwrite(gt_path, gt * 255)
                    # cv2.imwrite(d_map_path, difference*255)
                                        
                    # combine
                    # label = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                    # anomaly_map_rgb = cv2.imread(a_map_path)
                    # label = cv2.resize(label, [256, 256])                    
                    # anomaly_map_rgb[:,:,1] = label
                    # cv2.imwrite(anomaly_map_rgb_path, anomaly_map_rgb)
                            
                            
                    prediction_map = np.where(difference > threshold, 255, 0)
                    cv2.imwrite(a_map_path, difference*255)
                    cv2.imwrite(p_map_path, prediction_map)
            
            # anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            # prediction_map = np.where(min_max_norm(anomaly_map) > threshold, 1, 0)
            prediction_map = np.where(difference > threshold, 1, 0)

        
            gt_list_px.extend(gt.astype(int).ravel())
            pr_list_px.extend(difference.ravel())
            pr_binary_list_px.extend(prediction_map.ravel())
            gt_list_sp.append(np.max(gt.astype(int)))
            pr_list_sp.append(np.max(difference))
            
            intersection = (prediction_map.ravel() * gt.astype(int).ravel()).sum()
            
            img_paths.append(img_path[0].split('/')[-1])
            preds.append(prediction_map.sum())
            gts.append(gt.sum())
            intersections.append(intersection)
            dice_sample_value = dice(prediction_map, gt)
            # dices.append(dice(prediction_map, gt.squeeze(0).squeeze(0).cpu().numpy().astype(int)))
            dices.append(dice_sample_value)
            a_map_max.append(difference.max())
        
        # dice_value = mean(dice_error)
        dice_value = dice(np.array(gt_list_px), np.array(pr_binary_list_px))
        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 3)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3)
        Average_precesion = round(average_precision_score(gt_list_px, pr_list_px), 3)
        
        csv_path = os.path.join('/home/zhaoxiang/output', run_name, 'dice_results.csv')
        df = pd.DataFrame({'img_path': img_paths, 'pred': preds, 'gt': gts, 'intersection': intersections, 'dice': dices, 'a_map_max': a_map_max})
        df.to_csv(csv_path, index=False)
        
    return dice_value, auroc_px, auroc_sp, Average_precesion

def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:
    
    
    """
    Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """
    
    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=np.bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        # df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)
        df_new = pd.DataFrame([[mean(pros), fpr, th]],  columns=['pro', 'fpr', 'threshold'])
        # df = df.concat({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)
        df = pd.concat([df, df_new], ignore_index=True)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc
        
        
        
        
if __name__ == '__main__':
    import argparse
    from model_noise import UNet
    
    from dataloader_zzx import MVTecDataset
    from torch.utils.data import DataLoader
    from tensorboard_visualizer import TensorboardVisualizer
    from torchvision import transforms
    
    
    

    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_id', default=1,  action='store', type=int)
    parser.add_argument('--lr', default=0.0001, action='store', type=float)
    parser.add_argument('--epochs', default=700, action='store', type=int)
    parser.add_argument('--c v', default='/home/zhaoxiang/baselines/DRAEM/datasets/mvtec/', action='store', type=str)
    parser.add_argument('--anomaly_source_path', default='/home/zhaoxiang/baselines/DRAEM/datasets/dtd/images/', action='store', type=str)
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
    parser.add_argument('--gpu_id', default = ['0','1'], action='store', type=str, required=False)
    parser.add_argument('--experiment_name', default='mood_cv2', choices=['retina, liver, brain, head', 'chest'], action='store')
    parser.add_argument('--dataset_name', default='Mood_brain_cv2', choices=['hist_DIY', 'Brain_MRI', 'Head_CT', 'CovidX', 'RESC_average'], action='store')
    parser.add_argument('--model', default='ws_skip_connection', choices=['ws_skip_connection', 'DRAEM_reconstruction', 'DRAEM_discriminitive'], action='store')
    parser.add_argument('--process_method', default='Gaussian_noise', choices=['none', 'Gaussian_noise', 'DRAEM_natural', 'DRAEM_tumor', 'Simplex_noise', 'Simplex_noise_best_best'], action='store')
    parser.add_argument('--resume_training', default = True, action='store', type=int)
    
    parser.add_argument('--augmentation', default = 'blackStrip', action='store', type=str)
    
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
    
    data_transform, gt_transform = get_data_transforms(args.img_size, args.img_size)
    main_path = '/home/zhaoxiang/dataset/{}'.format(args.dataset_name)
    
    for args.augmentation in ['toy']:
        # for args.process_method in ['Gaussian_noise', 'Multi_randomShape_wo_distortion', 'Multi_randomShape', 'Multi']:
        for args.process_method in ['Multi_randomShape_wo_distortion', 'Multi']:
        
            dirs = os.listdir(main_path)
            
            # for dir_name in dirs:
            #     if 'train' in dir_name:
            #         train_dir = dir_name
            #     elif 'test' in dir_name and args.augmentation in dir_name:
            #         if 'label' in dir_name:
            #             label_dir = dir_name
            #         else:
            #             test_dir = dir_name
            # if 'label_dir' in locals():
            #     dirs = [train_dir, test_dir, label_dir] 
            for dir_name in dirs:
                if 'train' in dir_name:
                    train_dir = dir_name
                elif dir_name == 'test':
                    test_dir = dir_name
                elif 'test_label' == dir_name:
                    label_dir = dir_name

                        
            if 'label_dir' in locals():
                dirs = [train_dir, test_dir, label_dir]                

            
            
            device = None
            n_input = 1
            n_classes = 1           # the target is the reconstructed image
            depth = 4
            wf = 6
            
            run_name = args.experiment_name + '_' +str(args.lr)+'_'+str(args.epochs)+'_bs'+str(args.bs)+"_" + args.model + "_" + args.process_method
            
            loss_l1 = torch.nn.L1Loss()
            visualizer = TensorboardVisualizer(log_dir=os.path.join(args.log_path, run_name+"/"))
            
            model = UNet(in_channels=n_input, n_classes=n_classes, norm="group", up_mode="upconv", depth=depth, wf=wf, padding=True).cuda()
            if not 'Gaussian' in args.process_method:
                model = torch.nn.DataParallel(model, device_ids=[0, 1])
            
            if args.resume_training:
                base_path= '/home/zhaoxiang/baselines/pretrain'
                output_path = os.path.join(base_path, 'output')

                experiment_path = os.path.join(output_path, run_name)
                ckp_path = os.path.join(experiment_path, 'last.pth')
                model.load_state_dict(torch.load(ckp_path))
            
            val_data = MVTecDataset(root=main_path, transform = data_transform, gt_transform=gt_transform, phase='test', dirs = dirs, data_source=args.experiment_name)
            val_dataloader = torch.utils.data.DataLoader(val_data, batch_size = 1, shuffle = False)
            
            
            # names = ['20_120', '19_184', '18_100', '11_60', '12_80', '13_90', '14_100', '15_110', '16_220', '17_150']
            epoch = 'test'
            output_path =  '/home/zhaoxiang/output/{}'.format(args.augmentation)
            with torch.no_grad():
                model.eval()
                evaluation(args, model, val_dataloader, epoch, loss_l1, visualizer, run_name, output_path = output_path)