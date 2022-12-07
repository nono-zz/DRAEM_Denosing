from torchvision import transforms
from PIL import Image, ImageOps
import os
import torch
import glob
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, ImageFolder
import numpy as np

import cv2
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F

from anomaly_sythesis import distortion, blackStrip, randomShape, colorJitterRandom

from cutpaste_sythesis import CutPasteUnion, CutPaste3Way

# from cutpaste import CutPasteNormal,CutPasteScar, CutPaste3Way, CutPasteUnion, cut_paste_collate_fn, CutPasteBlack, Cut

def get_data_transforms(size, isize):
    # mean_train = [0.485]         # how do you set the mean_train and std_train in the get_data_transforms function?
    # mean_train = [-0.1]
    # std_train = [0.229]
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        
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
    x = x.unsqueeze(dim = 0)
    ns = torch.normal(mean=torch.zeros(x.shape[0], x.shape[1], noise_res, noise_res), std=noise_std).to(x.device)

    ns = F.upsample_bilinear(ns, size=[img_size, img_size])

    # Roll to randomly translate the generated noise.
    roll_x = random.choice(range(128))
    roll_y = random.choice(range(128))
    ns = torch.roll(ns, shifts=[roll_x, roll_y], dims=[-2, -1])
    mask = x.sum(dim=1, keepdim=True) > 0.01
    ns *= mask # Only apply the noise in the foreground.
    res = x + ns
    
    res = res.squeeze(dim = 0)
    return res


def cutpaste_transform(size,isize):
    cutpaste_type = CutPaste3Way
    
    after_cutpaste_transform = transforms.Compose([])
    after_cutpaste_transform.transforms.append(transforms.ToTensor())
    train_transform = transforms.Compose([])
    train_transform.transforms.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1))
    train_transform.transforms.append(transforms.Resize((size,size)))
    train_transform.transforms.append(cutpaste_type(transform = after_cutpaste_transform))
    
    gt_transforms = transforms.Compose([
                    transforms.Resize((size, size)),
                    transforms.CenterCrop(isize),
                    transforms.ToTensor()])
    
    return train_transform, gt_transforms

def get_data_transforms_augumentation(size, isize):
    mean_train = [0.485]
    std_train = [0.229]
    train_data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(10),
        # transforms.RandomCrop([10, 10]),              # 先不加crop
        transforms.RandomAffine(10),
        transforms.RandomAutocontrast(0.2),
        # transforms.RandomEqualize(0.1),                 # 先不加equalization
        transforms.ToTensor()])
        # transforms.Normalize(mean=mean_train,
        #                      std=std_train)])
    
    test_data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.CenterCrop(isize),
        #transforms.CenterCrop(args.input_size),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor()])
    
    return train_data_transforms, test_data_transforms, gt_transforms
    


class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase, dirs, data_source = 'liver', rgb=False, args=None):
        if len(dirs) == 3:
            [train_dir, test_dir, label_dir] = dirs
        elif len(dirs) == 2:
            [train_dir, test_dir] = dirs
            
        self.phase = phase
        self.transform = transform
        self.args = args
        
        self.cutpaste = CutPasteUnion(transform=self.transform)

        self.gt_transform = gt_transform
        self.data_source = data_source
        self.rgb = rgb
        
        if phase == 'train':
            self.img_path = os.path.join(root, 'train/good')
            # self.img_paths = glob.glob(self.img_path + "/*.png")
            if data_source == 'retina':
                self.img_paths = glob.glob(self.img_path + "/*.bmp")
            else:
                self.img_paths = glob.glob(self.img_path + "/*.png")
            self.img_paths.sort()
        elif phase == 'test':
            self.img_path = os.path.join(root, test_dir)
            self.gt_path = os.path.join(root, label_dir)
            # load dataset
            self.img_paths, self.gt_paths = self.load_dataset()  # self.labels => good : 0, anomaly : 1
            
        elif phase == 'eval':
            self.img_path = os.path.join(root, 'evaluation')
            self.gt_path = os.path.join(root, 'evaluation_label')
            self.img_paths, self.gt_paths = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):         # only used in test phase
        
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []
        
        if self.data_source == 'retina':
            defect_types = os.listdir(self.img_path)
            for defect_type in defect_types:
                if defect_type == 'good':
                    img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                    img_paths.sort()
                    img_tot_paths.extend(img_paths)
                    gt_tot_paths.extend(["/home/zhaoxiang/dataset/RESC_average/test_label/1.png"] * len(img_paths))

                    tot_labels.extend([0] * len(img_paths))
                    tot_types.extend(['good'] * len(img_paths))


                elif defect_type == 'Ungood':
                    img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                    gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                    img_paths.sort()
                    gt_paths.sort()
                    img_tot_paths.extend(img_paths)
                    gt_tot_paths.extend(gt_paths)

                    tot_labels.extend([1] * len(img_paths))
                    tot_types.extend([defect_type] * len(img_paths))
        
        
        else:
            img_paths = glob.glob(self.img_path + "/*.png")
            gt_paths = glob.glob(self.gt_path + "/*.png")            # ground truth mask.
            img_paths.sort()                        # list them with a specific sequence
            gt_paths.sort()
            img_tot_paths.extend(img_paths)         # what does extend means? add all the elements of an iterrable to the end of the list.
            gt_tot_paths.extend(gt_paths)

            assert len(img_tot_paths) == len(gt_tot_paths), "Number of test and ground truth pair doesn't match!"          # if not, then raise an error.

        return img_tot_paths, gt_tot_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        
        if self.phase == 'train':
            img_path = self.img_paths[idx]
            img = Image.open(img_path)
            img = ImageOps.grayscale(img)
            img = img.resize([self.args.img_size, self.args.img_size])
            
            """ Sample level augmentation"""
            img_numpy = np.array(img)
            # cv2.imwrite('img_numpy.png', img_numpy)
            if img_numpy.max() == 0:
                img_tensor = self.transform(img)
                img_tensor = img_tensor.repeat(2, 1, 1, 1)
                gt_tensor = torch.zeros_like(img_tensor)
                # return img_tensor,img_tensor, gt_tensor
                return img_tensor, img_tensor
            
            blackStrip_img, blackStrip_gt = blackStrip(img_numpy)
            # cv2.imwrite('blackStrip.png', blackStrip_img)
            
            randomShape_img, randomShape_gt, random_map = randomShape(img_numpy)
            # cv2.imwrite('randomShape.png', randomShape_img)
            
            randomShapeLow_img, randomShapeLow_gt, randomLow_map = randomShape(img_numpy, scaleUpper=10)
            # cv2.imwrite('randomShape.png', randomShapeLow_img)
            
            colorJitter_img, colorJitter_gt = colorJitterRandom(img_numpy, self.args, colorRange=self.args.colorRange, threshold=self.args.threshold, number_iterations=self.args.number_iterations)
            
            if self.args.rejection: # too generate the anomaly on each image
                while colorJitter_gt.sum() == 0:
                    colorJitter_img, colorJitter_gt = colorJitterRandom(img_numpy, self.args, colorRange=self.args.colorRange, threshold=self.args.threshold)
                
            
            # distortion_img = distortion(np.array(img))
            # cv2.imwrite('distortion.png', distortion_img)
            
            
            # distortion_img = np.expand_dims(distortion_img, axis=2)
            blackStrip_img = np.expand_dims(blackStrip_img, axis=2)
            randomShape_img = np.expand_dims(randomShape_img, axis=2)
            randomShapeLow_img = np.expand_dims(randomShapeLow_img, axis=2)
            colorJitter_img = np.expand_dims(colorJitter_img, axis=2)
            
            # distortion_img = self.transform(distortion_img)
            blackStrip_img = self.transform(blackStrip_img)
            randomShape_img = self.transform(randomShape_img)
            randomShapeLow_img = self.transform(randomShapeLow_img)
            colorJitter_img = self.transform(colorJitter_img)
            
            """ Pixel level augmentation"""
            org, cut_img, cut_gt = self.cutpaste(img)
            # img = self.transform(img)
            
            """ Gaussian Noise"""
            Gaussian_img = add_Gaussian_noise(org, self.args.noise_res, self.args.noise_std, self.args.img_size)  

            # img_list = [Gaussian_img, cut_img, distortion_img, blackStrip_img, randomShape_img, randomShapeLow_img]
            # img_list = [blackStrip_img, randomShape_img]
            img_list = [colorJitter_img]
            # gt_list = [cut_gt, blackStrip_gt, randomShape_gt, randomShapeLow_gt]
            # gt_list = [blackStrip_gt, randomShape_gt]
            gt_list = [colorJitter_gt]
            gt_list = [(np.where(x > 0, 255, 0)).astype(np.uint8) for x in gt_list]
            
            # gt_list.append(cut_gt)
            gt_list = [self.transform(x) for x in gt_list]
            gt_list = [x.unsqueeze(dim=0) for x in gt_list]
            gt_tensor = torch.cat(gt_list, dim=0)
            
            img_list = [x.unsqueeze(dim=0) for x in img_list]
            aug_tensor = torch.cat(img_list, dim = 0)
            
            org_tensor = (torch.unsqueeze(org, dim=0)).repeat(len(img_list), 1, 1, 1)
            return org_tensor, aug_tensor, gt_tensor
            # return org_tensor, aug_tensor
        
        else:
            img_path, gt_path= self.img_paths[idx], self.gt_paths[idx]
            # img = Image.open(img_path).convert('RGB')s
            img = Image.open(img_path)
            img = ImageOps.grayscale(img)
            img = img.resize([self.args.img_size, self.args.img_size])
            if self.rgb:    
                img = img.convert('RGB')
            img = self.transform(img)
            
            gt = Image.open(gt_path)
            gt = ImageOps.grayscale(gt)
            gt = gt.resize([self.args.img_size, self.args.img_size])
            gt = self.gt_transform(gt)

            # determine the label
            if torch.sum(gt) != 0:
                label = 1
            else:
                label = 0
                
            save = False
            return img, gt, label, img_path, save
        
        
class MVTecDataset_cross_validation(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase, data_source = 'liver', rgb=False, args=None, fold_index = 0, test_whole=True):
        self.phase = phase
        self.transform = transform
        self.args = args
        self.root = root
        
        self.cutpaste = CutPasteUnion(transform=self.transform)

        self.gt_transform = gt_transform
        self.data_source = data_source
        self.rgb = rgb
        
        file_path = os.path.join(root, 'fold_{}.npy'.format(fold_index))
        data = np.load(file_path, allow_pickle=True)
        
        self.train_names = data.item()['Train_images']
        self.test_names = data.item()['Test_images']
        
        if test_whole:
            self.test_names = os.listdir('/home/zhaoxiang/dataset/LiTs_with_labels/image')
        
        self.train_labels = data.item()['Train_labels']
        self.test_labels = data.item()['Test_labels']
        
        # retrieve all the healthy images as training samples
        self.label_mask = np.array(self.train_labels)
        self.label_mask = np.where(self.label_mask == 1, False, True)
        self.train_names = self.train_names[self.label_mask]
        
        
        self.PILtoTensor_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                    transforms.RandomRotation(10),
                    # transforms.RandomCrop([10, 10]),              # 先不加crop
                    transforms.RandomAffine(10),
                    transforms.RandomAutocontrast(0.2)
                    # transforms.ToTensor()
                    ])

    def __len__(self):
        if self.phase == 'train':
            return len(self.train_names)
        
        else:
            return len(self.test_names)

    def __getitem__(self, idx):
        
        if self.phase == 'train':
            img_name = self.train_names[idx]
            img_path = os.path.join(self.root, 'image', img_name)
            img = Image.open(img_path)
            img = ImageOps.grayscale(img)
            img = img.resize([self.args.img_size, self.args.img_size])
            img.save('before_transformations.png')
            
            img = self.PILtoTensor_transform(img)
            img.save('random_transformations.png')
            
            """ Sample level augmentation"""
            img_numpy = np.array(img)
            if img_numpy.max() == 0:
                img_tensor = self.transform(img)
                img_tensor = img_tensor.repeat(2, 1, 1, 1)
                gt_tensor = torch.zeros_like(img_tensor)
                return img_tensor, img_tensor
        
            
            colorJitter_img, colorJitter_gt = colorJitterRandom(img_numpy, colorRange=self.args.colorRange, threshold=self.args.threshold)
            
            colorJitter_img = np.expand_dims(colorJitter_img, axis=2)
            
            colorJitter_img = self.transform(colorJitter_img)
            
            """ Pixel level augmentation"""
            org, cut_img, cut_gt = self.cutpaste(img)
            # img = self.transform(img)
            
            img_list = [colorJitter_img]
            gt_list = [colorJitter_gt]
            gt_list = [(np.where(x > 0, 255, 0)).astype(np.uint8) for x in gt_list]
            
            gt_list = [self.transform(x) for x in gt_list]
            gt_list = [x.unsqueeze(dim=0) for x in gt_list]
            gt_tensor = torch.cat(gt_list, dim=0)
            
            img_list = [x.unsqueeze(dim=0) for x in img_list]
            aug_tensor = torch.cat(img_list, dim = 0)
            
            org_tensor = (torch.unsqueeze(org, dim=0)).repeat(len(img_list), 1, 1, 1)
            return org_tensor, aug_tensor, gt_tensor
        
        else:
            # img_path, gt_path= self.img_paths[idx], self.gt_paths[idx]
            img_name = self.test_names[idx]
            img_path = os.path.join(self.root, 'image', img_name)
            
            gt_name = img_name.replace('liver', 'liver_gt')
            gt_path = os.path.join(self.root, 'image_labels', gt_name)
            
            img = Image.open(img_path)
            img = ImageOps.grayscale(img)
            img = img.resize([self.args.img_size, self.args.img_size])
            if self.rgb:    
                img = img.convert('RGB')
            img = self.transform(img)
            
            gt = Image.open(gt_path)
            gt = ImageOps.grayscale(gt)
            gt = gt.resize([self.args.img_size, self.args.img_size])
            gt = self.gt_transform(gt)

            # determine the label
            if torch.sum(gt) != 0:
                label = 1
            else:
                label = 0
                
            save = False
            return img, gt, label, img_path, save
        
        
        
class MVTecDataset_fixed(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase, data_source = 'liver', rgb=False, args=None):
            
        self.phase = phase
        self.transform = transform
        self.args = args
        self.rgb = rgb
        

        self.gt_transform = gt_transform
        
        if phase == 'train':
            
            self.raw_path = os.path.join(root, 'train_raw')
            self.aug_path = os.path.join(root, 'train_aug')
            self.gt_path = os.path.join(root, 'train_label')
            
            self.raw_paths = glob.glob(self.raw_path + "/*.png")
            self.raw_paths.sort()
            self.aug_paths = glob.glob(self.aug_path + "/*.png")
            self.aug_paths.sort()
            self.gt_paths = glob.glob(self.gt_path + "/*.png")
            self.gt_paths.sort()
        elif phase == 'test':
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'test_label')
            # load dataset
            self.img_paths, self.gt_paths = self.load_dataset()  # self.labels => good : 0, anomaly : 1
            
        elif phase == 'eval':
            self.img_path = os.path.join(root, 'image')
            self.gt_path = os.path.join(root, 'image_labels')
            self.img_paths, self.gt_paths = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):         # only used in test phase
        
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []
        

        img_paths = glob.glob(self.img_path + "/*.png")
        gt_paths = glob.glob(self.gt_path + "/*.png")            # ground truth mask.
        img_paths.sort()                        # list them with a specific sequence
        gt_paths.sort()
        img_tot_paths.extend(img_paths)         # what does extend means? add all the elements of an iterrable to the end of the list.
        gt_tot_paths.extend(gt_paths)

        assert len(img_tot_paths) == len(gt_tot_paths), "Number of test and ground truth pair doesn't match!"          # if not, then raise an error.

        return img_tot_paths, gt_tot_paths

    def __len__(self):
        
        if self.phase == 'train':
            return len(self.raw_paths)
        else:
            return len(self.img_paths)

    def __getitem__(self, idx):
        
        if self.phase == 'train':
            raw_path = self.raw_paths[idx]
            aug_path = self.aug_paths[idx]
            gt_path = self.gt_paths[idx]
            
            raw = Image.open(raw_path)
            raw = ImageOps.grayscale(raw)
            raw = raw.resize([self.args.img_size, self.args.img_size])
            raw = self.transform(raw)
            
            aug = Image.open(aug_path)
            aug = ImageOps.grayscale(aug)
            aug = aug.resize([self.args.img_size, self.args.img_size])
            aug = self.transform(aug)
            
            gt = Image.open(gt_path)
            gt = ImageOps.grayscale(gt)
            gt = gt.resize([self.args.img_size, self.args.img_size])
            gt = self.gt_transform(gt)
            
            return raw, aug, gt
        
        else:
            img_path, gt_path= self.img_paths[idx], self.gt_paths[idx]
            # img = Image.open(img_path).convert('RGB')s
            img = Image.open(img_path)
            img = ImageOps.grayscale(img)
            img = img.resize([self.args.img_size, self.args.img_size])
            if self.rgb:    
                img = img.convert('RGB')
            img = self.transform(img)
            
            gt = Image.open(gt_path)
            gt = ImageOps.grayscale(gt)
            gt = gt.resize([self.args.img_size, self.args.img_size])
            gt = self.gt_transform(gt)

            # determine the label
            if torch.sum(gt) != 0:
                label = 1
            else:
                label = 0
                
            save = False
            return img, gt, label, img_path, save
    
    
    
class Medical_dataset(torch.utils.data.Dataset):
    def __init__(self, transform, phase, args):
        self.phase = phase
        
        if args.dataset_name == 'Brain_MRI':
            self.root = '/home/zhaoxiang/dataset/Brain_MRI'
            if phase == 'train':
                self.img_path = os.path.join(self.root, 'train')
                
            else:
                self.img_path = os.path.join(self.root, 'test')
                
        # head CT
        elif args.dataset_name == 'Head_CT':
            self.root = '/home/zhaoxiang/dataset/Head_CT/imgs'
            if phase == 'train':
                self.img_path = os.path.join(self.root, 'train')
            else:
                self.img_path = os.path.join(self.root, 'test')
                
        
        elif args.dataset_name == 'CovidX':
            self.root = '/home/zhaoxiang/dataset/CovidX'
            if phase == 'train':
                self.img_path = os.path.join(self.root, 'train')
            else:
                self.img_path = os.path.join(self.root, 'test')
                
        self.transform = transform
        self.img_paths = self.load_dataset()
        self.args = args
        
    def load_dataset(self):
    
        img_tot_paths = []
        img_paths = []
        
        
        for (dirpath, dirnames, filenames) in os.walk(self.img_path):
            for filename in filenames:
                img_paths.append(os.path.join(dirpath, filename))
        
        img_paths.sort()                        # list them with a specific sequence
        img_tot_paths.extend(img_paths)         # what does extend mean
        
        return img_tot_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path)
        img = ImageOps.grayscale(img)
        img = self.transform(img)

        # determine the label by checking y and no
        img_name = img_path.split('/')[-1]
        if self.args.dataset_name in ['Brain_MRI', 'Head_CT']:
            if 'Y' in img_name:
                label = 1
            else:
                label = 0
                
        elif self.args.dataset_name in ['CovidX']:
            if 'good' in img_path:
                label = 0
            else:
                label = 1

        return img, label, img_path


