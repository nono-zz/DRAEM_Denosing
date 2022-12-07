import os
from turtle import color
import numpy as np
import torch
import cv2
import random
from PIL import Image, ImageOps

from cutpaste_sythesis import CutPasteUnion, CutPaste3Way
from torchvision import transforms
from torchvision.utils import save_image

import skimage.exposure
import numpy as np
from numpy.random import default_rng

# from scipy.misc import lena


"""Here we define all kinds of pseudo anomalies that can be directly apply on single images
"""

def getBbox(image):
    mask = np.zeros_like(image)
    B = np.argwhere(image)
    (ystart, xstart), (ystop, xstop) = B.min(0), B.max(0) + 1
    mask[ystart:ystop, xstart:xstop] = 1
    return mask, (ystart, xstart), (ystop, xstop)
        

# corruptions
def singleStrip(img, start, stop, mode, p = 0.3):
    if mode == 0:       # do horizonally
        start = start[0]
        stop = stop[0]
        # width = random.randint(0, start - stop)
        width = random.randint(0, int((stop - start) * p))
        
        stripStart = random.randint(start, stop - width)
        stripStop = stripStart + width
        
        # generate a mask
        mask = np.ones_like(img)
        mask[stripStart:stripStop, :] = 0
        
        new_img = mask * img
        return new_img, mask
    
    elif mode == 1:
        start = start[1]
        stop = stop[1]
        # width = random.randint(start, stop)
        
        width = random.randint(0, int((stop - start) * p))
        stripStart = random.randint(start, stop - width)
        stripStop = stripStart + width
        
        # generate a mask
        mask = np.ones_like(img)
        mask[:, stripStart:stripStop] = 0
        
        new_img = mask * img
        return new_img, mask
        


def blackStrip(img):
    # try:
    mask, start, stop = getBbox(img)
        
    # except:
    #     return None
    if mask.sum() > 800:
        # decide which mode it is
        mode = random.randint(0,2)
        if mode != 2:       # do horizonally
            new_img, stripMask = singleStrip(img, start, stop, mode)
            gtMask = mask*(1-stripMask)
            # return new_img, gtMask
            return new_img, gtMask     
        else:
            img_1, stripMask_1 = singleStrip(img, start, stop, mode = 0)
            new_img, stripMask_2 = singleStrip(img_1, start, stop, mode = 1)

            gt_mask = (1 - (stripMask_1 * stripMask_2)) * mask
            return new_img, gt_mask
            # return new_img
        
    else:
        gt_mask = np.zeros_like(img)
        return img, gt_mask

            
        
        
""" distortion
"""
def distortion(sss):
    img = sss
    symbol = random.randint(0,1)
    if symbol == 0:
        A = img.shape[0] / 3.0
    else:
        A = -img.shape[0] / 3.0
    
    i = random.randint(3,7)
    w = i/100 / img.shape[1]

    shift = lambda x: A * np.sin(2.0*np.pi*x * w)

    mode = random.randint(0,2)
    if mode == 0:
        for i in range(img.shape[0]):
            img[:,i] = np.roll(img[:,i], int(shift(i)))
    elif mode == 1:
        for i in range(img.shape[0]):
            img[i,:] = np.roll(img[i,:], int(shift(i)))
    else:
        for i in range(img.shape[0]):
            img[:,i] = np.roll(img[:,i], int(shift(i)))
        for i in range(img.shape[0]):
            img[i,:] = np.roll(img[i,:], int(shift(i)))
    return img


def cp(img_path):
    img = Image.open(img_path)
    img = ImageOps.grayscale(img)
    
    org, cut_img = cutpaste(img)
    return org, cut_img


"""random shape
"""
def randomShape(img, scaleUpper=255, threshold=200):


    # define random seed to change the pattern
    rng = default_rng()

    # define image size
    width=img.shape[0]
    height=img.shape[1]

    # create random noise image
    noise = rng.integers(0, 255, (height,width), np.uint8, True)

    # blur the noise image to control the size
    blur = cv2.GaussianBlur(noise, (0,0), sigmaX=15, sigmaY=15, borderType = cv2.BORDER_DEFAULT)

    # stretch the blurred image to full dynamic range
    stretch = skimage.exposure.rescale_intensity(blur, in_range='image', out_range=(0,255)).astype(np.uint8)

    # threshold stretched image to control the size
    # thresh = cv2.threshold(stretch, 200, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.threshold(stretch, threshold, 255, cv2.THRESH_BINARY)[1]

    # apply morphology open and close to smooth out shapes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    result = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

    # mask, start, stop = getBbox(img)
    mask = img > 0.01
    
    anomalyMask = mask * result
    anomalyMask = np.where(anomalyMask > 0, 1, 0)
    
    addImg = np.ones_like(img)
    scale = random.randint(0,scaleUpper)
    
    augImg = img * (1-anomalyMask) + addImg * anomalyMask * scale
    return augImg.astype(np.uint8), anomalyMask.astype(np.uint8), stretch.astype(np.uint8)
    
""" Colorjitter random shape"""

def colorJitterRandom_PIL(img_path, colorjitterScale=0):
    colorJitter_fn = transforms.ColorJitter(brightness = colorjitterScale,
                                                      contrast = colorjitterScale,
                                                      saturation = colorjitterScale,
                                                      hue = colorjitterScale)
    new_img, gt_mask = randomShape(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE))
    gt_mask = Image.fromarray(np.uint8(gt_mask * 255))
    
    # transfer img PIL to tensor
    img = Image.open(img_path)
    img = ImageOps.grayscale(img)
    img = img.resize((256, 256))
    
    # img_tensor = torch.reshape(img, [1,1,img.shape[0], img.shape[1]])
    # img_jitter = colorJitter_fn(img)

    # filter = ImageEnhance.Brightness(img)
    # new_image = img.filter(1.2)
    while colorjitterScale < 0.5:
        colorjitterScale = random.uniform(0,1)
        
    img_jitter = img
    img_jitter = img_jitter.point(lambda i: i*colorjitterScale)
    
    img_jitter = img_jitter.save('color_jitter.png')
    img = img.save('color_jitter_none.png')
    
    # combine the jitter_img with the raw img
    # new_img = Image.composite(ImageOps.invert(gt_mask), img)  + Image.composite(gt_mask, img_jitter)
    new_img = Image.composite(img, img_jitter, gt_mask)
    
    # return new_img.reshape([img.shape[0], img.shape[1]]), gt_mask.reshape([img.shape[0], img.shape[1]])
    return new_img.astype(np.uint8), gt_mask.astype(np.uint8)


def colorJitterRandom(img, args, colorRange = 100, minscale = 50, colorjitterScale=0, threshold=200, number_iterations=1, control_texture=False):
    colorJitter_fn = transforms.ColorJitter(brightness = colorjitterScale,
                                                      contrast = colorjitterScale,
                                                      saturation = colorjitterScale,
                                                      hue = colorjitterScale)
    img = cv2.resize(img, [256, 256])
    tot_gt_mask = np.zeros_like(img)
    for i in range(number_iterations):
        new_img, gt_mask, randomMap = randomShape(img, threshold=threshold)
        
        
        if args.rejection:
            while gt_mask.sum() == 0:
                new_img, gt_mask, randomMap = randomShape(img, threshold=threshold)
        
        tot_gt_mask = tot_gt_mask + gt_mask
        
    tot_gt_mask = np.where(tot_gt_mask > 0, 1, 0)
    
    # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    while abs(colorjitterScale) < minscale:        # from 50 to 5
        colorjitterScale = random.uniform(-colorRange,colorRange)
    
    
    # control the texture
    if not control_texture:
        color_mask = np.ones_like(img) * colorjitterScale
        img_jitter = img + color_mask
        img_jitter = img_jitter.clip(0, 255)
        new_img = img * (1-tot_gt_mask) + img_jitter * tot_gt_mask
    else:
        texture_index = random.randint(1,2)
        if texture_index == 1:
            color_mask = np.ones_like(img) * colorjitterScale
            img_jitter = img + color_mask
            img_jitter = img_jitter.clip(0, 255)
            new_img = img * (1-tot_gt_mask) + img_jitter * tot_gt_mask
            # cv2.imwrite('new_image_1.png', new_img)
            
            
        elif texture_index == 2:
            liver_average = img[np.nonzero(img)].mean()
            # randomMap = randomMap - randomMap.mean()
            # if randomMap.max() > 
            # randomMap = randomMap.max() * colorjitterScale
            # color_mask = randomMap * colorjitterScale
            # img_jitter = img + randomMap
            # img_jitter = randomMap/randomMap.max() * colorRange
            img_jitter = randomMap/randomMap.max() * liver_average
            # img_jitter = randomMap
            img_jitter = img_jitter.clip(0, 255)
            new_img = img * (1-tot_gt_mask) + img_jitter * tot_gt_mask
            # cv2.imwrite('new_image_2.png', new_img)

    
    # cv2.imwrite('new_image.png', new_img)
    # cv2.imwrite('tot_gt.png', tot_gt_mask*255)
        

    # return new_img.reshape([img.shape[0], img.shape[1]]), gt_mask.reshape([img.shape[0], img.shape[1]])
    return new_img.astype(np.uint8), tot_gt_mask.astype(np.uint8)


def colorJitterRandom_Mask(img, colorRange = 150, colorjitterScale=0):
    colorJitter_fn = transforms.ColorJitter(brightness = colorjitterScale,
                                                      contrast = colorjitterScale,
                                                      saturation = colorjitterScale,
                                                      hue = colorjitterScale)
    new_img, gt_mask = randomShape(img)
    # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, [256, 256])

    while abs(colorjitterScale) < 50:
        colorjitterScale = random.uniform(-colorRange,colorRange)
        
    color_mask = np.ones_like(img) * colorjitterScale
    img_jitter = img + color_mask
    img_jitter = img_jitter.clip(0, 255)
    
    # img_jitter = img_jitter.save('color_jitter.png')
    # img = img.save('color_jitter_none.png')
    # cv2.imwrite('color_jitter.png', img_jitter)
    # cv2.imwrite('color_jitter_none.png', img)
    
    # combine the jitter_img with the raw img
    # new_img = Image.composite(ImageOps.invert(gt_mask), img)  + Image.composite(gt_mask, img_jitter)
    # new_img = Image.composite(img, img_jitter, gt_mask)
    new_img = img * (1-gt_mask) + img_jitter * gt_mask
    
    cv2.imwrite('new_img.png', new_img)
    
    # return new_img.reshape([img.shape[0], img.shape[1]]), gt_mask.reshape([img.shape[0], img.shape[1]])
    return new_img.astype(np.uint8), gt_mask.astype(np.uint8)
    

            
            
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--rejection', default=True, action='store')
    args = parser.parse_args()
    
    # root = '/home/zhaoxiang/DRAEM_Denosing/sample_liver_images'
    # img_names = os.listdir(os.path.join(root, 'raw'))
    root = '/home/zhaoxiang/dataset/hist_DIY/train/good'
    dst_root = '/home/zhaoxiang/dataset/hist_DIY_pseudo/train_aug'
    dst_root_gt = '/home/zhaoxiang/dataset/hist_DIY_pseudo/train_label'
    dst_root_raw = '/home/zhaoxiang/dataset/hist_DIY_pseudo/train_raw'
    img_names = os.listdir(root)
    img_names.sort()
    
    
    for i in range(5):
        for img_name in img_names:
            
            img_path = os.path.join(root, img_name)
            img = Image.open(img_path)
            img = ImageOps.grayscale(img)
            img = img.resize([256, 256])

            """ Sample level augmentation"""
            img_numpy = np.array(img)
            
        
            # big light anomalies
            colorJitter_img, colorJitter_gt = colorJitterRandom(img_numpy, args, colorRange=100, threshold=200)
            while(colorJitter_gt.sum() == 0):
                colorJitter_img, colorJitter_gt = colorJitterRandom(img_numpy, colorRange=100, threshold=200)
                
                # # small anomalies
                # colorJitter_img, colorJitter_gt_2 = colorJitterRandom(img_numpy, minscale=80, colorRange=100, threshold=230)
                # colorJitter_gt = np.where((colorJitter_gt + colorJitter_gt_2) > 0, 1, 0)
                
            img_save_path = os.path.join(dst_root, img_name.replace('.png', '_{}.png'.format(i)))
            gt_save_path = os.path.join(dst_root_gt, img_name.replace('.png', '_{}.png'.format(i)))
            raw_save_path = os.path.join(dst_root_raw, img_name.replace('.png', '_{}.png'.format(i)))
            
            cv2.imwrite(img_save_path, colorJitter_img)
            cv2.imwrite(raw_save_path, img_numpy)   
            cv2.imwrite(gt_save_path, colorJitter_gt*255)
        
        
        