import os
from turtle import color
import numpy as np
import torch
import cv2
import random
from PIL import Image, ImageOps
from skimage.draw import random_shapes

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
    return augImg.astype(np.uint8), anomalyMask.astype(np.uint8)
    
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


def colorJitterRandom(img, colorRange = 150, colorjitterScale=0, threshold=200):
    colorJitter_fn = transforms.ColorJitter(brightness = colorjitterScale,
                                                      contrast = colorjitterScale,
                                                      saturation = colorjitterScale,
                                                      hue = colorjitterScale)
    new_img, gt_mask = randomShape(img, threshold=threshold)
    # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, [256, 256])

    while abs(colorjitterScale) < 50:
        colorjitterScale = random.uniform(-colorRange,colorRange)
        
    color_mask = np.ones_like(img) * colorjitterScale
    img_jitter = img + color_mask
    img_jitter = img_jitter.clip(0, 255)
    
    # img_jitter = img_jitter.save('color_jitter.png')
    # img = img.save('color_jitter_none.png')
    cv2.imwrite('color_jitter.png', img_jitter)
    cv2.imwrite('color_jitter_none.png', img)
    
    # combine the jitter_img with the raw img
    # new_img = Image.composite(ImageOps.invert(gt_mask), img)  + Image.composite(gt_mask, img_jitter)
    # new_img = Image.composite(img, img_jitter, gt_mask)
    new_img = img * (1-gt_mask) + img_jitter * gt_mask
    # return new_img.reshape([img.shape[0], img.shape[1]]), gt_mask.reshape([img.shape[0], img.shape[1]])
    return new_img.astype(np.uint8), gt_mask.astype(np.uint8)


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
    
    # method = blackStrip
    method = distortion
    method = randomShape
    method = colorJitterRandom
    
    # method = cp
    cutpaste = CutPasteUnion(transform=None)
    
    img_dir = '/home/zhaoxiang/mood_challenge/Sample_images/raw'
    save_dir = '/home/zhaoxiang/mood_challenge/Sample_images/{}'.format(method.__name__)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
        
    files = os.listdir(img_dir)
    files.sort()
    for f in files:
        img_path = os.path.join(img_dir, f)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        
        # mask = np.where(img>0, 255, 0)
        # mask = mask.astype(np.uint8)
        # kernel = np.ones((5, 5), np.uint8)
        # mask_dil = cv2.dilate(mask, kernel, iterations=1)
        # cv2.imwrite('mask.png',mask_dil)
        
        if img.max() == 0:
            continue
        """cutpaste"""
        # org, cut_img = cp(img_path)
        # cut_img = cut_img.save(os.path.join(save_dir, f))
        
        
        """other augmentations"""
        try:
            new_img, gt_mask = method(img)
            
            
            cv2.imwrite(os.path.join(save_dir, f), new_img)
            cv2.imwrite(os.path.join(save_dir, f.replace('.png', '_gt.png')), gt_mask*255)
        except:
            # new_img = method(img)
            # cv2.imwrite(os.path.join(save_dir, f), new_img)
            
            new_img, gt_mask = method(img_path)
            # save_image(new_img, os.path.join(save_dir, f))
            # save_image(gt_mask*255, save_dir, f.replace('.png', '_gt.png'))
            # new_img = new_img.save(os.path.join(save_dir, f))
            # gt_mask = gt_mask.save(os.path.join(save_dir, f.replace('.png', '_gt.png')))
            cv2.imwrite(os.path.join(save_dir, f), new_img)
            cv2.imwrite(os.path.join(save_dir, f.replace('.png', '_gt.png')), gt_mask*255)