#Import for Signal Intensity calculations
import numpy as np
import torch
#import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import math
#from torch.utils.data import Dataset, DataLoader

#The functions in these cells allow you to calculate the structural similarity indices between 2 tensors. We use this
#to keep track of SSIM between predicted and ground truth T1rho maps, but you could in theory use this in other ways
#too
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

def psnr(img1,img2):
    mse = np.mean(np.square(img1-img2))
    maxval = np.amax()
    psnr = 10
    
#Calculates the normalized root mean square error between predicted and target maps. There can be some pretty extreme
#outliers with T1rho values that are not physiologically realistic and associated with noise; that's why we cap the
#maximal predicted T1rho values at max_map
def calculate_NMSE(x,y):
    
    err  = np.sum((x-y)**2)
    denom = np.sum(x**2)
    if denom > 0:
        nmse = 100*err/denom
    else:
        nmse = 0
    return nmse


def calculate_peak_snr(x_pred,y_target,binary_mask):
    
    x_seg = x_pred[binary_mask==1]
    y_seg = y_target[binary_mask==1]
    
    # Calculate psnr
    if len(x_seg) > 0 and len(y_seg) > 0:
        mse  = np.mean((x_seg-y_seg)**2)
        max_target = np.amax(y_seg)
        
        if mse > 0 and max_target > 0:
            psnr = 20*math.log10(max_target) - 10*math.log10(mse)
        else:
             psnr = 0
    else:
        psnr = 0
        
    return psnr

def calculate_stats(x,binary_mask):
#    x[x > max_map_calc_only] = max_map_calc_only
#    x[x < 0]       = 0

    x_seg = x[np.where(binary_mask==1)]
    
    # Calculate similarity
    if len(x_seg) > 0:
        mean_value = np.mean(x_seg)
        stdev_value = np.std(x_seg)
        median_value = np.median(x_seg)
    else:
        mean_value = 0
        stdev_value = 0
        median_value = 0
        
    return mean_value, stdev_value, median_value

def calculate_corr(x,y,binary_mask):
    
    x_seg = x[binary_mask==1]
    y_seg = y[binary_mask==1]
    
    # Calculate similarity
    if len(x_seg) > 0 and len(y_seg) > 0:
        corr_value = np.corrcoef(x_seg,y_seg)[0,1]
    else:
        corr_value = 0
        
    return corr_value

