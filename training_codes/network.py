import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchvision import models, transforms
from math import pi
import numpy as np
import os
import csv
import math
import random
from tqdm import tqdm
from io import StringIO

from PIL import Image
from argparse import ArgumentParser
import torch.nn as nn

from torch.optim import SGD, Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor, ToPILImage
from torch.autograd import Function 
import scipy
from scipy import io
#from modules.layers import ConvOffset2D

import math 
import numpy as np
import lpips

import numpy as np
import torch
from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import conv2d_resample
from torch_utils.ops import upfirdn2d
from torch_utils.ops import bias_act
from torch_utils.ops import fma

#outs = tanh(ylogit), outc = tanh(xlogit)) with a loss function 0.5((sin(pred) - outs)^2 + (cos(pred) - outc)^2
from torch.autograd.function import once_differentiable
import torch.nn.functional as F

import torch

from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Function
from torch.nn.utils import weight_norm
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

from math import sqrt

import random

import cv2

import math
import random
import functools
import operator
import pickle

import torch
from torch import nn, einsum
from torch.nn import functional as F
from torch.autograd import Function

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from functools import partial, wraps
from packaging import version

from collections import namedtuple

import torch
from torch.optim.optimizer import Optimizer, required

from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.nn import Parameter
from typing import List
from torchvision.ops import StochasticDepth
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

# Losses

class ISOSharpnessLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        F_np = np.array([[0, 1, 1, 2, 2, 2, 1, 1, 0],
                      [1, 2, 4, 5, 5, 5, 4, 2, 1],
                      [1, 4, 5, 3, 0, 3, 5, 4, 1],
                      [2, 5, 3, -12, -24, -12, 3, 5, 2],
                      [2, 5, 0, -24, -40, -24, 0, 5, 2],
                      [2, 5, 3, -12, -24, -12, 3, 5, 2],
                      [1, 4, 5, 3, 0, 3, 5, 4, 1],
                      [1, 2, 4, 5, 5, 5, 4, 2, 1],
                      [0, 1, 1, 2, 2, 2, 1, 1, 0]])
        self.F = torch.from_numpy(F_np).float().unsqueeze(0).unsqueeze(0).to(device)
    def forward(self, image):
        IF = torch.nn.functional.conv2d(image, self.F)
        POWER = torch.mean(torch.square(IF), dim=(1,2,3))
        C = 1800000
        SHARPNESS = 100 * (torch.square(POWER)/(torch.square(POWER)+C**2))
        return SHARPNESS
        
        
class SIFLayerMask(nn.Module):
    def __init__(self, polar_height, polar_width, filter_mat1, filter_mat2, device, channels = 1):
        super().__init__()
        with torch.no_grad():
            self.device = device
            self.polar_height = polar_height
            self.polar_width = polar_width
            self.channels = channels
            
            self.filter1_size = filter_mat1.shape[0]
            self.num_filters1 = filter_mat1.shape[2]
            self.filter1 = torch.FloatTensor(filter_mat1).to(self.device)
            self.filter_mat1 = filter_mat1
            self.filter1 = torch.moveaxis(self.filter1.unsqueeze(0), 3, 0).detach().requires_grad_(False)
            
            self.filter2_size = filter_mat2.shape[0]
            self.num_filters2 = filter_mat2.shape[2]
            self.filter2 = torch.FloatTensor(filter_mat2).to(self.device)
            self.filter_mat2 = filter_mat2
            self.filter2 = torch.moveaxis(self.filter2.unsqueeze(0), 3, 0).detach().requires_grad_(False)
            
            self.torch_filters = []
            
            self.torch_filters.append(self.filter1)
            self.torch_filters.append(self.filter2)
            
            self.max_filter_size = max(self.filter1.shape[2], self.filter2.shape[2])
            
            print('Max Filter Size:', self.max_filter_size)
            self.n_filters = len(self.torch_filters)
        
    def grid_sample(self, input, grid, interp_mode='bilinear'):

        # grid: [-1, 1]
        N, C, H, W = input.shape
        gridx = grid[:, :, :, 0]
        gridy = grid[:, :, :, 1]
        gridx = ((gridx + 1) / 2 * W - 0.5) / (W - 1) * 2 - 1
        gridy = ((gridy + 1) / 2 * H - 0.5) / (H - 1) * 2 - 1
        newgrid = torch.stack([gridx, gridy], dim=-1).requires_grad_(False)
        return torch.nn.functional.grid_sample(input, newgrid, mode=interp_mode, padding_mode='zeros')

    def cartToPol(self, image, mask, pupil_xyr, iris_xyr):
        with torch.no_grad():
            batch_size = image.shape[0]
            width = image.shape[3]
            height = image.shape[2]
    
            polar_height = self.polar_height
            polar_width = self.polar_width
    
            pupil_xyr = pupil_xyr.clone().detach().requires_grad_(False)
            iris_xyr = iris_xyr.clone().detach().requires_grad_(False)
            
            theta = (2*pi*torch.linspace(0,polar_width-1,polar_width)/polar_width).requires_grad_(False).to(self.device)
    
            pxCirclePoints = pupil_xyr[:, 0].reshape(-1, 1) + pupil_xyr[:, 2].reshape(-1, 1) @ torch.cos(theta).reshape(1, polar_width) #b x 512
            pyCirclePoints = pupil_xyr[:, 1].reshape(-1, 1) + pupil_xyr[:, 2].reshape(-1, 1) @ torch.sin(theta).reshape(1, polar_width)  #b x 512
            
            ixCirclePoints = iris_xyr[:, 0].reshape(-1, 1) + iris_xyr[:, 2].reshape(-1, 1) @ torch.cos(theta).reshape(1, polar_width)  #b x 512
            iyCirclePoints = iris_xyr[:, 1].reshape(-1, 1) + iris_xyr[:, 2].reshape(-1, 1) @ torch.sin(theta).reshape(1, polar_width)  #b x 512
    
            radius = (torch.linspace(1,polar_height,polar_height)/polar_height).reshape(-1, 1).requires_grad_(False).to(self.device)  #64 x 1
            
            pxCoords = torch.matmul((1-radius), pxCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
            pyCoords = torch.matmul((1-radius), pyCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
            
            ixCoords = torch.matmul(radius, ixCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
            iyCoords = torch.matmul(radius, iyCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
    
            x = (pxCoords + ixCoords).float()
            x_norm = ((x-1)/(width-1))*2 - 1 #b x 64 x 512
    
            y = (pyCoords + iyCoords).float()
            y_norm = ((y-1)/(height-1))*2 - 1  #b x 64 x 512
    
            grid_sample_mat = torch.cat([x_norm.unsqueeze(-1), y_norm.unsqueeze(-1)], dim=-1).detach().requires_grad_(False)
            
            if mask is not None:
                if torch.is_tensor(mask):
                    mask_t = mask.clone().detach().to(self.device)
                else: 
                    mask_t = torch.tensor(mask).float().to(self.device)
                mask_polar = self.grid_sample(mask_t, grid_sample_mat, interp_mode='nearest')
                mask_polar = (mask_polar - mask_polar.min()) / mask_polar.max()
                mask_polar = torch.where(mask_polar > 0.5, 1.0, 0.0)
            else:
                mask_polar = None
                
        image_polar = self.grid_sample(image, grid_sample_mat, interp_mode='bilinear')
        return image_polar, mask_polar
    
    def cartToPolIrisCenter(self, image, mask, pupil_xyr, iris_xyr):
        with torch.no_grad():
            batch_size = image.shape[0]
            width = image.shape[3]
            height = image.shape[2]
    
            polar_height = self.polar_height
            polar_width = self.polar_width
    
            pupil_xyr = pupil_xyr.clone().detach().requires_grad_(False)
            iris_xyr = iris_xyr.clone().detach().requires_grad_(False)
            
            theta = (2*pi*torch.linspace(0,polar_width-1,polar_width)/polar_width).requires_grad_(False)
    
            pxCirclePoints = iris_xyr[:, 0].reshape(-1, 1) + pupil_xyr[:, 2].reshape(-1, 1) @ torch.cos(theta).reshape(1, polar_width) #b x 512
            pyCirclePoints = iris_xyr[:, 1].reshape(-1, 1) + pupil_xyr[:, 2].reshape(-1, 1) @ torch.sin(theta).reshape(1, polar_width)  #b x 512
            
            ixCirclePoints = iris_xyr[:, 0].reshape(-1, 1) + iris_xyr[:, 2].reshape(-1, 1) @ torch.cos(theta).reshape(1, polar_width)  #b x 512
            iyCirclePoints = iris_xyr[:, 1].reshape(-1, 1) + iris_xyr[:, 2].reshape(-1, 1) @ torch.sin(theta).reshape(1, polar_width)  #b x 512
    
            radius = (torch.linspace(1,polar_height,polar_height)/polar_height).reshape(-1, 1).requires_grad_(False) #64 x 1
            
            pxCoords = torch.matmul((1-radius), pxCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
            pyCoords = torch.matmul((1-radius), pyCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
            
            ixCoords = torch.matmul(radius, ixCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
            iyCoords = torch.matmul(radius, iyCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
    
            x = (pxCoords + ixCoords).float()
            x_norm = ((x-1)/(width-1))*2 - 1 #b x 64 x 512
    
            y = (pyCoords + iyCoords).float()
            y_norm = ((y-1)/(height-1))*2 - 1  #b x 64 x 512
    
            grid_sample_mat = torch.cat([x_norm.unsqueeze(-1), y_norm.unsqueeze(-1)], dim=-1).detach().requires_grad_(False)
            
            if mask is not None:
                if torch.is_tensor(mask):
                    mask_t = mask.clone().detach()
                else: 
                    mask_t = torch.tensor(mask).float()
                mask_polar = self.grid_sample(mask_t, grid_sample_mat, interp_mode='nearest')
                mask_polar = (mask_polar - mask_polar.min()) / mask_polar.max()
                mask_polar = torch.where(mask_polar > 0.5, 1.0, 0.0)
            else:
                mask_polar = None
                
        image_polar = self.grid_sample(image, grid_sample_mat, interp_mode='bilinear')
        return image_polar, mask_polar
    
    def getCodes(self, image_polar):
        codes_list = []
        for filter in self.torch_filters:
            r1 = int(filter.shape[2]/2)
            r2 = int(filter.shape[3]/2)
            imgWrap = nn.functional.pad(image_polar, (0, 0, r1, r1), mode='replicate')
            imgWrap = nn.functional.pad(imgWrap, (r2, r2, 0, 0), mode='circular')
            code = nn.functional.conv2d(imgWrap, filter, stride=1, padding='valid')
            codes_list.append(code)
        codes = torch.cat(codes_list, dim=1)
        return codes
    
    def getCodesCPU(self, image_polar):
        image_polar = image_polar.cpu()
        codes_list = []
        for filter in self.torch_filters:
            r1 = int(filter.shape[2]/2)
            r2 = int(filter.shape[3]/2)
            imgWrap = nn.functional.pad(image_polar, (0, 0, r1, r1), mode='replicate')
            imgWrap = nn.functional.pad(imgWrap, (r2, r2, 0, 0), mode='circular')
            code = nn.functional.conv2d(imgWrap, filter.cpu(), stride=1, padding='valid')
            codes_list.append(code)
        codes = torch.cat(codes_list, dim=1)
        return codes

    def forward(self, image, pupil_xyr, iris_xyr, mask=None):
        image_polar, mask_polar = self.cartToPol(image, mask, pupil_xyr, iris_xyr)
        codes = self.getCodes(image_polar)
        return codes, image_polar, mask_polar

class SIFLayerMaskOSIRIS(nn.Module):
    def __init__(self, polar_height, polar_width, filter_mat1, filter_mat2, osiris_filters, device, channels = 1):
        super().__init__()
        with torch.no_grad():
            self.device = device
            self.polar_height = polar_height
            self.polar_width = polar_width
            self.channels = channels
            
            self.filter1_size = filter_mat1.shape[0]
            self.num_filters1 = filter_mat1.shape[2]
            self.filter1 = torch.FloatTensor(filter_mat1).to(self.device)
            self.filter_mat1 = filter_mat1
            self.filter1 = torch.moveaxis(self.filter1.unsqueeze(0), 3, 0).detach().requires_grad_(False)
            
            self.filter2_size = filter_mat2.shape[0]
            self.num_filters2 = filter_mat2.shape[2]
            self.filter2 = torch.FloatTensor(filter_mat2).to(self.device)
            self.filter_mat2 = filter_mat2
            self.filter2 = torch.moveaxis(self.filter2.unsqueeze(0), 3, 0).detach().requires_grad_(False)
            
            self.np_filters = []
            with open(osiris_filters, 'r') as osirisFile:
                string_inp = ''
                for line in osirisFile:
                    if not line.strip() == 'end':
                        string_inp += line.strip() + '\n'
                    else:
                        c = StringIO(string_inp)
                        self.np_filters.append(np.loadtxt(c))
                        string_inp = ''
            
            self.torch_filters = []
            
            self.torch_filters.append(self.filter1)
            self.torch_filters.append(self.filter2)
            
            self.max_filter_size = max(self.filter1.shape[2], self.filter2.shape[2])
            
            for np_filter in self.np_filters:
                torch_filter = torch.FloatTensor(np_filter).to(self.device)
                torch_filter = torch_filter.unsqueeze(0).unsqueeze(0).detach().requires_grad_(False)
                self.max_filter_size = max(self.max_filter_size, torch_filter.shape[2])
                #print(torch_filter.shape)
                self.torch_filters.append(torch_filter)
            
            print('Max Filter Size:', self.max_filter_size)
            self.n_filters = len(self.torch_filters)
        
    def grid_sample(self, input, grid, interp_mode='bilinear'):

        # grid: [-1, 1]
        N, C, H, W = input.shape
        gridx = grid[:, :, :, 0]
        gridy = grid[:, :, :, 1]
        gridx = ((gridx + 1) / 2 * W - 0.5) / (W - 1) * 2 - 1
        gridy = ((gridy + 1) / 2 * H - 0.5) / (H - 1) * 2 - 1
        newgrid = torch.stack([gridx, gridy], dim=-1).requires_grad_(False)
        return torch.nn.functional.grid_sample(input, newgrid, mode=interp_mode, padding_mode='zeros')

    def cartToPol(self, image, mask, pupil_xyr, iris_xyr):
        with torch.no_grad():
            batch_size = image.shape[0]
            width = image.shape[3]
            height = image.shape[2]
    
            polar_height = self.polar_height
            polar_width = self.polar_width
    
            pupil_xyr = pupil_xyr.clone().detach().requires_grad_(False)
            iris_xyr = iris_xyr.clone().detach().requires_grad_(False)
            
            theta = (2*pi*torch.linspace(0,polar_width-1,polar_width)/polar_width).requires_grad_(False).to(self.device)
    
            pxCirclePoints = pupil_xyr[:, 0].reshape(-1, 1) + pupil_xyr[:, 2].reshape(-1, 1) @ torch.cos(theta).reshape(1, polar_width) #b x 512
            pyCirclePoints = pupil_xyr[:, 1].reshape(-1, 1) + pupil_xyr[:, 2].reshape(-1, 1) @ torch.sin(theta).reshape(1, polar_width)  #b x 512
            
            ixCirclePoints = iris_xyr[:, 0].reshape(-1, 1) + iris_xyr[:, 2].reshape(-1, 1) @ torch.cos(theta).reshape(1, polar_width)  #b x 512
            iyCirclePoints = iris_xyr[:, 1].reshape(-1, 1) + iris_xyr[:, 2].reshape(-1, 1) @ torch.sin(theta).reshape(1, polar_width)  #b x 512
    
            radius = (torch.linspace(1,polar_height,polar_height)/polar_height).reshape(-1, 1).requires_grad_(False).to(self.device)  #64 x 1
            
            pxCoords = torch.matmul((1-radius), pxCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
            pyCoords = torch.matmul((1-radius), pyCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
            
            ixCoords = torch.matmul(radius, ixCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
            iyCoords = torch.matmul(radius, iyCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
    
            x = (pxCoords + ixCoords).float()
            x_norm = ((x-1)/(width-1))*2 - 1 #b x 64 x 512
    
            y = (pyCoords + iyCoords).float()
            y_norm = ((y-1)/(height-1))*2 - 1  #b x 64 x 512
    
            grid_sample_mat = torch.cat([x_norm.unsqueeze(-1), y_norm.unsqueeze(-1)], dim=-1).detach().requires_grad_(False)
            
            if mask is not None:
                if torch.is_tensor(mask):
                    mask_t = mask.clone().detach().to(self.device)
                else: 
                    mask_t = torch.tensor(mask).float().to(self.device)
                mask_polar = self.grid_sample(mask_t, grid_sample_mat, interp_mode='nearest')
                mask_polar = (mask_polar - mask_polar.min()) / mask_polar.max()
                mask_polar = torch.where(mask_polar > 0.5, 1.0, 0.0)
            else:
                mask_polar = None
                
        image_polar = self.grid_sample(image, grid_sample_mat, interp_mode='bilinear')
        return image_polar, mask_polar
    
    def cartToPolIrisCenter(self, image, mask, pupil_xyr, iris_xyr):
        with torch.no_grad():
            batch_size = image.shape[0]
            width = image.shape[3]
            height = image.shape[2]
    
            polar_height = self.polar_height
            polar_width = self.polar_width
    
            pupil_xyr = pupil_xyr.clone().detach().requires_grad_(False)
            iris_xyr = iris_xyr.clone().detach().requires_grad_(False)
            
            theta = (2*pi*torch.linspace(0,polar_width-1,polar_width)/polar_width).requires_grad_(False)
    
            pxCirclePoints = iris_xyr[:, 0].reshape(-1, 1) + pupil_xyr[:, 2].reshape(-1, 1) @ torch.cos(theta).reshape(1, polar_width) #b x 512
            pyCirclePoints = iris_xyr[:, 1].reshape(-1, 1) + pupil_xyr[:, 2].reshape(-1, 1) @ torch.sin(theta).reshape(1, polar_width)  #b x 512
            
            ixCirclePoints = iris_xyr[:, 0].reshape(-1, 1) + iris_xyr[:, 2].reshape(-1, 1) @ torch.cos(theta).reshape(1, polar_width)  #b x 512
            iyCirclePoints = iris_xyr[:, 1].reshape(-1, 1) + iris_xyr[:, 2].reshape(-1, 1) @ torch.sin(theta).reshape(1, polar_width)  #b x 512
    
            radius = (torch.linspace(1,polar_height,polar_height)/polar_height).reshape(-1, 1).requires_grad_(False) #64 x 1
            
            pxCoords = torch.matmul((1-radius), pxCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
            pyCoords = torch.matmul((1-radius), pyCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
            
            ixCoords = torch.matmul(radius, ixCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
            iyCoords = torch.matmul(radius, iyCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
    
            x = (pxCoords + ixCoords).float()
            x_norm = ((x-1)/(width-1))*2 - 1 #b x 64 x 512
    
            y = (pyCoords + iyCoords).float()
            y_norm = ((y-1)/(height-1))*2 - 1  #b x 64 x 512
    
            grid_sample_mat = torch.cat([x_norm.unsqueeze(-1), y_norm.unsqueeze(-1)], dim=-1).detach().requires_grad_(False)
            
            if mask is not None:
                if torch.is_tensor(mask):
                    mask_t = mask.clone().detach()
                else: 
                    mask_t = torch.tensor(mask).float()
                mask_polar = self.grid_sample(mask_t, grid_sample_mat, interp_mode='nearest')
                mask_polar = (mask_polar - mask_polar.min()) / mask_polar.max()
                mask_polar = torch.where(mask_polar > 0.5, 1.0, 0.0)
            else:
                mask_polar = None
                
        image_polar = self.grid_sample(image, grid_sample_mat, interp_mode='bilinear')
        return image_polar, mask_polar
    
    def getCodes(self, image_polar):
        codes_list = []
        for filter in self.torch_filters:
            r1 = int(filter.shape[2]/2)
            r2 = int(filter.shape[3]/2)
            imgWrap = nn.functional.pad(image_polar, (0, 0, r1, r1), mode='replicate')
            imgWrap = nn.functional.pad(imgWrap, (r2, r2, 0, 0), mode='circular')
            code = nn.functional.conv2d(imgWrap, filter, stride=1, padding='valid')
            codes_list.append(code)
        codes = torch.cat(codes_list, dim=1)
        return codes
    
    def getCodesCPU(self, image_polar):
        image_polar = image_polar.cpu()
        codes_list = []
        for filter in self.torch_filters:
            r1 = int(filter.shape[2]/2)
            r2 = int(filter.shape[3]/2)
            imgWrap = nn.functional.pad(image_polar, (0, 0, r1, r1), mode='replicate')
            imgWrap = nn.functional.pad(imgWrap, (r2, r2, 0, 0), mode='circular')
            code = nn.functional.conv2d(imgWrap, filter.cpu(), stride=1, padding='valid')
            codes_list.append(code)
        codes = torch.cat(codes_list, dim=1)
        return codes

    def forward(self, image, pupil_xyr, iris_xyr, mask=None):
        image_polar, mask_polar = self.cartToPol(image, mask, pupil_xyr, iris_xyr)
        codes = self.getCodes(image_polar)
        return codes, image_polar, mask_polar

class SIFLayerMaskModC2P(nn.Module):
    def __init__(self, polar_height, polar_width, filter_mat1, filter_mat2, osiris_filters, device, channels = 1):
        super().__init__()
        self.device = device
        self.polar_height = polar_height
        self.polar_width = polar_width
        self.channels = channels
        
        self.filter1_size = filter_mat1.shape[0]
        self.num_filters1 = filter_mat1.shape[2]
        self.filter1 = torch.FloatTensor(filter_mat1).to(self.device)
        self.filter_mat1 = filter_mat1
        self.filter1 = torch.moveaxis(self.filter1.unsqueeze(0), 3, 0)
        self.filter1 = torch.flip(self.filter1, dims=[0]).requires_grad_(False)
        #print(self.filter1.shape)
        
        self.filter2_size = filter_mat2.shape[0]
        self.num_filters2 = filter_mat2.shape[2]
        self.filter2 = torch.FloatTensor(filter_mat2).to(self.device)
        self.filter_mat2 = filter_mat2
        self.filter2 = torch.moveaxis(self.filter2.unsqueeze(0), 3, 0)
        self.filter2 = torch.flip(self.filter2, dims=[0]).requires_grad_(False)
        #print(self.filter2.shape)
        
        self.np_filters = []
        with open(osiris_filters, 'r') as osirisFile:
            string_inp = ''
            for line in osirisFile:
                if not line.strip() == 'end':
                    string_inp += line.strip() + '\n'
                else:
                    c = StringIO(string_inp)
                    self.np_filters.append(np.loadtxt(c))
                    string_inp = ''
        
        self.torch_filters = []
        
        self.torch_filters.append(self.filter1)
        self.torch_filters.append(self.filter2)
        
        self.max_filter_size = max(self.filter1.shape[2], self.filter2.shape[2])
        
        for np_filter in self.np_filters:
            torch_filter = torch.FloatTensor(np_filter).to(self.device)
            torch_filter = torch_filter.unsqueeze(0).unsqueeze(0)
            self.max_filter_size = max(self.max_filter_size, torch_filter.shape[2])
            #print(torch_filter.shape)
            self.torch_filters.append(torch_filter)
        
        print('Max Filter Size:', self.max_filter_size)
        self.n_filters = len(self.torch_filters)
        
    def mod_grid_sample(self, input, grid, backprop_grid, interp_mode='bicubic'):
        # grid: [-1, 1]
        N, C, H, W = input.shape
        gridx = grid[:, :, :, 0]
        gridy = grid[:, :, :, 1]
        gridx = ((gridx + 1) / 2 * W - 0.5) / (W - 1) * 2 - 1
        gridy = ((gridy + 1) / 2 * H - 0.5) / (H - 1) * 2 - 1
        newgrid = torch.stack([gridx, gridy], dim=-1).requires_grad_(False)
        #print(newgrid.shape, backprop_grid.shape)
        return torch.nn.functional.grid_sample(input, newgrid + backprop_grid, mode=interp_mode, padding_mode='zeros', align_corners=True)
    
    def grid_sample(self, input, grid, interp_mode='bicubic'):
        # grid: [-1, 1]
        N, C, H, W = input.shape
        gridx = grid[:, :, :, 0]
        gridy = grid[:, :, :, 1]
        gridx = ((gridx + 1) / 2 * W - 0.5) / (W - 1) * 2 - 1
        gridy = ((gridy + 1) / 2 * H - 0.5) / (H - 1) * 2 - 1
        newgrid = torch.stack([gridx, gridy], dim=-1).requires_grad_(False)
        return torch.nn.functional.grid_sample(input, newgrid, mode=interp_mode, padding_mode='zeros', align_corners=True)

    def cartToPol_bpgrid(self, image, mask, pupil_xyr, iris_xyr, backprop_grid):
        
        image_scaled = (image * 255).requires_grad_(True)

        batch_size = image_scaled.shape[0]
        width = image_scaled.shape[3]
        height = image_scaled.shape[2]

        polar_height = self.polar_height
        polar_width = self.polar_width

        pupil_xyr = pupil_xyr.clone().detach().requires_grad_(False)
        iris_xyr = iris_xyr.clone().detach().requires_grad_(False)
        
        theta = (2*pi*torch.linspace(0,polar_width-1,polar_width)/polar_width).requires_grad_(False).to(self.device)

        pxCirclePoints = pupil_xyr[:, 0].reshape(-1, 1) + pupil_xyr[:, 2].reshape(-1, 1) @ torch.cos(theta).reshape(1, polar_width) #b x 512
        pyCirclePoints = pupil_xyr[:, 1].reshape(-1, 1) + pupil_xyr[:, 2].reshape(-1, 1) @ torch.sin(theta).reshape(1, polar_width)  #b x 512
        
        ixCirclePoints = iris_xyr[:, 0].reshape(-1, 1) + iris_xyr[:, 2].reshape(-1, 1) @ torch.cos(theta).reshape(1, polar_width)  #b x 512
        iyCirclePoints = iris_xyr[:, 1].reshape(-1, 1) + iris_xyr[:, 2].reshape(-1, 1) @ torch.sin(theta).reshape(1, polar_width)  #b x 512

        radius = (torch.linspace(1,polar_height,polar_height)/polar_height).reshape(-1, 1).requires_grad_(False).to(self.device)  #64 x 1
        
        pxCoords = torch.matmul((1-radius), pxCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
        pyCoords = torch.matmul((1-radius), pyCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
        
        ixCoords = torch.matmul(radius, ixCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
        iyCoords = torch.matmul(radius, iyCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512

        x = (pxCoords + ixCoords).float()
        x_norm = ((x-1)/(width-1))*2 - 1 #b x 64 x 512

        y = (pyCoords + iyCoords).float()
        y_norm = ((y-1)/(height-1))*2 - 1  #b x 64 x 512

        grid_sample_mat = torch.cat([x_norm.unsqueeze(-1), y_norm.unsqueeze(-1)], dim=-1)

        image_polar = self.mod_grid_sample(image_scaled, grid_sample_mat, backprop_grid)
        if mask is not None:
            if torch.is_tensor(mask):
                mask_t = mask.clone().detach().to(self.device)
            else: 
                mask_t = torch.tensor(mask).float().to(self.device)
            mask_t /= mask_t.max()
            one_mask = torch.ones(mask.shape).to(self.device)
            zero_mask = torch.zeros(mask.shape).to(self.device)
            mask_t = torch.where(mask_t > 0.5, one_mask, zero_mask)
            mask_polar = self.grid_sample(mask_t, grid_sample_mat)
        else:
            mask_polar = None

        return image_polar, mask_polar
        
    def cartToPol(self, image, mask, pupil_xyr, iris_xyr):
        
        image_scaled = (image * 255).requires_grad_(True)

        batch_size = image_scaled.shape[0]
        width = image_scaled.shape[3]
        height = image_scaled.shape[2]

        polar_height = self.polar_height
        polar_width = self.polar_width

        pupil_xyr = pupil_xyr.clone().detach().requires_grad_(False)
        iris_xyr = iris_xyr.clone().detach().requires_grad_(False)
        
        theta = (2*pi*torch.linspace(0,polar_width-1,polar_width)/polar_width).requires_grad_(False).to(self.device)

        pxCirclePoints = pupil_xyr[:, 0].reshape(-1, 1) + pupil_xyr[:, 2].reshape(-1, 1) @ torch.cos(theta).reshape(1, polar_width) #b x 512
        pyCirclePoints = pupil_xyr[:, 1].reshape(-1, 1) + pupil_xyr[:, 2].reshape(-1, 1) @ torch.sin(theta).reshape(1, polar_width)  #b x 512
        
        ixCirclePoints = iris_xyr[:, 0].reshape(-1, 1) + iris_xyr[:, 2].reshape(-1, 1) @ torch.cos(theta).reshape(1, polar_width)  #b x 512
        iyCirclePoints = iris_xyr[:, 1].reshape(-1, 1) + iris_xyr[:, 2].reshape(-1, 1) @ torch.sin(theta).reshape(1, polar_width)  #b x 512

        radius = (torch.linspace(1,polar_height,polar_height)/polar_height).reshape(-1, 1).requires_grad_(False).to(self.device)  #64 x 1
        
        pxCoords = torch.matmul((1-radius), pxCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
        pyCoords = torch.matmul((1-radius), pyCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
        
        ixCoords = torch.matmul(radius, ixCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
        iyCoords = torch.matmul(radius, iyCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512

        x = (pxCoords + ixCoords).float()
        x_norm = ((x-1)/(width-1))*2 - 1 #b x 64 x 512

        y = (pyCoords + iyCoords).float()
        y_norm = ((y-1)/(height-1))*2 - 1  #b x 64 x 512

        grid_sample_mat = torch.cat([x_norm.unsqueeze(-1), y_norm.unsqueeze(-1)], dim=-1)

        image_polar = self.grid_sample(image_scaled, grid_sample_mat)
        if mask is not None:
            if torch.is_tensor(mask):
                mask_t = mask.clone().detach().to(self.device)
            else: 
                mask_t = torch.tensor(mask).float().to(self.device)
            mask_t /= mask_t.max()
            one_mask = torch.ones(mask.shape).to(self.device)
            zero_mask = torch.zeros(mask.shape).to(self.device)
            mask_t = torch.where(mask_t > 0.5, one_mask, zero_mask)
            mask_polar = self.grid_sample(mask_t, grid_sample_mat)
            one_mask_polar = torch.ones(mask_polar.shape).to(self.device)
            zero_mask_polar = torch.zeros(mask_polar.shape).to(self.device)
            mask_polar = torch.where(mask_polar > 0.5, one_mask_polar, zero_mask_polar)
        else:
            mask_polar = None

        return image_polar, mask_polar
    
    def getCodes(self, image_polar):
        codes_list = []
        for filter in self.torch_filters:
            r1 = int(filter.shape[2]/2)
            r2 = int(filter.shape[3]/2)
            imgWrap = torch.zeros((image_polar.shape[0], image_polar.shape[1], r1*2+self.polar_height, r2*2+self.polar_width)).requires_grad_(False).to(self.device)
            imgWrap[:, :, :r1, :r2] += torch.clone(image_polar[:, :, -r1:, -r2:]).requires_grad_(True)
            imgWrap[:, :, :r1, r2:-r2] += torch.clone(image_polar[:, :, -r1:, :]).requires_grad_(True)
            imgWrap[:, :, :r1, -r2:] += torch.clone(image_polar[:, :, -r1:, :r2]).requires_grad_(True)

            imgWrap[:, :, r1:-r1, :r2] += torch.clone(image_polar[:, :, :, -r2:]).requires_grad_(True)
            imgWrap[:, :, r1:-r1, r2:-r2] += torch.clone(image_polar).requires_grad_(True)
            imgWrap[:, :, r1:-r1, -r2:] += torch.clone(image_polar[:, :, :, :r2]).requires_grad_(True)

            imgWrap[:, :, -r1:, :r2] += torch.clone(image_polar[:, :, :r1, -r2:]).requires_grad_(True)
            imgWrap[:, :, -r1:, r2:-r2] += torch.clone(image_polar[:, :, :r1, :]).requires_grad_(True)
            imgWrap[:, :, -r1:, -r2:] += torch.clone(image_polar[:, :, :r1, :r2]).requires_grad_(True)
            #imgWrap = torch.cat([image_polar, image_polar[:, :, :, :filter.shape[3]]], dim=3)
            code = nn.functional.conv2d(imgWrap, filter, stride=1, padding='valid')
            #print(code.shape)
            codes_list.append(code)

        codes = torch.cat(codes_list, dim=1)
        return codes

    def nonlinear_deform(self, backprop_grid, image, pupil_xyr, iris_xyr, mask=None):
        image_polar, mask_polar = self.cartToPol_bpgrid(image, mask, pupil_xyr, iris_xyr, backprop_grid)
        codes = self.getCodes(image_polar)
        return codes, image_polar, mask_polar
        
    def linear_deform(self, image, pupil_xyr, iris_xyr, mask=None):
        image_polar, mask_polar = self.cartToPol(image, mask, pupil_xyr, iris_xyr)
        codes = self.getCodes(image_polar)
        return codes, image_polar, mask_polar
    
    def forward(self, image, pupil_xyr, iris_xyr, deform_type, backprop_grid=None, mask=None):
        if deform_type == 'nonlinear':
            return self.nonlinear_deform(backprop_grid, image, pupil_xyr, iris_xyr, mask)
        elif deform_type == 'linear':
            return self.linear_deform(image, pupil_xyr, iris_xyr, mask)
                  
class SIFLayerPolar(nn.Module):
    def __init__(self, filter_mat, device):
        super().__init__()
        self.device = device
        self.filter_size = filter_mat.shape[0]
        self.num_filters = filter_mat.shape[2]
        self.filter = torch.FloatTensor(filter_mat).to(self.device).requires_grad_(False)
        self.filter_mat = filter_mat
        self.filter = torch.moveaxis(self.filter.unsqueeze(0), 3, 0)
        self.filter = torch.flip(self.filter, dims=[0]).requires_grad_(False)

    def getCodes(self, image_polar):
        r = int(np.floor(self.filter_size / 2))
        imgWrap = torch.zeros((image_polar.shape[0], image_polar.shape[1], r*2+image_polar.shape[2], r*2+image_polar.shape[3])).requires_grad_(False).to(self.device)
        
        imgWrap[:, :, :r, :r] += torch.clone(image_polar[:, :, -r:, -r:]).requires_grad_(True)
        imgWrap[:, :, :r, r:-r] += torch.clone(image_polar[:, :, -r:, :]).requires_grad_(True)
        imgWrap[:, :, :r, -r:] += torch.clone(image_polar[:, :, -r:, :r]).requires_grad_(True)

        imgWrap[:, :, r:-r, :r] += torch.clone(image_polar[:, :, :, -r:]).requires_grad_(True)
        imgWrap[:, :, r:-r, r:-r] += torch.clone(image_polar).requires_grad_(True)
        imgWrap[:, :, r:-r, -r:] += torch.clone(image_polar[:, :, :, :r]).requires_grad_(True)

        imgWrap[:, :, -r:, :r] += torch.clone(image_polar[:, :, :r, -r:]).requires_grad_(True)
        imgWrap[:, :, -r:, r:-r] += torch.clone(image_polar[:, :, :r, :]).requires_grad_(True)
        imgWrap[:, :, -r:, -r:] += torch.clone(image_polar[:, :, :r, :r]).requires_grad_(True)
        
        codes = nn.functional.conv2d(imgWrap, self.filter, stride=1, padding='valid')

        return codes

    def forward(self, image_polar):
        codes = self.getCodes(image_polar) 
        return codes

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)
    
class NNIdentityLoss(nn.Module):
    def __init__(self, stem_width, network_path, device, vec_size=2000):
        super().__init__()
        self.model = ConvNext(in_channels=1, stem_features=stem_width, depths=[3,4,6,4], widths=[stem_width*4, stem_width*8, stem_width*16, stem_width*32], num_classes=vec_size)
        self.model.load_state_dict(torch.load(network_path, map_location=device))
        self.model.eval()
        self.cos_sim = nn.CosineSimilarity()
    def forward(self, input1, input2):
        output1 = self.model(input1)
        with torch.no_grad():
            output2 = self.model(input2)
        return 0.5 - 0.5 * self.cos_sim(output1, output2)
    
class TVLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        tv_h = torch.mean(torch.pow(input[:,:,1:,:]-input[:,:,:-1,:], 2), dim=(1,2,3))
        tv_w = torch.mean(torch.pow(input[:,:,:,1:]-input[:,:,:,:-1], 2), dim=(1,2,3))
        return tv_h+tv_w

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, device, size=(168, 224), resize=False, model='vgg19'):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        if model == 'vgg19':
            blocks.append(models.vgg19(pretrained=True).features[:4].eval().to(device))
            blocks.append(models.vgg19(pretrained=True).features[4:9].eval().to(device))
            blocks.append(models.vgg19(pretrained=True).features[9:18].eval().to(device))
            blocks.append(models.vgg19(pretrained=True).features[18:27].eval().to(device))
        elif model == 'vgg16':
            blocks.append(models.vgg19(pretrained=True).features[:4].eval().to(device))
            blocks.append(models.vgg19(pretrained=True).features[4:9].eval().to(device))
            blocks.append(models.vgg19(pretrained=True).features[9:16].eval().to(device))
            blocks.append(models.vgg19(pretrained=True).features[16:23].eval().to(device))
        self.blocks = torch.nn.ModuleList(blocks)
        self.resize = resize
        self.size = size
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=None):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = F.interpolate(input, mode='bilinear', size=self.size, align_corners=True)
            target = F.interpolate(target, mode='bilinear', size=self.size, align_corners=True)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += F.l1_loss(x, y)
            if style_layers is not None:
                if i in style_layers:
                    act_x = x.reshape(x.shape[0], x.shape[1], -1)
                    act_y = y.reshape(y.shape[0], y.shape[1], -1)
                    gram_x = act_x @ act_x.permute(0, 2, 1)
                    gram_y = act_y @ act_y.permute(0, 2, 1)
                    loss += F.l1_loss(gram_x, gram_y)
        return loss  

class LPIPSLoss(torch.nn.Module):
    def __init__(self, device, net_type='alex'):
        super(LPIPSLoss, self).__init__()
        self.lpips_loss = lpips.LPIPS(net=net_type).to(device)

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        return self.lpips_loss(input, target)

class PolarLPIPSLoss(torch.nn.Module):
    def __init__(self, device, net_type='alex'):
        super(PolarLPIPSLoss, self).__init__()
        self.lpips_loss = lpips.LPIPS(net=net_type).to(device)

    def forward(self, input, target):
        n_splits = int(input.shape[3] / input.shape[2])
        loss_splits = []
        for i in range(n_splits):
            input_split = torch.nn.functional.interpolate(input[:, :, :, i*input.shape[2]:(i+1)*input.shape[2]], size=(256, 256), mode='bilinear')
            target_split = torch.nn.functional.interpolate(target[:, :, :, i*target.shape[2]:(i+1)*target.shape[2]], size=(256, 256), mode='bilinear')
            if input_split.shape[1] != 3:
                input_split = input_split.repeat(1, 3, 1, 1)
                target_split = target_split.repeat(1, 3, 1, 1)
            loss_splits.append(self.lpips_loss(input_split, target_split).reshape(-1,1))
        loss = torch.cat(loss_splits, dim=1)
        return loss
        
class MS_SSIM_Loss(nn.Module):
    def __init__(self, resolution=(256,256), data_range=255, size_average=True, channel=1):
        super().__init__()
        smaller_side = min(resolution)
        win_size = (smaller_side / (2 ** 5)) - 1
        self.ms_ssim_module = MS_SSIM(data_range=data_range, size_average=size_average, channel=channel, win_size=win_size)
    def forward(self, img1, img2):
        return 1 - self.ms_ssim_module(img1, img2)
        
class MS_SSIM_Loss_polar(nn.Module):
    def __init__(self, data_range=255, size_average=True, channel=1):
        super().__init__()
        self.required_resolution = (176, 1408)
        self.ms_ssim_module = MS_SSIM(data_range=data_range, size_average=size_average, channel=channel)
    def forward(self, img1, img2):
        img1 = nn.functional.interpolate(img1, size=self.required_resolution, mode='bilinear')
        img2 = nn.functional.interpolate(img2, size=self.required_resolution, mode='bilinear')
        return 1 - self.ms_ssim_module(img1, img2)

class SSIM_Loss(SSIM):
    def forward(self, img1, img2):
        return ( 1 - super().forward(img1, img2) )

class SoftMarginLossWithSoftLabels(nn.Module):
    def __init__(self, device, label_smoothing=0.2):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.device = device
    def forward(self, input, target):
        zero = torch.zeros(target.shape).to(self.device)
        noise_neg = torch.rand(input.shape).to(self.device)*self.label_smoothing
        noise_neg = torch.where(target < 0, noise_neg, zero)
        noise_pos = torch.rand(input.shape).to(self.device)*self.label_smoothing
        noise_pos = torch.where(target > 0, noise_pos, zero)
        target_soft = target + noise_neg - noise_pos
        return nn.SoftMarginLoss()(input, target_soft.requires_grad_(False))

class L1LossWithSoftLabels(nn.Module):
    def __init__(self, device, label_smoothing=0.2):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.device = device
    def forward(self, input, target):
        zero = torch.zeros(target.shape).to(self.device)
        targetm1 = torch.clamp(target + torch.rand(target.shape).to(self.device)*self.label_smoothing, -1 ,1)
        target1 = torch.clamp(target - torch.rand(target.shape).to(self.device)*self.label_smoothing, -1, 1)
        target_soft = torch.where(target == -1, targetm1, zero) + torch.where(target == 1, target1, zero)
        return nn.L1Loss()(input, target_soft)

class BCELogitsLossWithSoftLabels(nn.Module):
    def __init__(self, device, label_smoothing=0.2):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.device = device
    def forward(self, input, target, mask):
        zero = torch.zeros(target.shape).to(self.device)
        target0 = torch.clamp(target + torch.rand(target.shape).to(self.device)*self.label_smoothing, 0 ,1)
        target1 = torch.clamp(target - torch.rand(target.shape).to(self.device)*self.label_smoothing, 0, 1)
        target_soft = torch.where(target == 0, target0, zero) + torch.where(target == 1, target1, zero)
        return nn.BCEWithLogitsLoss(pos_weight=mask)(input, target_soft)

class CosineLoss(nn.Module):
    def __init__(self, dim=1, eps=1e-08):
        super().__init__()
        self.cos_sim = nn.CosineSimilarity(dim=dim, eps=eps)
    def forward(self, inp1, inp2):
        return 0.5 - 0.5 * self.cos_sim(inp1, inp2)

class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None):
        super().__init__()

        self.loss = nn.NLLLoss(weight)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, outputs, targets):
        return self.loss(self.softmax(outputs), targets)

class HingeLoss(nn.Module):
    
    def __init__(self, device, p=1):
        super().__init__()
        self.p = p
        self.device = device
    
    def forward(self, outputs, targets):
        loss = torch.pow(torch.max(torch.tensor(0.).to(self.device), 1 - targets * outputs), self.p)
        return torch.mean(loss)

class HingeLossWithSoftLabels(nn.Module):
    def __init__(self, device, label_smoothing=0.2):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.device = device
        self.hinge_loss = HingeLoss(device, p=1)
    def forward(self, input, target):
        zero = torch.zeros(input.shape).to(self.device)
        noise_neg = torch.rand(input.shape).to(self.device)*self.label_smoothing
        noise_neg = torch.where(input < 0, noise_neg, zero)
        noise_pos = torch.rand(input.shape).to(self.device)*self.label_smoothing
        noise_pos = torch.where(input > 0, noise_pos, zero)
        input_soft = input + noise_neg.requires_grad_(False) - noise_pos.requires_grad_(False)
        return self.hinge_loss(input_soft, target)

#Networks    
            
class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class U_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class ConvNormAct(nn.Sequential):
    """
    A little util layer composed by (conv) -> (norm) -> (act) layers.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        norm = nn.BatchNorm2d,
        act = nn.ReLU,
        **kwargs
    ):
        super().__init__(
            nn.Conv2d(
                in_features,
                out_features,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                **kwargs
            ),
            norm(out_features),
            act(),
        )

class ConvNextStem(nn.Sequential):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(
            nn.Conv2d(in_features, out_features, kernel_size=4, stride=4),
            nn.BatchNorm2d(out_features)
        )

class LayerScaler(nn.Module):
    def __init__(self, init_value: float, dimensions: int):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones((dimensions)), 
                                    requires_grad=True)
        
    def forward(self, x):
        return self.gamma[None,...,None,None] * x

class BottleNeckBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        expansion: int = 4,
        drop_p: float = .0,
        layer_scaler_init_value: float = 1e-6,
    ):
        super().__init__()
        expanded_features = out_features * expansion
        self.block = nn.Sequential(
            # narrow -> wide (with depth-wise and bigger kernel)
            nn.Conv2d(
                in_features, in_features, kernel_size=7, padding=3, bias=False, groups=in_features
            ),
            # GroupNorm with num_groups=1 is the same as LayerNorm but works for 2D data
            nn.GroupNorm(num_groups=1, num_channels=in_features),
            # wide -> wide 
            nn.Conv2d(in_features, expanded_features, kernel_size=1),
            nn.GELU(),
            # wide -> narrow
            nn.Conv2d(expanded_features, out_features, kernel_size=1),
        )
        self.layer_scaler = LayerScaler(layer_scaler_init_value, out_features)
        self.drop_path = StochasticDepth(drop_p, mode="batch")

        
    def forward(self, x: Tensor) -> Tensor:
        res = x
        x = self.block(x)
        x = self.layer_scaler(x)
        x = self.drop_path(x)
        x += res
        return x

class ConvNexStage(nn.Sequential):
    def __init__(
        self, in_features: int, out_features: int, depth: int, **kwargs
    ):
        super().__init__(
            # add the downsampler
            nn.Sequential(
                nn.GroupNorm(num_groups=1, num_channels=in_features),
                nn.Conv2d(in_features, out_features, kernel_size=2, stride=2)
            ),
            *[
                BottleNeckBlock(out_features, out_features, **kwargs)
                for _ in range(depth)
            ],
        )

class ConvNextEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stem_features: int,
        depths: List[int],
        widths: List[int],
        drop_p: float = .0,
    ):
        super().__init__()
        self.stem = ConvNextStem(in_channels, stem_features)

        in_out_widths = list(zip(widths, widths[1:]))
        # create drop paths probabilities (one for each stage)
        drop_probs = [x.item() for x in torch.linspace(0, drop_p, sum(depths))] 
        
        self.stages = nn.ModuleList(
            [
                ConvNexStage(stem_features, widths[0], depths[0], drop_p=drop_probs[0]),
                *[
                    ConvNexStage(in_features, out_features, depth, drop_p=drop_p)
                    for (in_features, out_features), depth, drop_p in zip(
                        in_out_widths, depths[1:], drop_probs[1:]
                    )
                ],
            ]
        )
        

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x

class VectorHead(nn.Sequential):
    def __init__(self, num_channels: int, num_classes: int = 2000):
        super().__init__(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
            nn.LayerNorm(num_channels),
            nn.Linear(num_channels, num_channels),
            nn.GELU(),
            nn.Linear(num_channels, num_classes)
        )
    
    
class ConvNext(nn.Module):
    def __init__(self,  
                 in_channels: int,
                 stem_features: int,
                 depths: List[int],
                 widths: List[int],
                 drop_p: float = .0,
                 num_classes: int = 2000):
        super().__init__()
        self.encoder = ConvNextEncoder(in_channels, stem_features, depths, widths, drop_p)
        self.head = VectorHead(widths[-1], num_classes)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x
        
class ResBlockLR(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.conv_res = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(middle_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(middle_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        x = (x + res) * (1 / math.sqrt(2))
        return x 
        
class NestedResUNetResizedGrid(nn.Module):
    def __init__(self, num_channels, polar_height=64, polar_width=512, width=32, resolution=(240, 320)):
        super().__init__()
        self.resolution = resolution
        self.polar_height = polar_height
        self.polar_width = polar_width
        nb_filter = [width, width*2, width*4, width*8, width*16]

        self.pool = nn.AvgPool2d(2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv0_0 = ResBlockLR(num_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = ResBlockLR(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = ResBlockLR(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = ResBlockLR(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = ResBlockLR(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = ResBlockLR(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = ResBlockLR(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = ResBlockLR(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = ResBlockLR(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = ResBlockLR(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = ResBlockLR(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = ResBlockLR(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = ResBlockLR(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = ResBlockLR(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = ResBlockLR(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0]*4, 2, kernel_size=1)
        
        self.tanh = nn.Tanh()

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        
        output = self.final(torch.cat([x0_1, x0_2, x0_3, x0_4], 1))
        
        output = F.interpolate(output, size=(self.polar_height, self.polar_width), mode='bilinear')
        
        output = output.reshape(-1, output.shape[2], output.shape[3], 2)
        output = self.tanh(output)
        
        return output
        
class ResNetDiscriminator(nn.Module):
    def __init__(self, num_channels, num_classes, width=32, resolution=(64, 512)):
        super().__init__()
        self.resolution = resolution
        self.num_classes = num_classes
        self.num_channels = num_channels
        nb_filter = [width, width*2, width*4, width*8, width*16]
        self.pool = Resize(scale_factor=0.5, mode='bilinear')
        
        self.conv0 = ResBlockLR(num_channels, nb_filter[0], nb_filter[0]) 
        self.conv1 = ResBlockLR(nb_filter[0], nb_filter[1], nb_filter[1]) #32x256 
        self.conv2 = ResBlockLR(nb_filter[1], nb_filter[2], nb_filter[2]) #16x128 
        self.conv3 = ResBlockLR(nb_filter[2], nb_filter[3], nb_filter[3]) #8x64 
        self.conv4 = ResBlockLR(nb_filter[3], nb_filter[4], nb_filter[4]) #4x32
        
        self.fc_init = nn.Sequential(
            nn.Linear(int(nb_filter[4]*resolution[0]*resolution[1]*(1/256)), nb_filter[4], bias=False),
            nn.BatchNorm1d(nb_filter[4]),
            nn.Mish(inplace=True)
        )
        self.fc_rf = nn.Linear(nb_filter[4], 1)
        self.fc_cls = nn.Linear(nb_filter[4], num_classes)
        
    def forward(self, input):
        x0 = self.conv0(input)
        x1 = self.conv1(self.pool(x0))
        x2 = self.conv2(self.pool(x1))
        x3 = self.conv3(self.pool(x2))
        x4 = self.conv4(self.pool(x3))
        
        x5 = self.fc_init(nn.Flatten()(x4))
        out_rf = self.fc_rf(x5)
        out_cls = self.fc_cls(x5)
        
        return out_rf, out_cls        
            
class SharedAtrousConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True):
        super(SharedAtrousConv2d,self).__init__()
        self.weights = nn.Parameter(torch.rand(int(out_channels/2), in_channels, kernel_size, kernel_size))
        nn.init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        if bias:
            self.bias1 = nn.Parameter(torch.zeros(int(out_channels/2)))
            self.bias2 = nn.Parameter(torch.zeros(int(out_channels/2)))
        else:
            self.bias1 = None
            self.bias2 = None
    def forward(self, x):
        x1 = nn.functional.conv2d(x, self.weights, stride=1, padding='same', bias=self.bias1)
        x2 = nn.functional.conv2d(x, self.weights, stride=1, padding='same', dilation=2, bias=self.bias2)
        x3 = torch.cat([x1, x2], 1)
        return x3
        
class SharedAtrousResBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, groups=8):
        super(SharedAtrousResBlock,self).__init__()
        self.conv_res = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=1),
            nn.GroupNorm(groups, out_channels)
        )
        self.net = nn.Sequential(
            SharedAtrousConv2d(in_channels, middle_channels),
            nn.GroupNorm(groups, middle_channels),
            nn.SiLU(),
            SharedAtrousConv2d(middle_channels, out_channels),
            nn.GroupNorm(groups, out_channels)
        )
        self.relu = nn.SiLU()
    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        x = (x + res) * (1 / math.sqrt(2))
        x = self.relu(x)
        return x
        
class SharedAtrousConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8):
        super(SharedAtrousConvBlock,self).__init__()
        self.net = nn.Sequential(
            SharedAtrousConv2d(in_channels, out_channels),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU()
        )
    def forward(self, x):
        return self.net(x)

class SharedAtrousUpConv(nn.Module):
    def __init__(self,ch_in,ch_out,groups=8):
        super(SharedAtrousUpConv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            SharedAtrousConv2d(ch_in, ch_out),
		        nn.GroupNorm(groups, ch_out),
            nn.SiLU()
        )
    def forward(self, x):
        return self.up(x)
        
class SharedAtrousDownConv(nn.Module):
    def __init__(self,ch_in,ch_out,groups=8):
        super(SharedAtrousDownConv,self).__init__()
        self.down = nn.Sequential(
            nn.PixelUnshuffle(2),
            SharedAtrousConv2d(ch_in*4, ch_out),
            nn.GroupNorm(groups, ch_out),
            nn.SiLU()
         )
    def forward(self, x):
        return self.down(x)

class NestedSharedAtrousResUNet(nn.Module):
    def __init__(self, num_classes, num_channels, width=32, resolution=(256, 256)):
        super().__init__()
        self.resolution = resolution
        nb_filter = [width, width*2, width*4, width*8, width*16]

        #self.pool = nn.PixelUnshuffle(2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.pool1 = SharedAtrousDownConv(ch_in=nb_filter[0], ch_out=nb_filter[0])
        self.pool2 = SharedAtrousDownConv(ch_in=nb_filter[1], ch_out=nb_filter[1])
        self.pool3 = SharedAtrousDownConv(ch_in=nb_filter[2], ch_out=nb_filter[2])
        self.pool4 = SharedAtrousDownConv(ch_in=nb_filter[3], ch_out=nb_filter[3])

        self.conv0_0 = SharedAtrousResBlock(num_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = SharedAtrousResBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = SharedAtrousResBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = SharedAtrousResBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = SharedAtrousResBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = SharedAtrousResBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = SharedAtrousResBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = SharedAtrousResBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = SharedAtrousResBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = SharedAtrousResBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = SharedAtrousResBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = SharedAtrousResBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = SharedAtrousResBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = SharedAtrousResBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = SharedAtrousResBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0]*4, num_classes, kernel_size=1)

    def forward(self, input):
        
        x0_0 = self.conv0_0(input) #320x240
        x1_0 = self.conv1_0(self.pool1(x0_0)) #160x120
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool2(x1_0)) #80x60
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool3(x2_0)) #40x30
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool4(x3_0)) #20x15
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        
        output = self.final(torch.cat([x0_1, x0_2, x0_3, x0_4], 1))
        
        return output

class SharedAtrousPatchDiscriminator(nn.Module):
    def __init__(self, in_channels=1, groups=8):
        super(SharedAtrousPatchDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            #layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            layers = [SharedAtrousConv2d(in_filters, out_filters)]
            if normalization:
                layers.append(nn.GroupNorm(groups, out_filters))
            layers.append(nn.SiLU())
            layers.append(Resize(scale_factor=0.5, mode='nearest'))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False), #128
            *discriminator_block(64, 128), #64
            *discriminator_block(128, 256), #32
            *discriminator_block(256, 512), #16
        )
        self.head_realfake = nn.Conv2d(512, 1, 3, stride=1, padding='same', bias=False)
        self.head_classifier = nn.Conv2d(512, 1, 3, stride=1, padding='same', bias=False)

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        feat = self.model(img_input)
        return self.head_realfake(feat), self.head_classifier(feat)

class NestedSharedAtrousResUNetPS(nn.Module):
    def __init__(self, num_classes, num_channels, width=32, resolution=(256, 256)):
        super().__init__()
        self.resolution = resolution
        nb_filter = [width, width*2, width*4, width*8, width*16]

        self.pool = nn.PixelUnshuffle(2)
        self.up = nn.Upsample(scale_factor=2, mode='bicubic')

        self.conv0_0 = SharedAtrousResBlock(num_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = SharedAtrousResBlock(nb_filter[0]*4, nb_filter[1], nb_filter[1])
        self.conv2_0 = SharedAtrousResBlock(nb_filter[1]*4, nb_filter[2], nb_filter[2])
        self.conv3_0 = SharedAtrousResBlock(nb_filter[2]*4, nb_filter[3], nb_filter[3])
        self.conv4_0 = SharedAtrousResBlock(nb_filter[3]*4, nb_filter[4], nb_filter[4])

        self.conv0_1 = SharedAtrousResBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = SharedAtrousResBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = SharedAtrousResBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = SharedAtrousResBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = SharedAtrousResBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = SharedAtrousResBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = SharedAtrousResBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = SharedAtrousResBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = SharedAtrousResBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = SharedAtrousResBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0]*5, num_classes, kernel_size=1)

    def forward(self, input):
        
        x0_0 = self.conv0_0(input) #320x240
        x1_0 = self.conv1_0(self.pool(x0_0)) #160x120
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0)) #80x60
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0)) #40x30
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0)) #20x15
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        
        output = self.final(torch.cat([x0_0, x0_1, x0_2, x0_3, x0_4], 1))
        
        return output
        
class SharedAtrousResBlockIN(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(SharedAtrousResBlockIN,self).__init__()
        self.conv_res = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=1),
            nn.InstanceNorm2d(out_channels)
        )
        self.net = nn.Sequential(
            SharedAtrousConv2d(in_channels, middle_channels),
            nn.InstanceNorm2d(middle_channels),
            nn.GELU(),
            SharedAtrousConv2d(middle_channels, out_channels),
            nn.InstanceNorm2d(out_channels)
        )
        self.act = nn.GELU()
    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        x = (x + res) * (1 / math.sqrt(2))
        x = self.act(x)
        return x
        
class SharedAtrousConvBlockIN(nn.Module):
    def __init__(self, in_channels, out_channels, normalization=True):
        super(SharedAtrousConvBlockIN,self).__init__()
        self.conv = SharedAtrousConv2d(in_channels, out_channels)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.act = nn.GELU()
        self.do_norm = normalization
    def forward(self, x):
        x = self.conv(x)
        if self.do_norm:
            x = self.norm(x)
        x = self.act(x)
        return x

class SharedAtrousUpConvIN(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(SharedAtrousUpConvIN,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            SharedAtrousConv2d(ch_in, ch_out),
		        nn.InstanceNorm2d(ch_out),
            nn.GELU()
        )
    def forward(self, x):
        return self.up(x)
        
class SharedAtrousDownConvIN(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(SharedAtrousDownConvIN,self).__init__()
        self.down = nn.Sequential(
            nn.PixelUnshuffle(2),
            SharedAtrousConv2d(ch_in*4, ch_out),
            nn.InstanceNorm2d(ch_out),
            nn.GELU()
         )
    def forward(self, x):
        return self.down(x)

class SharedAtrousPatchDiscriminatorIN(nn.Module):
    def __init__(self, in_channels=1):
        super(SharedAtrousPatchDiscriminatorIN, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            #layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            layers = [SharedAtrousConv2d(in_filters, out_filters)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.GELU())
            layers.append(Resize(scale_factor=0.5, mode='nearest'))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64), #128
            *discriminator_block(64, 128), #64
            *discriminator_block(128, 256), #32
            *discriminator_block(256, 512), #16
        )
        self.head_realfake = nn.Conv2d(512, 1, 3, stride=1, padding='same', bias=False)
        self.head_classifier = nn.Conv2d(512, 1, 3, stride=1, padding='same', bias=False)

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        feat = self.model(img_input)
        return self.head_realfake(feat), self.head_classifier(feat)

class NestedSharedAtrousResUNetIN(nn.Module):
    def __init__(self, num_classes, num_channels, width=32, resolution=(256, 256)):
        super().__init__()
        self.resolution = resolution
        nb_filter = [width, width*2, width*4, width*8, width*16]

        self.pool = nn.PixelUnshuffle(2)
        self.up = nn.Upsample(scale_factor=2, mode='bicubic')

        self.conv0_0 = SharedAtrousResBlockIN(num_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = SharedAtrousResBlockIN(nb_filter[0]*4, nb_filter[1], nb_filter[1])
        self.conv2_0 = SharedAtrousResBlockIN(nb_filter[1]*4, nb_filter[2], nb_filter[2])
        self.conv3_0 = SharedAtrousResBlockIN(nb_filter[2]*4, nb_filter[3], nb_filter[3])
        self.conv4_0 = SharedAtrousResBlockIN(nb_filter[3]*4, nb_filter[4], nb_filter[4])

        self.conv0_1 = SharedAtrousResBlockIN(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = SharedAtrousResBlockIN(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = SharedAtrousResBlockIN(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = SharedAtrousResBlockIN(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = SharedAtrousResBlockIN(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = SharedAtrousResBlockIN(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = SharedAtrousResBlockIN(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = SharedAtrousResBlockIN(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = SharedAtrousResBlockIN(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = SharedAtrousResBlockIN(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0]*5, num_classes, kernel_size=1)

    def forward(self, input):
        
        x0_0 = self.conv0_0(input) #320x240
        x1_0 = self.conv1_0(self.pool(x0_0)) #160x120
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0)) #80x60
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0)) #40x30
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0)) #20x15
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        
        output = self.final(torch.cat([x0_0, x0_1, x0_2, x0_3, x0_4], 1))
        
        return output

class SharedAtrousVectorDiscriminatorIN(nn.Module):
    def __init__(self, in_channels=1):
        super(SharedAtrousVectorDiscriminatorIN, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            #layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            layers = [SharedAtrousConv2d(in_filters, out_filters)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.GELU())
            layers.append(Resize(scale_factor=0.5, mode='nearest'))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64), #128  / 32x256
            *discriminator_block(64, 128), #64  / 16x128
            *discriminator_block(128, 256), #32  / 8x64
            *discriminator_block(256, 512), #16 / 4x32
            SharedAtrousConv2d(512, 16),
            nn.InstanceNorm2d(16),
            nn.GELU(),
            nn.Flatten(),
            nn.LazyLinear(2048)
        )

    def forward(self, img):
        # Concatenate image and condition image by channels to produce input
        return self.model(img)
        

class ResBlockIN(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, downsample=False):
        super(ResBlockIN,self).__init__()
        self.conv_res = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding='same', bias=False),
            nn.InstanceNorm2d(out_channels)
        )
        self.downsample = downsample
        if downsample:
            self.net_downsample = nn.Sequential(
                nn.Conv2d(in_channels, middle_channels, 4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(middle_channels),
                nn.GELU(),
                nn.Conv2d(middle_channels, out_channels, 3, stride=1, padding='same', bias=False),
                nn.InstanceNorm2d(out_channels)
            )
        else:
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, middle_channels, 3, stride=1, padding='same', bias=False),
                nn.InstanceNorm2d(middle_channels),
                nn.GELU(),
                nn.Conv2d(middle_channels, out_channels, 3, stride=1, padding='same', bias=False),
                nn.InstanceNorm2d(out_channels)
            )
        self.act = nn.GELU()
    def forward(self, x):
        res = self.conv_res(x)
        if self.downsample:
            x = self.net_downsample(x)
            res = nn.functional.interpolate(res, scale_factor=0.5)
        else:
            x = self.net(x)
        x = (x + res) * (1 / math.sqrt(2))
        x = self.act(x)
        return x          
        
class ConvBlockIN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlockIN,self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=1, bias=False)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.act = nn.GELU()
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class UpConvIN(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(UpConvIN,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, 3, stride=1, bias=False),
		        nn.InstanceNorm2d(ch_out),
            nn.GELU()
        )
    def forward(self, x):
        return self.up(x)
        
class DownConv(nn.Module):
    def __init__(self,ch):
        super(DownConv,self).__init__()
        self.down = nn.Sequential(
            Resize(scale_factor=0.5, mode='bilinear'),
            nn.Conv2d(ch, ch, 3, stride=1, bias=False),
            nn.InstanceNorm2d(ch_out),
            nn.GELU()
         )
    def forward(self, x):
        return self.down(x)
        
class WSNetwork(nn.Module):
    def __init__(self, in_channels, width = 8, blocks_per_res=1):
        super().__init__()
        modules = []
        modules.append(ResBlockIN(in_channels, width, width)) #8x256x256
        for i in range(blocks_per_res):
            modules.append(ResBlockIN(width, width, width))
        modules.append(ResBlockIN(width, width*2, width*2, downsample=True)) #128x128
        for i in range(blocks_per_res):
            modules.append(ResBlockIN(width*2, width*2, width*2))
        modules.append(ResBlockIN(width*2, width*4, width*4, downsample=True)) #64x64
        for i in range(blocks_per_res):
            modules.append(ResBlockIN(width*4, width*4, width*4))
        modules.append(ResBlockIN(width*4, width*8, width*8, downsample=True)) #32x32
        for i in range(blocks_per_res):
            modules.append(ResBlockIN(width*8, width*8, width*8))
        modules.append(ResBlockIN(width*8, width*16, width*16, downsample=True)) #16x16
        for i in range(blocks_per_res):
            modules.append(ResBlockIN(width*16, width*16, width*16))
        modules.append(ResBlockIN(width*16, width*32, width*32, downsample=True)) #8x8
        for i in range(blocks_per_res):
            modules.append(ResBlockIN(width*32, width*32, width*32))
        modules.append(ResBlockIN(width*32, width*64, width*64, downsample=True)) #4x4
        for i in range(blocks_per_res):
            modules.append(ResBlockIN(width*64, width*64, width*64)) #512x4x4
        modules.append(nn.Conv2d(width*64, 512, 1, stride=1, padding='same'))
        self.net = nn.Sequential(*modules)
    def forward(self, x):
        x = self.net(x)
        x = x.reshape(-1, 16, 512)
        return x


class PatchDiscriminatorIN(nn.Module):
    def __init__(self, in_channels=1, width=64):
        super(PatchDiscriminatorIN, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1, bias=False)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.GELU())
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, width), #128
            *discriminator_block(width, 2*width), #64
            *discriminator_block(2*width, 4*width), #32
            *discriminator_block(4*width, 8*width), #16
            nn.ZeroPad2d((1, 0, 1, 0))
        )
        self.head_realfake = nn.Conv2d(8*width, 1, 4, padding=1, bias=False)
        self.head_classifier = nn.Conv2d(8*width, 1, 4, padding=1, bias=False)

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        feat = self.model(img_input)
        return self.head_realfake(feat), self.head_classifier(feat)

class NestedResUNetIN(nn.Module):
    def __init__(self, num_classes, num_channels, width=64, resolution=(256, 256)):
        super().__init__()
        self.resolution = resolution
        nb_filter = [width, width*2, width*4, width*8, width*16]

        self.pool = nn.PixelUnshuffle(2)
        self.up = nn.Upsample(scale_factor=2, mode='bicubic')

        self.conv0_0 = ResBlockIN(num_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = ResBlockIN(nb_filter[0]*4, nb_filter[1], nb_filter[1])
        self.conv2_0 = ResBlockIN(nb_filter[1]*4, nb_filter[2], nb_filter[2])
        self.conv3_0 = ResBlockIN(nb_filter[2]*4, nb_filter[3], nb_filter[3])
        self.conv4_0 = ResBlockIN(nb_filter[3]*4, nb_filter[4], nb_filter[4])

        self.conv0_1 = ResBlockIN(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = ResBlockIN(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = ResBlockIN(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = ResBlockIN(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = ResBlockIN(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = ResBlockIN(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = ResBlockIN(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = ResBlockIN(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = ResBlockIN(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = ResBlockIN(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0]*5, num_classes, 1)

    def forward(self, input):
        
        x0_0 = self.conv0_0(input) #320x240
        x1_0 = self.conv1_0(self.pool(x0_0)) #160x120
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0)) #80x60
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0)) #40x30
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0)) #20x15
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        
        output = self.final(torch.cat([x0_0, x0_1, x0_2, x0_3, x0_4], 1))
        
        return output
        
class AttBlockIN(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(AttBlockIN,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.InstanceNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.InstanceNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.InstanceNorm2d(1),
            nn.Sigmoid()
        )
        
        self.act = nn.GELU()
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.act(g1+x1)
        psi = self.psi(psi)

        return x*psi
        
class NestedAttentionResUNetIN(nn.Module):
    def __init__(self, num_classes, num_channels, width=32, resolution=(256, 256)):
        super().__init__()
        self.resolution = resolution
        nb_filter = [width, width*2, width*4, width*8, width*16]
        self.nb_filter = nb_filter

        self.pool = nn.PixelUnshuffle(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv0_0 = ResBlockIN(num_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = ResBlockIN(nb_filter[0]*4, nb_filter[1], nb_filter[1])
        self.conv2_0 = ResBlockIN(nb_filter[1]*4, nb_filter[2], nb_filter[2])
        self.conv3_0 = ResBlockIN(nb_filter[2]*4, nb_filter[3], nb_filter[3])
        self.conv4_0 = ResBlockIN(nb_filter[3]*4, nb_filter[4], nb_filter[4])

        self.conv0_1 = ResBlockIN(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.att0_1 = AttBlockIN(nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = ResBlockIN(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.att1_1 = AttBlockIN(nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = ResBlockIN(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.att2_1 = AttBlockIN(nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = ResBlockIN(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.att3_1 = AttBlockIN(nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = ResBlockIN(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.att0_2 = AttBlockIN(nb_filter[1], nb_filter[0]*2, nb_filter[0])
        self.conv1_2 = ResBlockIN(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.att1_2 = AttBlockIN(nb_filter[2], nb_filter[1]*2, nb_filter[1])
        self.conv2_2 = ResBlockIN(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])
        self.att2_2 = AttBlockIN(nb_filter[3], nb_filter[2]*2, nb_filter[2])

        self.conv0_3 = ResBlockIN(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.att0_3 = AttBlockIN(nb_filter[1], nb_filter[0]*3, nb_filter[0])
        self.conv1_3 = ResBlockIN(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])
        self.att1_3 = AttBlockIN(nb_filter[2], nb_filter[1]*3, nb_filter[1])

        self.conv0_4 = ResBlockIN(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])
        self.att0_4 = AttBlockIN(nb_filter[1], nb_filter[0]*4, nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0]*4, num_classes, kernel_size=1)

    def forward(self, input):
        
        x0_0 = self.conv0_0(input) #320x240
        x1_0 = self.conv1_0(self.pool(x0_0)) #160x120
        x0_0 = self.att0_1(g=self.up(x1_0), x=x0_0)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0)) #80x60
        
        x1_0 = self.att1_1(g=self.up(x2_0), x=x1_0)
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        
        x0_01 = self.att0_2(g=self.up(x1_1), x=torch.cat([x0_0, x0_1], 1))
        x0_2 = self.conv0_2(torch.cat([x0_01, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0)) #40x30
        x2_0 = self.att2_1(g=self.up(x3_0), x=x2_0)
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_01 = self.att1_2(g=self.up(x2_1), x=torch.cat([x1_0, x1_1], 1))
        x1_2 = self.conv1_2(torch.cat([x1_01, self.up(x2_1)], 1))
        x0_012 = self.att0_3(g=self.up(x1_2), x=torch.cat([x0_0, x0_1, x0_2], 1))
        x0_3 = self.conv0_3(torch.cat([x0_012, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0)) #20x15
        x3_0 = self.att3_1(g=self.up(x4_0), x=x3_0)
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_01 = self.att2_2(g=self.up(x3_1), x=torch.cat([x2_0, x2_1], 1))
        x2_2 = self.conv2_2(torch.cat([x2_01, self.up(x3_1)], 1))
        x1_012 = self.att1_3(g=self.up(x2_2), x=torch.cat([x1_0, x1_1, x1_2], 1))
        x1_3 = self.conv1_3(torch.cat([x1_012, self.up(x2_2)], 1))
        x0_0123 = self.att0_4(g=self.up(x1_3), x=torch.cat([x0_0, x0_1, x0_2, x0_3], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0123, self.up(x1_3)], 1))
        
        output = self.final(torch.cat([x0_1, x0_2, x0_3, x0_4], 1))
        
        return output

class VectorDiscriminatorIN(nn.Module):
    def __init__(self, in_channels=1, width=64, vec_size=2048):
        super().__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1, bias=False)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.GELU())
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, width), #128  / 32x256
            *discriminator_block(width, 2*width), #64  / 16x128
            *discriminator_block(2*width, 4*width), #32  / 8x64
            *discriminator_block(4*width, 8*width), #16 / 4x32
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(8*width, int(width/2), 4, padding=1, bias=False),
            nn.InstanceNorm2d(int(width/2)),
            nn.GELU(),
            nn.Flatten(),
            nn.LazyLinear(vec_size)
        )

    def forward(self, img):
        # Concatenate image and condition image by channels to produce input
        return self.model(img)
        
class VectorDiscriminatorINv2(nn.Module):
    def __init__(self, in_channels=1, width=64, vec_size=2048):
        super().__init__()
        self.model = nn.Sequential(
            ResBlockIN(in_channels, width, width, downsample=True), #128  / 32x256
            ResBlockIN(width, 2*width, 2*width, downsample=True), #64  / 16x128
            ResBlockIN(2*width, 4*width, 4*width, downsample=True), #32  / 8x64
            ResBlockIN(4*width, 8*width, 8*width, downsample=True), #16 / 4x32
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(8*width, width, 4, padding=1, bias=False),
            nn.InstanceNorm2d(width),
            nn.GELU(),
            nn.Flatten(),
            nn.LazyLinear(vec_size)
        )

    def forward(self, img):
        # Concatenate image and condition image by channels to produce input
        return self.model(img)
    
class VectorAndPatchDiscriminatorIN(nn.Module):
    def __init__(self, in_channels=1, width=64, vec_size=2048):
        super().__init__()
        self.model = nn.Sequential(
            ResBlockIN(in_channels, width, width, downsample=True), #128  / 32x256
            ResBlockIN(width, 2*width, 2*width, downsample=True), #64  / 16x128
            ResBlockIN(2*width, 4*width, 4*width, downsample=True), #32  / 8x64
            ResBlockIN(4*width, 8*width, 8*width, downsample=True), #16 / 4x32
            nn.ZeroPad2d((1, 0, 1, 0))
        )
        self.vector_head = nn.Sequential(
            nn.Conv2d(8*width, width, 4, padding=1, bias=False),
            nn.InstanceNorm2d(width),
            nn.GELU(),
            nn.Flatten(),
            nn.LazyLinear(vec_size)
        )
        self.realfake_head = nn.Conv2d(8*width, 1, 4, padding=1, bias=False)

    def forward(self, img):
        # Concatenate image and condition image by channels to produce input
        feat = self.model(img)
        return self.vector_head(feat), self.realfake_head(feat)

class fclayer(nn.Module):
    def __init__(self, in_h = 4, in_w = 32, out_n = 2048):
        super().__init__()
        self.in_h = in_h
        self.in_w = in_w
        self.out_n = out_n
        self.fc_list = []
        for i in range(out_n):
            self.fc_list.append(nn.Linear(in_h*in_w, 1))
        self.fc_list = nn.ModuleList(self.fc_list)
    def forward(self, x):
        x = x.reshape(-1, self.out_n, self.in_h, self.in_w)
        outs = []
        for i in range(self.out_n):
            outs.append(self.fc_list[i](x[:, i, :, :].reshape(-1, self.in_h*self.in_w)))
        out = torch.cat(outs, 1)
        return out

class conv(nn.Module):
    def __init__(self, in_channels=512, out_n = 6):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_n, kernel_size=1, stride=1, padding='same')
    def forward(self, x):
        x = self.conv(x)
        return x
  
class VectorAndPatchDiscriminatorINv2(nn.Module):
    def __init__(self, in_channels=1, width=128, vec_size=2048):
        super().__init__()
        self.model = nn.Sequential(
            ResBlockIN(in_channels, width, width, downsample=True), # 32x256
            ResBlockIN(width, 2*width, 2*width, downsample=True), # 16x128
            ResBlockIN(2*width, 4*width, 4*width, downsample=True), # 8x64
            ResBlockIN(4*width, 8*width, 8*width, downsample=True) # 4x32
        )
        self.vector_head = nn.Sequential(
            ResBlockIN(8*width, vec_size, vec_size),
            fclayer(in_h=4, in_w=32, out_n=vec_size)
        )
        self.realfake_head = nn.Sequential(
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(8*width, 1, 4, padding=1, bias=False)
        )
        
    def forward(self, img):
        # Concatenate image and condition image by channels to produce input
        feat = self.model(img)
        return self.vector_head(feat), self.realfake_head(feat)    
        
        
class LinearDeformer(object):
    def __init__(self, device):
        self.device = device
        
    def find_xy_between(self, r1, r2, r1p, r3p, xp, yp, xc, yc):
        r2_r1p = r2 - r1p
        r2_r1p = torch.where(r2_r1p == 0, 0.001, r2_r1p)
        r3 = (((r3p - r1p) * (r2 - r1)) / r2_r1p) + r1
        r3p = torch.where(r3p == 0, 0.001, r3p)
        x = ((r3 * (xp - xc)) / r3p) + xc
        y = ((r3 * (yp - yc)) / r3p) + yc
        return x, y
    
    def find_xy_less(self, r1, r1p, xp, yp, xc, yc):
        r1p = torch.where(r1p == 0, 0.001, r1p)
        x = ((r1 * (xp - xc)) / r1p) + xc
        y = ((r1 * (yp - yc)) / r1p) + yc
        return x, y
    
    def create_grid(self, r1, r2, r1p, b, h, w, xc, yc): # batched implementation
        Y = torch.arange(0, h).reshape(1, h, 1).repeat(b, 1, w).float().to(self.device)
        X = torch.arange(0, w).reshape(1, 1, w).repeat(b, h, 1).float().to(self.device)
        xcr = xc.reshape(-1, 1, 1).repeat(1, h, w).float()
        ycr = yc.reshape(-1, 1, 1).repeat(1, h, w).float()

        r3p = torch.sqrt(torch.square(X - xcr) + torch.square(Y - ycr)) #(b,h,w) #could be a source of nan
        r1r = r1.reshape(-1, 1, 1).repeat(1, h, w)
        r2r = r2.reshape(-1, 1, 1).repeat(1, h, w)
        r1pr = r1p.reshape(-1, 1, 1).repeat(1, h, w)

        no_change_cond = torch.where(r3p > r2r, 1, 0).unsqueeze(3).repeat(1,1,1,2).to(self.device)
        no_change = torch.cat((X.unsqueeze(3), Y.unsqueeze(3)), dim=3)
        xy_between_cond = torch.where(torch.logical_and(r3p >= r1pr, r3p <= r2r), 1, 0).unsqueeze(3).repeat(1,1,1,2).to(self.device)
        x_between, y_between = self.find_xy_between(r1r, r2r, r1pr, r3p, X, Y, xcr, ycr)
        xy_between = torch.cat((x_between.unsqueeze(3), y_between.unsqueeze(3)), dim=3).to(self.device)

        xy_less_cond = torch.where(r3p < r1pr, 1, 0).unsqueeze(3).repeat(1,1,1,2).to(self.device)
        x_less, y_less = self.find_xy_less(r1r, r1pr, X, Y, xcr, ycr)
        xy_less = torch.cat((x_less.unsqueeze(3), y_less.unsqueeze(3)), dim=3).to(self.device)

        grid = (no_change_cond * no_change + xy_between_cond * xy_between + xy_less_cond * xy_less).float().to(self.device)
        gridx = grid[:, :, :, 0]
        gridy = grid[:, :, :, 1]
        gridx = gridx / (w - 1)
        gridx = (gridx - 0.5) * 2
        gridy = gridy / (h - 1)
        gridy = (gridy - 0.5) * 2
        norm_grid = torch.stack([gridx, gridy], dim=-1).to(self.device)
        return norm_grid
    
    def linear_deform(self, image, input_pxyr, input_ixyr, alpha, mode='bilinear'):
        xc = ((input_pxyr[:, 0] + input_ixyr[:, 0])/2).float().reshape(-1,1).to(self.device)
        yc = ((input_pxyr[:, 1] + input_ixyr[:, 1])/2).float().reshape(-1,1).to(self.device)
        r1 = input_pxyr[:, 2].float().reshape(-1,1).to(self.device)
        r2 = input_ixyr[:, 2].float().reshape(-1,1).to(self.device)
        r1p = r2 * alpha.float().to(self.device)
        b, c, h, w = image.shape
        grid = self.create_grid(r1, r2, r1p, b, h, w, xc, yc)
        deformed_image = torch.nn.functional.grid_sample(image.to(self.device), grid, mode=mode)
        return deformed_image, grid

class Resize(nn.Module):
    def __init__(self, scale_factor, mode):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class ResBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.conv_res = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(middle_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(middle_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.final_relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        x = (x + res) * (1 / math.sqrt(2))
        x = self.final_relu(x)
        return x