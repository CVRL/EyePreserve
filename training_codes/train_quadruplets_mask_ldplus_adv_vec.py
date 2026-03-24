import numpy as np
import torch
#torch.multiprocessing.set_sharing_strategy('file_system')
import os
import math

import cv2
from argparse import ArgumentParser
import torch.nn as nn
import dnnlib
import legacy
import re
from typing import List, Optional, Tuple, Union
import click

from torch.optim import Adam, AdamW, SGD, lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
from torch.nn.functional import interpolate
import torch.nn.functional as F
from torch.autograd import Function 
from torch import autograd

from pytorch_optimizer import Prodigy

from dataset import *

from network import *

import datetime
import os
from typing import List, Optional, Tuple, Union
from modules.utils import get_cfg

from torchvision import models, transforms

from tqdm import tqdm
from scipy import io

import math
from math import pi

import math
import sys
from DISTS_pytorch import DISTS
import shutil
import madgrad
import random
from PIL import Image, ImageDraw, ImageFont

from torch.autograd import Variable

from kornia.geometry.transform import translate as tensor_translate
from kornia.geometry.transform import rotate as tensor_rotate

from ema_pytorch import EMA
        
def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst
    
def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst
    
def minmaximage(t):
    t_min, t_max = t.min(), t.max()
    t_p = ((t - t_min)/(t_max - t_min))*255
    return t_p
    
def calculate_shift(anchor_code, anchor_mask, inp_code, inp_mask, max_shift=16):
    #print(anchor_code.shape, anchor_mask.shape, inp_code.shape, inp_mask.shape)
    with torch.no_grad():
        scores = []
        anchor_code_binary = torch.where(anchor_code[:, :12, :, :] > 0, True, False) # b x filt_size x 64 x 512
        inp_code_binary = torch.where(inp_code[:, :12, :, :] > 0, True, False)  # b x filt_size x 64 x 512
        for shift in range(-max_shift, max_shift+1):
            andMasks = torch.logical_and(anchor_mask, torch.roll(inp_mask, shift, -1))  # b x 1 x  64 x 512
            xorCodes = torch.logical_xor(anchor_code_binary, torch.roll(inp_code_binary, shift, -1)) # b x filt_size x 64 x 512
            xorCodesMasked = torch.logical_and(xorCodes, andMasks.repeat(1, xorCodes.shape[1], 1, 1)) # b x filt_size x 64 x 512
            scores.append((torch.sum(xorCodesMasked, dim=(1,2,3)) / (torch.sum(andMasks, dim=(1,2,3)) * 12)).unsqueeze(-1)) # b * (max_shift+1)*2
        scores = torch.cat(scores, 1) # b x (max_shift+1)*2
        scores_index = torch.argmin(scores, dim=1)
        shifts = scores_index - max_shift
    return shifts.cpu().numpy().tolist()
        
def shift_tensor_batch(tensors, shifts, dim):
    shifted_tensors = []
    for i in range(tensors.shape[0]):
        shifted_tensor = torch.roll(tensors[i, :, :, :].clone().detach(), shifts[i], dim)
        shifted_tensors.append(shifted_tensor.unsqueeze(0))
    shifted_tensors = torch.cat(shifted_tensors, dim=0)
    return shifted_tensors

def rotate_tensor_batch(tensors, masks, shifts, pupil_xyr, iris_xyr):
    rotated_tensors = []
    rotated_masks = []
    pupil_xyr_new = pupil_xyr.clone().detach()
    iris_xyr_new = iris_xyr.clone().detach()
    for i in range(tensors.shape[0]):
        cx = iris_xyr[i, 0].item()
        cy = iris_xyr[i, 1].item()
        rot_center = torch.tensor([int(round(cx)), int(round(cy))]).reshape(1,2).float()
        tensor = tensors[i, :, :, :].clone().detach().unsqueeze(0)
        mask = masks[i, :, :, :].clone().detach().unsqueeze(0)
        rot_degree = torch.tensor((shifts[i]/512)*360).float()
        rotated_tensor = tensor_rotate(tensor, -rot_degree, center=rot_center, mode='bilinear', padding_mode='border')
        rotated_tensors.append(rotated_tensor)
        rotated_mask = tensor_rotate(mask, -rot_degree, center=rot_center, mode='bilinear', padding_mode='zeros')
        rotated_masks.append(rotated_mask)
        rot_radian = (shifts[i]/512) * 2 * math.pi
        pupil_xyr_new[i, 0] = (pupil_xyr[i, 0] - cx) * math.cos(-rot_radian) + (pupil_xyr[i, 1] - cy) * math.sin(-rot_radian) + cx
        pupil_xyr_new[i, 1] = (pupil_xyr[i, 1] - cy) * math.cos(-rot_radian) - (pupil_xyr[i, 0] - cx) * math.sin(-rot_radian) + cy
        iris_xyr_new[i, 0] = (iris_xyr[i, 0] - cx) * math.cos(-rot_radian) + (iris_xyr[i, 1] - cy) * math.sin(-rot_radian) + cx
        iris_xyr_new[i, 1] = (iris_xyr[i, 1] - cy) * math.cos(-rot_radian) - (iris_xyr[i, 0] - cx) * math.sin(-rot_radian) + cy
    rotated_tensors = torch.cat(rotated_tensors, dim=0)
    rotated_masks = torch.cat(rotated_masks, dim=0)
    return rotated_tensors, rotated_masks, pupil_xyr_new, iris_xyr_new

def visualize_image(image, mask, pupil_xyr, iris_xyr):
    #mask = self.getMask(image)
    #pupil_xyr, iris_xyr = self.circApprox(image)
    imVis = np.stack((np.array(image),)*3, axis=-1)
    #print(imVis.shape)
    if mask is not None:
        try:
            imVis[:,:,1] = np.clip(imVis[:,:,1] + 0.25*np.array(mask),0,255)
        except:
            print("Mask could not be visualized.")
            pass
    try:
        imVis = cv2.circle(imVis, (int(pupil_xyr[0]),int(pupil_xyr[1])), int(pupil_xyr[2]), (0, 0, 255), 2)
    except:
        print("Pupil circle could not be visualized, values are: ", pupil_xyr)
        pass
    try:
        imVis = cv2.circle(imVis, (int(iris_xyr[0]),int(iris_xyr[1])), int(iris_xyr[2]), (255, 0, 0), 2)
    except:
        print("Iris circle could not be visualized, values are: ", iris_xyr)
        pass
    imVis_pil = Image.fromarray(imVis)
    
    return imVis_pil
    
def crop_iris(img_tensors, mask_tensors, pupil_xyr, iris_xyr, res=256):
    cropped_imgs = []
    cropped_masks = []
    #print(img_tensors.shape, mask_tensors.shape)
    pupil_xyr_new = pupil_xyr.clone().detach()
    iris_xyr_new = iris_xyr.clone().detach()

    b,c,h,w = img_tensors.shape

    for i in range(img_tensors.shape[0]):
        ix = int(round(iris_xyr[i, 0].item()))
        iy = int(round(iris_xyr[i, 1].item()))
        ir = iris_xyr[i, 2].item()

        ir_ratio = (res/2) / ((res/2) - (res/16))
        x_min = (ix - int(round(ir_ratio*ir)))
        x_max = (ix + int(round(ir_ratio*ir)))
        y_min = (iy - int(round(ir_ratio*ir)))
        y_max = (iy + int(round(ir_ratio*ir)))
        
        if x_min >= 0 and x_max < w and y_min >= 0 and y_min < h:
            cropped_img = img_tensors[i, :, y_min:y_max, x_min:x_max].unsqueeze(0)
            cropped_mask = mask_tensors[i, :, y_min:y_max, x_min:x_max].unsqueeze(0)
        else:
            if x_min < 0:
                if y_min < 0:
                    cropped_img = img_tensors[i, :, 0:y_max, 0:x_max].unsqueeze(0)
                    cropped_mask = mask_tensors[i, :, 0:y_max, 0:x_max].unsqueeze(0)
                    cropped_img = nn.functional.pad(cropped_img, (-x_min, 0, -y_min, 0), mode='replicate')
                    cropped_mask = nn.functional.pad(cropped_mask, (-x_min, 0, -y_min, 0), mode='constant', value=0)
                elif y_max >= h:
                    cropped_img = img_tensors[i, :, y_min:h, 0:x_max].unsqueeze(0)
                    cropped_mask = mask_tensors[i, :, y_min:h, 0:x_max].unsqueeze(0)
                    cropped_img = nn.functional.pad(cropped_img, (-x_min, 0, 0, (y_max - h)), mode='replicate')
                    cropped_mask = nn.functional.pad(cropped_mask, (-x_min, 0, 0, (y_max - h)), mode='constant', value=0)
                else:
                    cropped_img = img_tensors[i, :, y_min:y_max, 0:x_max].unsqueeze(0)
                    cropped_mask = mask_tensors[i, :, y_min:y_max, 0:x_max].unsqueeze(0)
                    cropped_img = nn.functional.pad(cropped_img, (-x_min, 0, 0, 0), mode='replicate')
                    cropped_mask = nn.functional.pad(cropped_mask, (-x_min, 0, 0, 0), mode='constant', value=0)
            elif x_max >= w:
                if y_min < 0:
                    cropped_img = img_tensors[i, :, 0:y_max, x_min:w].unsqueeze(0)
                    cropped_mask = mask_tensors[i, :, 0:y_max, x_min:w].unsqueeze(0)
                    cropped_img = nn.functional.pad(cropped_img, (0, (x_max - w), -y_min, 0), mode='replicate')
                    cropped_mask = nn.functional.pad(cropped_mask, (0, (x_max - w), -y_min, 0), mode='constant', value=0)
                elif y_max >= h:
                    cropped_img = img_tensors[i, :, y_min:h, x_min:w].unsqueeze(0)
                    cropped_mask = mask_tensors[i, :, y_min:h, x_min:w].unsqueeze(0)
                    cropped_img = nn.functional.pad(cropped_img, (0, (x_max - w), 0, (y_max - h)), mode='replicate')
                    cropped_mask = nn.functional.pad(cropped_mask, (0, (x_max - w), 0, (y_max - h)), mode='constant', value=0)
                else:
                    cropped_img = img_tensors[i, :, y_min:y_max, x_min:w].unsqueeze(0)
                    cropped_mask = mask_tensors[i, :, y_min:y_max, x_min:w].unsqueeze(0)
                    cropped_img = nn.functional.pad(cropped_img, (0, (x_max - w), 0, 0), mode='replicate')
                    cropped_mask = nn.functional.pad(cropped_mask, (0, (x_max - w), 0, 0), mode='constant', value=0)
            else:
                if y_min < 0:
                    cropped_img = img_tensors[i, :, 0:y_max, x_min:x_max].unsqueeze(0)
                    cropped_mask = mask_tensors[i, :, 0:y_max, x_min:x_max].unsqueeze(0)
                    cropped_img = nn.functional.pad(cropped_img, (0, 0, -y_min, 0), mode='replicate')
                    cropped_mask = nn.functional.pad(cropped_mask, (0, 0, -y_min, 0), mode='constant', value=0)
                elif y_max >= h:
                    cropped_img = img_tensors[i, :, y_min:h, x_min:x_max].unsqueeze(0)
                    cropped_mask = mask_tensors[i, :, y_min:h, x_min:x_max].unsqueeze(0)
                    cropped_img = nn.functional.pad(cropped_img, (0, 0, 0, (y_max - h)), mode='replicate')
                    cropped_mask = nn.functional.pad(cropped_mask, (0, 0, 0, (y_max - h)), mode='constant', value=0)
   

        cropped_img = nn.functional.interpolate(cropped_img, size=(res, res), mode='bilinear')
        cropped_mask = nn.functional.interpolate(cropped_mask, size=(res, res), mode='nearest') 
        
        mult = res/(ir_ratio*ir * 2)
        pupil_xyr_new[i, 0] = (res/2) - mult * (ir - (pupil_xyr[i, 0] - ix + ir))
        pupil_xyr_new[i, 1] = (res/2) - mult * (ir - (pupil_xyr[i, 1] - iy + ir))
        pupil_xyr_new[i, 2] = mult * pupil_xyr[i, 2]
        iris_xyr_new[i, 0] = res/2
        iris_xyr_new[i, 1] = res/2
        iris_xyr_new[i, 2] = ((res/2) - (res/16))
        
        #cropped_image_np = cropped_img[0,0,:,:].cpu().numpy()
        #normalized_cropped_image_np = (cropped_image_np - cropped_image_np.min()) / (cropped_image_np.max() - cropped_image_np.min())
        #normalized_cropped_image = Image.fromarray((normalized_cropped_image_np * 255).astype(np.uint8), "L")
        #cropped_mask_np = cropped_mask[0,0,:,:].cpu().numpy()
        #normalized_cropped_mask_np = (cropped_mask_np - cropped_mask_np.min()) / (cropped_mask_np.max() - cropped_mask_np.min())
        #normalized_cropped_mask = Image.fromarray((normalized_cropped_mask_np * 255).astype(np.uint8), "L")
        #visualize_image(normalized_cropped_image, normalized_cropped_mask, pupil_xyr_new[i].cpu().numpy(), iris_xyr_new[i].cpu().numpy()).save("./crop_debug/cropped_img_"+str(i)+".png")
        
        cropped_imgs.append(cropped_img)
        cropped_masks.append(cropped_mask)
    cropped_imgs = torch.cat(cropped_imgs, 0)
    cropped_masks = torch.cat(cropped_masks, 0)
    return cropped_imgs, cropped_masks, pupil_xyr_new, iris_xyr_new
    
def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg

def train(args):
    
    Tensor = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
    res=256
    
    if args.cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
  
    checkpoint_dir = './'+args.model_type+'_dnet_cp_' + datetime.datetime.now().strftime('%Y%m%d%H%M') 
    
    if args.sif_tanh:
        checkpoint_dir += '_sif_tanh'
    
    checkpoint_dir += '/'
    
    if not os.path.exists('./data_debug/'):
        os.mkdir('./data_debug/')
    
    if not os.path.exists('./debug_training' + args.tag + '/'):
        os.mkdir('./debug_training' + args.tag + '/')
    
    if not os.path.exists('./debug_validation' + args.tag + '/'):
        os.mkdir('./debug_validation' + args.tag + '/')
        
    if not os.path.exists('./debug_loss_check' + args.tag + '/'):
        os.mkdir('./debug_loss_check' + args.tag + '/')
        
    # Declare Dataloaders
    train_dataset = QuadrupletFromBinAllDatasetBothUncropped(args.train_bins_path_wsd, args.parent_dir_wsd, args.train_bins_path_csoispad, args.parent_dir_csoispad, flip_data=True, res_mult=args.res_mult)
    #print('Train identities: ', train_dataset.all_ids)
    train_dataloader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    
    if args.val_bins_path_wsd:
        if args.val_bins_path_csoispad:
            val_dataset = QuadrupletFromBinAllDatasetBothUncropped(args.val_bins_path_wsd, args.parent_dir_wsd, args.val_bins_path_csoispad, args.parent_dir_csoispad, flip_data=False, res_mult=args.res_mult)
            #print('Validation identities: ', val_dataset.all_ids)
            val_dataloader = DataLoader(val_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False, pin_memory=True)
        else:
            val_dataloader = None
    else:
        val_dataloader = None
    
    print('No. of training batches', len(train_dataloader))
    if val_dataloader is not None:
        print('No. of validation batches', len(val_dataloader))
    
    # Declare Models
    
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    #conv_map_net = models.resnet50(pretrained=False)
    #conv_map_net.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    #conv_map_net.fc = nn.Linear(2048, G.z_dim)
    #conv_map_net = conv_map_net.to(device)
    
    #conv_map_net = models.resnet18(pretrained=False)
    #conv_map_net.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    #conv_map_net.fc = DenseLinearNetwork(n_layers=7, input_dim=512, hidden_dim=512, output_dim=G.z_dim)
    
    if args.weight_path is not None:
        print('Weight path given, loading', args.weight_path)
        if not args.load_only_dnet:
            models = torch.load(args.weight_path)
            if args.load_ema:
                ema = models[0].to(device)
                deform_net = ema.online_model.to(device)
            else:
                deform_net = models[0].to(device)
            discriminator = models[1].to(device)
        else:
            deform_net = torch.load(args.weight_path, map_location=device)
    else:
        if args.no_ld_in:
            if args.model_type == 'nestedresunet':
                deform_net = NestedResUNetIN(num_classes=1, num_channels=2, width=args.width, resolution=(res, res))
            elif args.model_type == 'nestedresunetps':
                deform_net = NestedResUNetINPS(num_classes=1, num_channels=2, width=args.width, resolution=(res, res))
            elif args.model_type == 'nestedsharedatrousresunet':
                deform_net = NestedSharedAtrousResUNetIN(num_classes=1, num_channels=2, width=args.width, resolution=(res, res))
            elif args.model_type == 'nestedattentionresunet':
                deform_net = NestedAttentionResUNetIN(num_classes=1, num_channels=2, width=args.width, resolution=(res, res))
        else:
            if args.model_type == 'nestedresunet':
                deform_net = NestedResUNetIN(num_classes=1, num_channels=3, width=args.width, resolution=(res, res))
            elif args.model_type == 'nestedresunetps':
                deform_net = NestedResUNetINPS(num_classes=1, num_channels=3, width=args.width, resolution=(res, res))
            elif args.model_type == 'nestedsharedatrousresunet':
                deform_net = NestedSharedAtrousResUNetIN(num_classes=1, num_channels=3, width=args.width, resolution=(res, res))
            elif args.model_type == 'nestedattentionresunet':
                deform_net = NestedAttentionResUNetIN(num_classes=1, num_channels=3, width=args.width, resolution=(res, res))
        
            
        if args.use_patch_adv_loss:
            if args.disc_model_type == 'vectordiscriminator':
                discriminator = VectorAndPatchDiscriminatorIN().to(device, non_blocking=True)
            elif args.disc_model_type == 'vectordiscriminatorv2':
                discriminator = VectorAndPatchDiscriminatorINv2().to(device, non_blocking=True)
            elif args.disc_model_type == 'sharedatrousvectordiscriminator':
                discriminator = SharedAtrousVectorDiscriminatorIN().to(device, non_blocking=True)
        
    
    #if args.no_ld_in and not args.model_wo_ld_in:
    #    deform_net.conv0_0 = ResBlockIN(2, args.width, args.width)
        

    if args.use_patch_adv_loss:
        criterion_GAN = nn.SmoothL1Loss()
        criterion_vec = CosineLoss()            
            
    print(deform_net)
    
    if args.multi_gpu:
        deform_net = nn.DataParallel(deform_net)
    
    deform_net = deform_net.to(device, non_blocking=True)
    
    if args.ema and (not args.load_ema):
        ema = EMA(deform_net, beta=0.999).to(device)

    if args.use_tv_loss:
        tvLossModel = TVLoss().to(device, non_blocking=True)
    
    if args.use_vgg_loss:
        vggLossModel = VGGPerceptualLoss(device, non_blocking=True)
    
    if args.use_lpips_loss:    
        lpipsLossModel = LPIPSLoss(device, net_type=args.lpips_net_type)
    
    if args.use_dists_loss:
        DISTSModel = DISTS(device).to(device, non_blocking=True)
    
    if args.use_msssim_loss:
        MSSSIMModel = MS_SSIM_Loss(size_average=False).to(device, non_blocking=True)
    
    if args.use_nn_id_loss:
        NNIdentityModel = NNIdentityLoss(stem_width=64, network_path=args.nn_id_path, device=device).to(device, non_blocking=True)
    
    if args.use_iso_loss:
        SharpLoss = ISOSharpnessLoss(device).to(device, non_blocking=True)
        
    if args.use_patch_adv_loss:
        if args.disc_model_type == 'vectordiscriminator':
            discriminator = VectorAndPatchDiscriminatorIN().to(device, non_blocking=True)
        elif args.disc_model_type == 'vectordiscriminatorv2':
            discriminator = VectorAndPatchDiscriminatorINv2().to(device, non_blocking=True)
        elif args.disc_model_type == 'sharedatrousvectordiscriminator':
            discriminator = SharedAtrousVectorDiscriminatorIN().to(device, non_blocking=True)
        criterion_GAN = nn.SmoothL1Loss(reduction='none', beta=0.1)
        criterion_vec = CosineLoss()
        patch = (1, 64 // 2 ** 4, 512 // 2 ** 4)
        

    filter_mat1 = io.loadmat(args.sif_filter_path1)['ICAtextureFilters']
    filter_mat2 = io.loadmat(args.sif_filter_path2)['ICAtextureFilters']
    #sifLossModel = SIFLayerMaskOSIRIS(polar_height = 64, polar_width = 512, filter_mat1 = filter_mat1, filter_mat2 = filter_mat2, osiris_filters = args.osiris_filters, device=device).to(device)
    sifLossModel = SIFLayerMask(polar_height = 64, polar_width = 512, filter_mat1 = filter_mat1, filter_mat2 = filter_mat2, device=device).to(device)
    
    if args.optim_type == 'adam':
        print('Using Adam for Generator')
        optimizer = Adam(deform_net.parameters(), lr=args.lr, amsgrad=True)
    elif args.optim_type == 'adamw' or args.optim_type == 'clipped_adamw':
        print('Using AdamW with lr =', args.lr, ', weight decay=', args.weight_decay, 'for Generator.')
        optimizer = AdamW(deform_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim_type == 'sgd':
        print('Using SGD with lr =', args.lr, ', momentum=', args.momentum, ', weight decay=', args.weight_decay, 'for Generator.')
        optimizer = SGD(deform_net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim_type == 'cyclic_lr':
        print('Using Cyclic LR - 1cycle policy with AdamW', 'for Generator.')
        optimizer = AdamW(deform_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = lr_scheduler.OneCycleLR(optimizer, 0.006, epochs=100, steps_per_epoch=len(train_dataloader))
    elif args.optim_type == 'clipped_sgd':
        print('Using Grad Norm Clipped SGD with lr =', args.lr, ', momentum=', args.momentum, ', weight decay=', args.weight_decay, 'for Generator.')
        optimizer = SGD(deform_net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim_type == 'madgrad' or args.optim_type == 'clipped_madgrad':
        print('Using MADGRAD', 'for Generator.')
        optimizer = madgrad.MADGRAD(deform_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim_type == 'prodigy' or args.optim_type == 'clipped_prodigy':
        print('Using Prodigy', 'for Generator.')
        optimizer = Prodigy(deform_net.parameters(), weight_decay=args.weight_decay)
        
    if args.use_patch_adv_loss:
        if args.optim_type_D == 'adam':
            print('Using Adam', 'for Discriminator.')
            optimizer_D = Adam(discriminator.parameters(), lr=args.d_lr, weight_decay=args.disc_weight_decay, amsgrad=True)
        elif args.optim_type_D == 'adamw' or args.optim_type_D == 'clipped_adamw':
            print('Using AdamW with lr =', args.d_lr, ', weight decay=', args.disc_weight_decay, 'for Discriminator.')
            optimizer_D = AdamW(discriminator.parameters(), lr=args.d_lr, weight_decay=args.disc_weight_decay)
        elif args.optim_type_D == 'sgd':
            print('Using SGD with lr =', args.d_lr, ', momentum=', args.momentum, ', weight decay=', args.disc_weight_decay, 'for Discriminator.')
            optimizer_D = SGD(discriminator.parameters(), lr=args.d_lr, momentum=args.momentum, weight_decay=args.disc_weight_decay)
        elif args.optim_type_D == 'cyclic_lr':
            print('Using Cyclic LR - 1cycle policy with AdamW', 'for Discriminator.')
            optimizer_D = AdamW(discriminator.parameters(), lr=args.d_lr, weight_decay=args.disc_weight_decay)
            scheduler_D = lr_scheduler.OneCycleLR(optimizer_D, 0.006, epochs=100, steps_per_epoch=len(train_dataloader))
        elif args.optim_type_D == 'clipped_sgd':
            print('Using Grad Norm Clipped SGD with lr =', args.d_lr, ', momentum=', args.momentum, ', weight decay=', args.disc_weight_decay, 'for Discriminator.')
            optimizer_D = SGD(discriminator.parameters(), lr=args.d_lr, momentum=args.momentum, weight_decay=args.disc_weight_decay)
        elif args.optim_type_D == 'madgrad' or args.optim_type == 'clipped_madgrad':
            print('Using MADGRAD', 'for Discriminator.')
            optimizer_D = madgrad.MADGRAD(discriminator.parameters(), lr=args.d_lr, weight_decay=args.disc_weight_decay)
        elif args.optim_type_D == 'prodigy' or args.optim_type == 'clipped_prodigy':
            print('Using Prodigy', 'for Discriminator.')
            optimizer_D = Prodigy(discriminator.parameters(), args.disc_weight_decay)
        optimizer_D.zero_grad(set_to_none=True)
        
    optimizer.zero_grad(set_to_none=True)
    #print(len(optimizer.param_groups))
    
    best_val_bit_diff = 100
    best_val_loss = float('inf')
    print('Starting Training...')
    sample_no = 0
    
    if args.sif_hinge:
        hinge = HingeLoss(device, p=1)
        soft_hinge = HingeLossWithSoftLabels(device, label_smoothing=0.2)
    
    unsaved = 0
    
    debug = True
    
    out_weight = args.deform_mult
    
    zero_loss = Variable(torch.Tensor([0])).to(device)
    
    if not args.no_ld_in:
        deformer = LinearDeformer(device)
    
    noise_weight = 0.2
    
    if args.disc_weight_decay > 0.001:
        change_d_decay = True
    else:
        change_d_decay = False
        
    smooth_l1 = nn.SmoothL1Loss(reduction='none', beta=0.1)
     
    discriminator.train()   
    for epoch in range(1, args.num_epochs+1):
        loss_polar_check = True
        deform_net.train()
        epoch_loss = []
        epoch_mse_loss = []
        epoch_mask_loss = []
        epoch_sif_loss = []
        epoch_tv_loss = []
        epoch_percept_loss = []
        epoch_lpips_loss = []
        epoch_sif_diff = []
        epoch_sif_bit_count = []
        epoch_linear_sif_diff = []
        epoch_linear_sif_bit_count = []
        epoch_disc_loss = []
        epoch_dists_loss = []
        epoch_msssim_loss = []
        epoch_nn_id_loss = []
        epoch_iso_loss = []
        epoch_D_loss = []
        epoch_G_loss = []

        train_dataloader.dataset.reset()
        
        vb_count = 0
        vb_loss = 0
        vb_batch = -1

        if epoch % 5 == 0:
            if noise_weight > 0:
                noise_weight /= 2
                if noise_weight < 0.001:
                    noise_weight = 0
            if change_d_decay:
                if args.disc_weight_decay > 0.001:
                    args.disc_weight_decay = args.disc_weight_decay/10
                    for i in range(len(optimizer_D.param_groups)):
                        optimizer_D.param_groups[i]['weight_decay'] = args.disc_weight_decay
                else:
                    args.weight_decay = 0.001
                    for i in range(len(optimizer_D.param_groups)):
                        optimizer_D.param_groups[i]['weight_decay'] = args.disc_weight_decay
                    change_d_decay = False 
        
        rot_debug = True
        
        n_before_step = 0

        for batch, data in enumerate(train_dataloader):
            if args.only_val:
                break
            inp_imgs = data['inp_img']
            inp_masks = data['inp_mask']
            
            inp_img_pxyr = data['inp_img_pxyr']
            inp_img_ixyr = data['inp_img_ixyr']
            
            tar_imgs = data['tar_img']
            tar_masks = data['tar_mask']
                
            tar_img_pxyr = data['tar_img_pxyr']
            tar_img_ixyr = data['tar_img_ixyr']
            
            tar2_imgs = data['tar2_img']
            tar2_masks = data['tar2_mask']
                
            tar2_img_pxyr = data['tar2_img_pxyr']
            tar2_img_ixyr = data['tar2_img_ixyr']
            
            imp_imgs = data['imp_img']
            imp_masks = data['imp_mask']
            
            imp_img_pxyr = data['imp_img_pxyr']
            imp_img_ixyr = data['imp_img_ixyr']

            with torch.no_grad():
                inp_norm = inp_imgs.float().requires_grad_(False)
                inp_polar_iris_norm, inp_mask_polar_iris = sifLossModel.cartToPolIrisCenter(inp_norm, ((inp_masks+1)/2), inp_img_pxyr, inp_img_ixyr)
                sif_inp_iris = sifLossModel.getCodes(inp_polar_iris_norm.to(device, non_blocking=True))
                
                tar_norm = tar_imgs.float().requires_grad_(False)
                tar_polar_unrot_norm, tar_mask_polar_unrot = sifLossModel.cartToPolIrisCenter(tar_norm, ((tar_masks+1)/2), tar_img_pxyr, tar_img_ixyr)
                sif_tar_unrot = sifLossModel.getCodes(tar_polar_unrot_norm.to(device, non_blocking=True))
                
                tar2_norm = tar2_imgs.float().requires_grad_(False)
                tar2_polar_unrot_norm, tar2_mask_polar_unrot = sifLossModel.cartToPolIrisCenter(tar2_norm, ((tar2_masks+1)/2), tar2_img_pxyr, tar2_img_ixyr)
                sif_tar2_unrot = sifLossModel.getCodes(tar2_polar_unrot_norm.to(device, non_blocking=True))
                
                imp_norm = imp_imgs.float().requires_grad_(False)       
                imp_polar_unrot_norm, imp_mask_polar_unrot = sifLossModel.cartToPolIrisCenter(imp_norm, ((imp_masks+1)/2), imp_img_pxyr, imp_img_ixyr)
                sif_imp_unrot = sifLossModel.getCodes(imp_polar_unrot_norm.to(device, non_blocking=True))
                
                # data alignment, cropping and move to gpu
                shift_tar = calculate_shift(sif_inp_iris, inp_mask_polar_iris.to(device, non_blocking=True), sif_tar_unrot, tar_mask_polar_unrot.to(device, non_blocking=True))
                shift_tar2 = calculate_shift(sif_inp_iris, inp_mask_polar_iris.to(device, non_blocking=True), sif_tar2_unrot, tar2_mask_polar_unrot.to(device, non_blocking=True))
                shift_imp = calculate_shift(sif_inp_iris, inp_mask_polar_iris.to(device, non_blocking=True), sif_imp_unrot, imp_mask_polar_unrot.to(device, non_blocking=True))
                
                del inp_mask_polar_iris, inp_polar_iris_norm, sif_inp_iris
                del tar_mask_polar_unrot, tar_polar_unrot_norm, sif_tar_unrot
                del tar2_mask_polar_unrot, tar2_polar_unrot_norm, sif_tar2_unrot
                del imp_mask_polar_unrot, imp_polar_unrot_norm, sif_imp_unrot
                
                inp_norm, inp_masks, inp_img_pxyr, inp_img_ixyr = crop_iris(inp_norm, inp_masks, inp_img_pxyr, inp_img_ixyr)
                inp_img_means = torch.mean(inp_norm, dim=(1,2,3)).view(-1, 1, 1, 1)
                inp_img_stds = torch.std(inp_norm, dim=(1,2,3)).view(-1, 1, 1, 1)
                inp = torch.clamp(torch.nan_to_num(((inp_norm - inp_img_means) / inp_img_stds), nan=4, posinf=4, neginf=-4), -4, 4).to(device)
            
                inp = inp.to(device, non_blocking=True)
                inp_norm = inp_norm.to(device, non_blocking=True)
                inp_masks = inp_masks.to(device, non_blocking=True)
                inp_img_pxyr = inp_img_pxyr.to(device, non_blocking=True)
                inp_img_ixyr = inp_img_ixyr.to(device, non_blocking=True)
                inp_img_means = inp_img_means.to(device, non_blocking=True)
                inp_img_stds = inp_img_stds.to(device, non_blocking=True)
                inp_polar, inp_mask_polar = sifLossModel.cartToPol(inp, ((inp_masks+1)/2), inp_img_pxyr, inp_img_ixyr)
                inp_polar_norm = torch.clamp(((inp_polar * inp_img_stds) + inp_img_means), 0, 255)
                sif_inp = sifLossModel.getCodes(inp_polar_norm)
                
                tar_norm, tar_masks, tar_img_pxyr, tar_img_ixyr = rotate_tensor_batch(tar_norm, tar_masks, shift_tar, tar_img_pxyr, tar_img_ixyr)
                tar_norm, tar_masks, tar_img_pxyr, tar_img_ixyr = crop_iris(tar_norm, tar_masks, tar_img_pxyr, tar_img_ixyr)
                tar_img_means = torch.mean(tar_norm, dim=(1,2,3)).view(-1, 1, 1, 1)
                tar_img_stds = torch.std(tar_norm, dim=(1,2,3)).view(-1, 1, 1, 1)
                tar = torch.clamp(torch.nan_to_num(((tar_norm - tar_img_means) / tar_img_stds), nan=4, posinf=4, neginf=-4), -4, 4).to(device)
                
                tar = tar.to(device, non_blocking=True)
                tar_norm = tar_norm.to(device, non_blocking=True)
                tar_masks = tar_masks.to(device, non_blocking=True)
                tar_img_pxyr = tar_img_pxyr.to(device, non_blocking=True)
                tar_img_ixyr = tar_img_ixyr.to(device, non_blocking=True)
                tar_img_means = tar_img_means.to(device, non_blocking=True)
                tar_img_stds = tar_img_stds.to(device, non_blocking=True)
                tar_polar, tar_mask_polar = sifLossModel.cartToPol(tar, ((tar_masks+1)/2), tar_img_pxyr, tar_img_ixyr)
                tar_polar_norm = torch.clamp(((tar_polar * tar_img_stds) + tar_img_means), 0, 255)
                sif_tar = sifLossModel.getCodes(tar_polar_norm)
                
                tar2_norm, tar2_masks, tar2_img_pxyr, tar2_img_ixyr = rotate_tensor_batch(tar2_norm, tar2_masks, shift_tar2, tar2_img_pxyr, tar2_img_ixyr)
                tar2_norm, tar2_masks, tar2_img_pxyr, tar2_img_ixyr = crop_iris(tar2_norm, tar2_masks, tar2_img_pxyr, tar2_img_ixyr)
                tar2_img_means = torch.mean(tar2_norm, dim=(1,2,3)).view(-1, 1, 1, 1)
                tar2_img_stds = torch.std(tar2_norm, dim=(1,2,3)).view(-1, 1, 1, 1)
                tar2 = torch.clamp(torch.nan_to_num(((tar2_norm - tar2_img_means) / tar2_img_stds), nan=4, posinf=4, neginf=-4), -4, 4).to(device)
                
                tar2 = tar2.to(device, non_blocking=True)
                tar2_norm = tar2_norm.to(device, non_blocking=True)
                tar2_masks = tar2_masks.to(device, non_blocking=True)
                tar2_img_pxyr = tar2_img_pxyr.to(device, non_blocking=True)
                tar2_img_ixyr = tar2_img_ixyr.to(device, non_blocking=True)
                tar2_img_means = tar2_img_means.to(device, non_blocking=True)
                tar2_img_stds = tar2_img_stds.to(device, non_blocking=True)
                tar2_polar, tar2_mask_polar = sifLossModel.cartToPol(tar2, ((tar2_masks+1)/2), tar2_img_pxyr, tar2_img_ixyr)
                tar2_polar_norm = torch.clamp(((tar2_polar * tar2_img_stds) + tar2_img_means), 0, 255)
                sif_tar2 = sifLossModel.getCodes(tar2_polar_norm)
                
                imp_norm, imp_masks, imp_img_pxyr, imp_img_ixyr = rotate_tensor_batch(imp_norm, imp_masks, shift_imp, imp_img_pxyr, imp_img_ixyr)
                imp_norm, imp_masks, imp_img_pxyr, imp_img_ixyr = crop_iris(imp_norm, imp_masks, imp_img_pxyr, imp_img_ixyr)
                imp_img_means = torch.mean(imp_norm, dim=(1,2,3)).view(-1, 1, 1, 1)
                imp_img_stds = torch.std(imp_norm, dim=(1,2,3)).view(-1, 1, 1, 1)
                imp = torch.clamp(torch.nan_to_num(((imp_norm - imp_img_means) / imp_img_stds), nan=4, posinf=4, neginf=-4), -4, 4).to(device)
                
                imp = imp.to(device, non_blocking=True)
                imp_norm = imp_norm.to(device, non_blocking=True)
                imp_masks = imp_masks.to(device, non_blocking=True)
                imp_img_pxyr = imp_img_pxyr.to(device, non_blocking=True)
                imp_img_ixyr = imp_img_ixyr.to(device, non_blocking=True)
                imp_img_means = imp_img_means.to(device, non_blocking=True)
                imp_img_stds = imp_img_stds.to(device, non_blocking=True)
                imp_polar, imp_mask_polar = sifLossModel.cartToPol(imp, ((imp_masks+1)/2), imp_img_pxyr, imp_img_ixyr)
                imp_polar_norm = torch.clamp(((imp_polar * imp_img_stds) + imp_img_means), 0, 255)
                sif_imp = sifLossModel.getCodes(imp_polar_norm)
                
                tar_norm_imp = torch.clamp(((tar * imp_img_stds) + imp_img_means), 0, 255)
                tar_polar_norm_imp = torch.clamp(((tar_polar * imp_img_stds) + imp_img_means), 0, 255)
                sif_tar_imp = sifLossModel.getCodes(tar_polar_norm_imp)
                
                # data alignment, cropping and move to gpu end
                
                if not args.no_ld_in:
                    target_alpha = (tar_img_pxyr[:, 2]/tar_img_ixyr[:, 2]).reshape(-1, 1)
                
                out_tar_mask_polar = tar_mask_polar.detach().requires_grad_(False)
                out_tar_mask_polar = torch.cat([out_tar_mask_polar]*sif_tar.shape[1], dim=1).requires_grad_(False)
                
                out_imp_mask_polar = (tar_mask_polar * imp_mask_polar).requires_grad_(False)
                out_imp_mask_polar = torch.cat([out_imp_mask_polar]*sif_imp.shape[1], dim=1).requires_grad_(False)
                
                tar_imp_mask_polar = out_imp_mask_polar.clone().detach().requires_grad_(False)
                
                tar_inp_mask_polar = (tar_mask_polar * inp_mask_polar).requires_grad_(False)
                tar_inp_mask_polar = torch.cat([tar_inp_mask_polar]*sif_inp.shape[1], dim=1).requires_grad_(False)
                
                if not args.no_ld_in:
                    ld_in, _ = deformer.linear_deform(inp, inp_img_pxyr, inp_img_ixyr, target_alpha)
                    ld_in = ld_in.detach().requires_grad_(False).float()
                    ld_in = torch.clamp(torch.nan_to_num(ld_in, nan=4, posinf=4, neginf=-4), -4, 4)
                    ld_in_masks, _ = deformer.linear_deform(inp_masks.float().requires_grad_(False), inp_img_pxyr, inp_img_ixyr, target_alpha)
                    ld_in_masks = ld_in_masks.detach().requires_grad_(False).float()
                    ld_in_masks = torch.clamp(torch.nan_to_num(ld_in_masks, nan=1.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
                    ld_in_masks = torch.where(ld_in_masks < 0, -1.0, 1.0)
            
            if args.no_ld_in:
                out = deform_net(torch.cat([inp, tar_masks], 1))
                if args.model_type == 'stylegan3':
                    out = G.synthesis(out, noise_mode='const') 
            else:
                out = deform_net(torch.cat([inp, ld_in, tar_masks], 1))
                if args.model_type == 'stylegan3':
                    out = G.synthesis(out, noise_mode='const')
                
            out_polar, _ = sifLossModel.cartToPol(out, None, tar_img_pxyr, tar_img_ixyr)
            out_norm = ((out * tar_img_stds) + tar_img_means).to(device, non_blocking=True)
            out_norm_imp = ((out * imp_img_stds) + imp_img_means).to(device, non_blocking=True)
            
            out_polar_tar = ((out_polar * tar_img_stds) + tar_img_means).to(device, non_blocking=True)
            out_polar_imp = ((out_polar * imp_img_stds) + imp_img_means).to(device, non_blocking=True)
            
            sif_out_tar = sifLossModel.getCodes(out_polar_tar)
            sif_out_imp = sifLossModel.getCodes(out_polar_imp)
             
            # debug check  
            if batch < 2:
                for i in range(out.shape[0]):
                    inp_img = torch.clamp(inp_norm[i].clone().detach(), 0, 255).cpu().numpy()[0]
                    out_img = torch.clamp(out_norm[i].clone().detach(), 0, 255).cpu().numpy()[0]
                    tar_img = torch.clamp(tar_norm[i].clone().detach(), 0, 255).cpu().numpy()[0]
                    
                    inp_img_vis = np.stack((np.uint8(inp_img),)*3, axis=-1)
                    tar_img_vis = np.stack((np.uint8(tar_img),)*3, axis=-1)
                   
                    get_concat_h(Image.fromarray(np.uint8(inp_img_vis)), get_concat_h(Image.fromarray(np.uint8(out_img)).convert('RGB'), Image.fromarray(np.uint8(tar_img_vis)))).save('./debug_training' + args.tag + '/'+ str(batch) + '_' + str(i)+'.png')
            
            sif_out_tar_continuous_masked = sif_out_tar * out_tar_mask_polar
            sif_out_imp_continuous_masked = sif_out_imp * out_imp_mask_polar
            
            with torch.no_grad():
                sif_tar_out_continuous_masked = (sif_tar * out_tar_mask_polar).requires_grad_(False)
                sif_imp_out_continuous_masked = (sif_imp * out_imp_mask_polar).requires_grad_(False)
                sif_tar_imp_continuous_masked = (sif_tar_imp * tar_imp_mask_polar).requires_grad_(False)
                sif_imp_tar_continuous_masked = (sif_imp * tar_imp_mask_polar).requires_grad_(False)
                sif_inp_tar_continuous_masked = (sif_inp * tar_inp_mask_polar).requires_grad_(False)
                sif_tar_inp_continuous_masked = (sif_tar * tar_inp_mask_polar).requires_grad_(False)
            
            
            if args.sif_tanh:           
                sif_loss_tar = torch.mean(smooth_l1(torch.tanh(sif_out_tar_continuous_masked), torch.tanh(sif_tar_out_continuous_masked)), dim=(1,2,3))
                sif_loss_imp = torch.mean(smooth_l1(torch.tanh(sif_out_imp_continuous_masked), torch.tanh(sif_imp_out_continuous_masked)), dim=(1,2,3))
                with torch.no_grad():
                    sif_loss_margin = torch.mean(smooth_l1(torch.tanh(sif_imp_tar_continuous_masked), torch.tanh(sif_tar_imp_continuous_masked)), dim=(1,2,3))
                sif_loss = torch.mean(sif_loss_tar + torch.maximum(sif_loss_margin - sif_loss_imp, torch.zeros(sif_loss_tar.shape).to(device)))
            else:
                sif_loss_tar = torch.mean(smooth_l1(sif_out_tar_continuous_masked / 255.0, sif_tar_out_continuous_masked / 255.0), dim=(1,2,3))
                sif_loss_imp = torch.mean(smooth_l1(sif_out_imp_continuous_masked / 255.0, sif_imp_out_continuous_masked / 255.0), dim=(1,2,3))
                with torch.no_grad():
                    sif_loss_margin = torch.mean(smooth_l1(sif_imp_tar_continuous_masked / 255.0, sif_tar_imp_continuous_masked / 255.0), dim=(1,2,3))
                sif_loss = torch.mean(sif_loss_tar + torch.maximum(sif_loss_margin - sif_loss_imp, torch.zeros(sif_loss_tar.shape).to(device)))
                
            epoch_sif_loss.append(sif_loss.item())
            
            with torch.no_grad():
                sif_out_binary = torch.where(sif_out_tar_continuous_masked.cpu() > 0, 1.0, 0.0)
                sif_tar_binary = torch.where(sif_tar_out_continuous_masked.cpu() > 0, 1.0, 0.0)
                bit_count = torch.sum(out_tar_mask_polar.int()).item()
                
                sif_tar_inp_binary = torch.where(sif_tar_inp_continuous_masked.cpu() > 0, 1.0, 0.0)
                sif_inp_tar_binary = torch.where(sif_inp_tar_continuous_masked.cpu() > 0, 1.0, 0.0)
                inp_bit_count = torch.sum(tar_inp_mask_polar.int()).item()
                
            epoch_sif_diff.append(torch.sum(torch.abs(sif_out_binary - sif_tar_binary)).item())
            epoch_sif_bit_count.append(bit_count)
            epoch_linear_sif_diff.append(torch.sum(torch.abs(sif_tar_inp_binary - sif_inp_tar_binary)).item())
            epoch_linear_sif_bit_count.append(inp_bit_count)            
            
            if not args.no_direct_loss:
                mse_loss_tar = torch.mean(smooth_l1(out, tar.requires_grad_(False)), dim=(1,2,3))
                mse_loss_imp = torch.mean(smooth_l1(out, imp.requires_grad_(False)), dim=(1,2,3))
                with torch.no_grad():
                    mse_margin = torch.mean(smooth_l1(tar, imp), dim=(1,2,3))
                mse_loss = torch.mean(mse_loss_tar + torch.maximum(mse_margin - mse_loss_imp, torch.zeros(mse_loss_tar.shape).to(device)))
                epoch_mse_loss.append(mse_loss.item())
            else:
                mse_loss = 0.
            
            if args.use_lpips_loss:
                out_norm_lpips = torch.clamp((out_norm - 127.5) / 127.5, -1.0, 1.0)
                tar_norm_lpips = torch.clamp((tar_norm - 127.5) / 127.5, -1.0, 1.0)
                imp_norm_lpips = torch.clamp((imp_norm - 127.5) / 127.5, -1.0, 1.0)
                lpips_loss_tar = lpipsLossModel(out_norm_lpips, tar_norm_lpips.requires_grad_(False))
                lpips_loss_imp = lpipsLossModel(out_norm_lpips, imp_norm_lpips.requires_grad_(False))
                with torch.no_grad():
                    lpips_margin = lpipsLossModel(imp_norm_lpips.requires_grad_(False), tar_norm_lpips.requires_grad_(False))
                lpips_loss = torch.mean(lpips_loss_tar + torch.maximum(lpips_margin - lpips_loss_imp, torch.zeros(lpips_loss_imp.shape).to(device)))                
                epoch_lpips_loss.append(lpips_loss.item())               
            else:
                lpips_loss = 0.
            
            if args.use_tv_loss:
                tv_out = tvLossModel(out)
                tv_tar = tvLossModel(tar).detach().requires_grad_(False)
                tv_tar2 = tvLossModel(tar2).detach().requires_grad_(False)
                tv_max = torch.maximum(tv_tar, tv_tar2)
                tv_loss = torch.mean(torch.maximum(tv_out - tv_max, torch.zeros(tv_tar.shape).to(device)))
                epoch_tv_loss.append(tv_loss.item())
            else:
                tv_loss = 0.
            
            if args.use_dists_loss:
                dists_loss_tar = DISTSModel(out_norm, tar_norm, require_grad=True, batch_average=False)
                dists_loss_imp = DISTSModel(out_norm, imp_norm, require_grad=True, batch_average=False)
                with torch.no_grad():
                    dists_margin = DISTSModel(imp_norm, tar_norm, require_grad=False, batch_average=False)
                dists_loss = torch.mean(dists_loss_tar + torch.maximum(dists_margin - dists_loss_imp, torch.zeros(dists_loss_imp.shape).to(device)))
                epoch_dists_loss.append(dists_loss.item())
            else:
                dists_loss = 0.
                
            if args.use_nn_id_loss:
                normalize = Normalize(mean=(0.5,), std=(0.5,))
                out_polar_tar_inp = normalize(out_polar_tar/255)
                out_polar_imp_inp = normalize(out_polar_imp/255)
                tar_polar_inp = normalize(tar_polar/255).requires_grad_(False)
                tar_polar_imp_inp = normalize(tar_polar_imp/255).requires_grad_(False)
                imp_img_polar_inp = normalize(imp_img_polar/255).requires_grad_(False)
                nn_id_loss_tar = NNIdentityModel(out_polar_tar_inp, tar_polar_inp)
                nn_id_loss_imp = NNIdentityModel(out_polar_imp_inp, imp_img_polar_inp)
                with torch.no_grad():
                    nn_id_loss_margin = NNIdentityModel(tar_polar_imp_inp, imp_img_polar_inp)
                nn_id_loss = torch.mean(nn_id_loss_tar + torch.maximum(nn_id_loss_margin - nn_id_loss_imp, torch.zeros(nn_id_loss_tar.shape).to(device)))
                epoch_nn_id_loss.append(nn_id_loss.item())
            else:
                nn_id_loss = 0.
                
            if args.use_iso_loss:
                inp_sharpness = SharpLoss(inp_norm)
                out_sharpness = SharpLoss(out_norm)
                sharpness_loss = torch.mean(torch.maximum(inp_sharpness - out_sharpness, torch.zeros(inp_sharpness.shape).to(device)))
                epoch_iso_loss.append(sharpness_loss.item())
            else:
                sharpness_loss = 0.
            
            if args.use_patch_adv_loss:                
                '''
                if loss_polar_check:
                    with torch.no_grad():
                        out_polar_tar = ((out_polar * tar_img_stds) + tar_img_means).to(device, non_blocking=True)
                        tar_polar_norm = ((tar_polar * tar_img_stds) + tar_img_means).to(device, non_blocking=True)
                        tar2_polar_norm = ((tar2_polar * tar2_img_stds) + tar2_img_means).to(device, non_blocking=True)
                        imp_polar_norm = ((imp_polar * imp_img_stds) + imp_img_means).to(device, non_blocking=True)
                        
                        for i in range(out_polar_tar.shape[0]):
                            Image.fromarray(np.clip(out_polar_tar[i][0].cpu().numpy(), 0, 255).astype(np.uint8), 'L').save('./debug_loss_check' + args.tag + '/out_polar_tar_' + str(i) + '.png')
                            Image.fromarray(np.clip(tar_polar_norm[i][0].cpu().numpy(), 0, 255).astype(np.uint8), 'L').save('./debug_loss_check' + args.tag + '/tar_polar_norm_' + str(i) + '.png')
                            Image.fromarray(np.clip(tar2_polar_norm[i][0].cpu().numpy(), 0, 255).astype(np.uint8), 'L').save('./debug_loss_check' + args.tag + '/tar2_polar_norm_' + str(i) + '.png')
                            Image.fromarray(np.clip(imp_polar_norm[i][0].cpu().numpy(), 0, 255).astype(np.uint8), 'L').save('./debug_loss_check' + args.tag + '/imp_polar_norm_' + str(i) + '.png')
                        
                        loss_polar_check = False
                '''
                
                vec_out, rf_out = discriminator(out_polar)
                with torch.no_grad():
                    vec_inp, rf_inp = discriminator(inp_polar)
                    vec_tar, rf_tar = discriminator(tar_polar)
                    vec_tar2, rf_tar2 = discriminator(tar2_polar)
                    vec_imp, rf_imp = discriminator(imp_polar)
                    imp_tar_loss = criterion_vec(vec_tar, vec_imp)
                    imp_tar2_loss = criterion_vec(vec_tar2, vec_imp)
                    inp_tar_loss = criterion_vec(vec_inp, vec_tar)
                    inp_tar2_loss = criterion_vec(vec_inp, vec_tar2)
                    
                valid = torch.tensor(np.ones((rf_out.size(0), *patch))).requires_grad_(False).to(device)
                fake = torch.tensor(np.zeros((rf_out.size(0), *patch))).requires_grad_(False).to(device)
                
                G_loss = criterion_vec(vec_out, vec_tar) + criterion_vec(vec_out, vec_tar2) + torch.maximum(criterion_vec(vec_out, vec_inp) - inp_tar_loss, torch.zeros(inp_tar_loss.shape).to(device)) + torch.maximum(imp_tar_loss - criterion_vec(vec_out, vec_imp), torch.zeros(imp_tar_loss.shape).to(device))       
                G_loss += torch.mean(criterion_GAN(rf_out, valid), dim=(1,2,3))
                G_loss = torch.mean(G_loss)
                epoch_G_loss.append(G_loss.item() / args.virtual_batch_mult)
            else:
                G_loss = 0.
            
            if args.use_msssim_loss:
                msssim_loss_tar = MSSSIMModel(out_norm, tar_norm.requires_grad_(False))
                msssim_loss_imp = MSSSIMModel(out_norm_imp, imp_norm.requires_grad_(False))
                with torch.no_grad():
                    msssim_margin = MSSSIMModel(imp_norm.requires_grad_(False), tar_norm_imp.requires_grad_(False))
                msssim_loss = torch.mean(msssim_loss_tar + torch.maximum(msssim_margin - msssim_loss_imp, torch.zeros(msssim_loss_imp.shape).to(device)))
                epoch_msssim_loss.append(msssim_loss.item())
            else:
                msssim_loss = 0.
            
            loss = mse_loss + sif_loss + nn_id_loss + msssim_loss + dists_loss + lpips_loss + sharpness_loss + G_loss
            epoch_loss.append(loss.item())
            loss = loss / args.virtual_batch_mult 
            loss.backward()
            
            n_before_step += 1
            
            if n_before_step == args.virtual_batch_mult:
                if args.optim_type == 'clipped_sgd' or args.optim_type == 'clipped_adamw' or args.optim_type == 'clipped_madgrad' or args.optim_type == 'clipped_prodigy':
                    nn.utils.clip_grad_norm_(deform_net.parameters(), args.clip) 
                optimizer.step()               
                if args.optim_type == 'cyclic_lr':
                    scheduler.step()
                if args.ema:
                    ema.update()
                optimizer.zero_grad(set_to_none=True)
                n_before_step = 0
            
            if args.use_patch_adv_loss:
                inp_polar = inp_polar.clone().detach().requires_grad_()
                out_polar = out_polar.clone().detach().requires_grad_()
                tar_polar = tar_polar.clone().detach().requires_grad_()
                tar2_polar = tar2_polar.clone().detach().requires_grad_()
                imp_polar = imp_polar.clone().detach().requires_grad_()
                             
                if noise_weight > 0:
                    vec_inp, rf_inp = discriminator(inp_polar + noise_weight * torch.randn(inp_polar.shape).requires_grad_().to(device))
                    vec_out, rf_out = discriminator(out_polar + noise_weight * torch.randn(out_polar.shape).requires_grad_().to(device))
                    vec_tar, rf_tar = discriminator(tar_polar + noise_weight * torch.randn(tar_polar.shape).requires_grad_(False).to(device))
                    vec_tar2, rf_tar2 = discriminator(tar2_polar + noise_weight * torch.randn(tar2_polar.shape).requires_grad_(False).to(device))
                    vec_imp, rf_imp = discriminator(imp_polar + noise_weight * torch.randn(imp_polar.shape).requires_grad_(False).to(device))
                else:
                    vec_inp, rf_inp = discriminator(inp_polar)
                    vec_out, rf_out = discriminator(out_polar)
                    vec_tar, rf_tar = discriminator(tar_polar)
                    vec_tar2, rf_tar2 = discriminator(tar2_polar)
                    vec_imp, rf_imp = discriminator(imp_polar)
                
                valid = torch.tensor(np.ones((rf_out.size(0), *patch))).requires_grad_(False).to(device)
                fake = torch.tensor(np.zeros((rf_out.size(0), *patch))).requires_grad_(False).to(device)
                
                #vec_out, rf_out = discriminator(out_polar)
                with torch.no_grad():
                    imp_tar_loss = criterion_vec(vec_tar, vec_imp)
                    imp_tar2_loss = criterion_vec(vec_tar2, vec_imp)
                    max_imp_loss = torch.maximum(imp_tar_loss, imp_tar2_loss)
                #D_loss = torch.mean(torch.maximum(6 * criterion_vec(vec_tar, vec_tar2) - criterion_vec(vec_out, vec_inp) - criterion_vec(vec_out, vec_tar) - criterion_vec(vec_out, vec_tar2) - criterion_vec(vec_out, vec_imp) - criterion_vec(vec_tar, vec_imp) - criterion_vec(vec_tar2, vec_imp) + 5 * args.D_margin, torch.zeros(vec_tar.shape[0]).to(device)))                
                D_loss_triplet = torch.mean(torch.maximum(criterion_vec(vec_tar, vec_tar2) - criterion_vec(vec_tar, vec_imp) + args.D_margin, torch.zeros(vec_tar.shape[0]).to(device)))
                D_loss_triplet = D_loss_triplet / args.virtual_batch_mult
                if args.r1_reg_param > 0:
                    D_loss_triplet.backward(retain_graph=True)
                    reg_triplet = args.r1_reg_param * (compute_grad2(vec_tar, tar_polar).mean() + compute_grad2(vec_tar2, tar2_polar).mean() + compute_grad2(vec_imp, imp_polar).mean())
                    reg_triplet.backward(retain_graph=True)
                
                D_loss_advid = torch.maximum(max_imp_loss - criterion_vec(vec_out, vec_tar.clone().detach().requires_grad_(False)), torch.zeros(vec_tar.shape[0]).to(device))
                D_loss_advid += torch.maximum(max_imp_loss - criterion_vec(vec_out, vec_tar2.clone().detach().requires_grad_(False)), torch.zeros(vec_tar2.shape[0]).to(device))
                D_loss_advid += torch.maximum(max_imp_loss - criterion_vec(vec_out, vec_imp.clone().detach().requires_grad_(False)), torch.zeros(vec_imp.shape[0]).to(device))
                D_loss_advid = torch.mean(D_loss_advid)
                D_loss_advid = D_loss_advid / args.virtual_batch_mult
                if args.r1_reg_param > 0:
                    D_loss_advid.backward(retain_graph=True)
                    reg_advid = args.r1_reg_param * compute_grad2(vec_out, out_polar).mean()
                    reg_advid.backward(retain_graph=True)
                
                #D_loss += torch.maximum(criterion_vec(vec_out, vec_tar) - criterion_vec(vec_out, vec_imp) + args.D_margin, torch.zeros(vec_tar.shape[0]).to(device))
                #D_loss += torch.maximum(criterion_vec(vec_out, vec_tar2) - criterion_vec(vec_out, vec_imp) + args.D_margin, torch.zeros(vec_tar2.shape[0]).to(device))
                D_loss_real = torch.mean(criterion_GAN(rf_tar, valid) + criterion_GAN(rf_tar2, valid) + criterion_GAN(rf_imp, valid) + criterion_GAN(rf_inp, valid))
                D_loss_real = D_loss_real / args.virtual_batch_mult
                if args.r1_reg_param > 0:
                    D_loss_real.backward(retain_graph=True)
                    reg_real = args.r1_reg_param * (compute_grad2(rf_tar, tar_polar).mean() + compute_grad2(rf_tar2, tar2_polar).mean() + compute_grad2(rf_imp, imp_polar).mean() + compute_grad2(rf_inp, inp_polar).mean())
                    reg_real.backward(retain_graph=True)
                
                D_loss_fake = torch.mean(criterion_GAN(rf_out, fake))
                D_loss_fake = D_loss_fake / args.virtual_batch_mult
                if args.r1_reg_param > 0:
                    D_loss_fake.backward(retain_graph=True)
                    reg_fake = args.r1_reg_param * compute_grad2(rf_out, out_polar).mean()
                    reg_fake.backward(retain_graph=True)
                
                #total_loss = loss + D_loss
                #total_loss.backward()
                
                #D_loss = D_loss / args.virtual_batch_mult
                    
                if not args.r1_reg_param > 0:
                    D_loss = (D_loss_triplet + D_loss_advid + D_loss_real + D_loss_fake)
                    epoch_D_loss.append(D_loss.item())
                    D_loss.backward()
                else:
                    with torch.no_grad():
                        D_loss = (D_loss_triplet + D_loss_advid + D_loss_real + D_loss_fake)
                        epoch_D_loss.append(D_loss.item())
                
                
                
                if n_before_step == 0:
                    if args.optim_type_D == 'clipped_sgd' or args.optim_type_D == 'clipped_adamw' or args.optim_type_D == 'clipped_madgrad' or args.optim_type_D == 'clipped_prodigy':
                        nn.utils.clip_grad_norm_(discriminator.parameters(), args.D_clip)
                    optimizer_D.step()
                    if args.optim_type == 'cyclic_lr':
                        scheduler_D.step()
                    optimizer_D.zero_grad(set_to_none=True)
                    #if args.optim_type == 'clipped_sgd' or args.optim_type == 'clipped_adamw' or args.optim_type == 'clipped_madgrad' or args.optim_type == 'clipped_prodigy':
                    #    nn.utils.clip_grad_norm_(deform_net.parameters(), args.clip) 
                    #optimizer.step()               
                    #if args.optim_type == 'cyclic_lr':
                    #    scheduler.step()
                    #optimizer.zero_grad(set_to_none=True)
            
            if batch % args.log_batch == 0:
                train_loss_average = sum(epoch_loss) / len(epoch_loss)                    
                loss_string = "Train loss: {aver} (epoch: {epoch}, batch: {batch} / {total_batch})".format(aver = train_loss_average, epoch = epoch, batch = batch, total_batch=len(train_dataloader))
                
                if not args.no_direct_loss:
                    mse_loss_average = sum(epoch_mse_loss) / len(epoch_mse_loss)
                    loss_string += " Direct Triplet loss: {mse}".format(mse=mse_loss_average)
                
                if args.use_tv_loss:
                    tv_loss_average = sum(epoch_tv_loss) / len(epoch_tv_loss)
                    loss_string += ", TV loss: {tv}".format(tv = tv_loss_average)
                
                if args.use_dists_loss:
                    dists_loss_average = sum(epoch_dists_loss) / len(epoch_dists_loss)
                    loss_string += ", DISTS loss: {dists}".format(dists = dists_loss_average)
                
                if args.use_msssim_loss:
                    msssim_loss_average = sum(epoch_msssim_loss) / len(epoch_msssim_loss)
                    loss_string += ", MS-SSIM/SSIM Triplet loss: {msssim}".format(msssim = msssim_loss_average)
                
                if args.use_lpips_loss:
                    lpips_loss_average = sum(epoch_lpips_loss) / len(epoch_lpips_loss)
                    loss_string += ", LPIPS Triplet loss: {lpips}".format(lpips = lpips_loss_average)
                
                if args.use_nn_id_loss:
                    nn_id_loss_average = sum(epoch_nn_id_loss) / len(epoch_nn_id_loss)
                    loss_string += ", NN ID Triplet loss: {nn_id}".format(nn_id = nn_id_loss_average)
                
                if args.use_iso_loss:
                    iso_loss_average = sum(epoch_iso_loss) / len(epoch_iso_loss)
                    loss_string += ", ISO Sharpness loss: {sharp}".format(sharp = iso_loss_average)
                
                if args.use_patch_adv_loss:
                    G_loss_average = sum(epoch_G_loss) / len(epoch_G_loss)
                    loss_string += ", Generator loss: {g_loss}".format(g_loss = G_loss_average)
                    D_loss_average = sum(epoch_D_loss) / len(epoch_D_loss)
                    loss_string += ", Discriminator loss: {d_loss}".format(d_loss = D_loss_average)
                    
                sif_loss_average = sum(epoch_sif_loss) / len(epoch_sif_loss)
                loss_string += ", SIF Triplet loss: {sif}".format(sif = sif_loss_average)
                sif_bit_average = (sum(epoch_sif_diff) / sum(epoch_sif_bit_count)) * 100
                loss_string += ", SIF bit match: {sif_bit}%".format(sif_bit = (100 - sif_bit_average))
                sif_linear_bit_average = (sum(epoch_linear_sif_diff) / sum(epoch_linear_sif_bit_count)) * 100
                loss_string += ", Linear deform SIF bit match: {sif_bit}%".format(sif_bit = (100 - sif_linear_bit_average))
                if args.model_type == 'multiresolutionnestednet':
                    loss_string += ", Stage : {stage}".format(stage = stage)
                '''
                if sample_no < 10:
                    inp_img_t = interpolate((inp_imgs + 1)/2, size=(240, 320), mode='bilinear')
                    tar_img_t = interpolate((tar_imgs + 1)/2, size=(240, 320), mode='bilinear')
                    inp_mask_t = interpolate((inp_masks + 1)/2, size=(240, 320), mode='nearest')
                    tar_mask_t = interpolate((tar_masks + 1)/2, size=(240, 320), mode='nearest')
                    
                    for b in range(inp_img_t.shape[0]):
                    
                        s_img = img_transform(inp_img_t[b])
                        b_img = img_transform(tar_img_t[b])
                        s_mask = img_as_bool((inp_mask_t[b][0].clone().detach().cpu().numpy() * 255).astype(np.uint8))
                        b_mask = img_as_bool((tar_mask_t[b][0].clone().detach().cpu().numpy() * 255).astype(np.uint8))
                        
                        s_pxyr = np.around(inp_img_pxyr[b].clone().detach().cpu().numpy()).astype(np.intc)
                        s_ixyr = np.around(inp_img_ixyr[b].clone().detach().cpu().numpy()).astype(np.intc)
                        b_pxyr = np.around(tar_img_pxyr[b].clone().detach().cpu().numpy()).astype(np.intc)
                        b_ixyr = np.around(tar_img_ixyr[b].clone().detach().cpu().numpy()).astype(np.intc)
                        
                        s_img_c = s_img.convert('RGB')
                        b_img_c = b_img.convert('RGB')
                        
                        s_draw = ImageDraw.Draw(s_img_c)
                        s_draw.ellipse((s_pxyr[0]-s_pxyr[2], s_pxyr[1]-s_pxyr[2], s_pxyr[0]+s_pxyr[2], s_pxyr[1]+s_pxyr[2]), outline ="red")
                        s_draw.ellipse((s_ixyr[0]-s_ixyr[2], s_ixyr[1]-s_ixyr[2], s_ixyr[0]+s_ixyr[2], s_ixyr[1]+s_ixyr[2]), outline ="blue")
                        b_draw = ImageDraw.Draw(b_img_c)
                        b_draw.ellipse((b_pxyr[0]-b_pxyr[2], b_pxyr[1]-b_pxyr[2], b_pxyr[0]+b_pxyr[2], b_pxyr[1]+b_pxyr[2]), outline ="red")
                        b_draw.ellipse((b_ixyr[0]-b_ixyr[2], b_ixyr[1]-b_ixyr[2], b_ixyr[0]+b_ixyr[2], b_ixyr[1]+b_ixyr[2]), outline ="blue")
                        
                        all_imgs = get_concat_v(get_concat_v(s_img_c, Image.fromarray(s_mask.astype(np.uint8) * 255)), get_concat_v(b_img_c, Image.fromarray(b_mask.astype(np.uint8) * 255)))
                        all_imgs.save('samples/img_sample_'+str(batch)+'_'+str(b)+'.png')
                        
                    sample_no += 1
                '''   
                
                print(loss_string)
                if args.log_file is not None:
                    with open(args.log_file, 'a') as f:
                        f.write(loss_string + '\n')
                
                    
        if val_dataloader is not None:
            deform_net.eval()
            val_loss_average = []
            val_bit_diff_average = []
            val_linear_bit_diff_average = []
            save_count = 0
            with torch.inference_mode():
                for i in range(args.val_repeats):
                    val_dataloader.dataset.reset()
                    val_epoch_loss = []
                    val_bit_diff = []
                    val_bit_count = []
                    val_linear_bit_diff = []
                    val_linear_bit_count = []
                    for batch, data in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
                        inp_imgs = data['inp_img']
                        inp_masks = data['inp_mask']
                        
                        inp_img_pxyr = data['inp_img_pxyr']
                        inp_img_ixyr = data['inp_img_ixyr']
                        
                        tar_imgs = data['tar_img']
                        tar_masks = data['tar_mask']
                            
                        tar_img_pxyr = data['tar_img_pxyr']
                        tar_img_ixyr = data['tar_img_ixyr']
                        
                        tar2_imgs = data['tar2_img']
                        tar2_masks = data['tar2_mask']
                            
                        tar2_img_pxyr = data['tar2_img_pxyr']
                        tar2_img_ixyr = data['tar2_img_ixyr']
                        
                        imp_imgs = data['imp_img']
                        imp_masks = data['imp_mask']
                        
                        imp_img_pxyr = data['imp_img_pxyr']
                        imp_img_ixyr = data['imp_img_ixyr']
            
                        inp_norm = inp_imgs.float().requires_grad_(False)
                        inp_polar_iris_norm, inp_mask_polar_iris = sifLossModel.cartToPolIrisCenter(inp_norm, ((inp_masks+1)/2), inp_img_pxyr, inp_img_ixyr)
                        sif_inp_iris = sifLossModel.getCodes(inp_polar_iris_norm.to(device, non_blocking=True))
                        
                        tar_norm = tar_imgs.float().requires_grad_(False)
                        tar_polar_unrot_norm, tar_mask_polar_unrot = sifLossModel.cartToPolIrisCenter(tar_norm, ((tar_masks+1)/2), tar_img_pxyr, tar_img_ixyr)
                        sif_tar_unrot = sifLossModel.getCodes(tar_polar_unrot_norm.to(device, non_blocking=True))
                        
                        tar2_norm = tar2_imgs.float().requires_grad_(False)
                        tar2_polar_unrot_norm, tar2_mask_polar_unrot = sifLossModel.cartToPolIrisCenter(tar2_norm, ((tar2_masks+1)/2), tar2_img_pxyr, tar2_img_ixyr)
                        sif_tar2_unrot = sifLossModel.getCodes(tar2_polar_unrot_norm.to(device, non_blocking=True))
                        
                        imp_norm = imp_imgs.float().requires_grad_(False)       
                        imp_polar_unrot_norm, imp_mask_polar_unrot = sifLossModel.cartToPolIrisCenter(imp_norm, ((imp_masks+1)/2), imp_img_pxyr, imp_img_ixyr)
                        sif_imp_unrot = sifLossModel.getCodes(imp_polar_unrot_norm.to(device, non_blocking=True))
                        
                        # data alignment, cropping and move to gpu
                        shift_tar = calculate_shift(sif_inp_iris, inp_mask_polar_iris.to(device, non_blocking=True), sif_tar_unrot, tar_mask_polar_unrot.to(device, non_blocking=True))
                        shift_tar2 = calculate_shift(sif_inp_iris, inp_mask_polar_iris.to(device, non_blocking=True), sif_tar2_unrot, tar2_mask_polar_unrot.to(device, non_blocking=True))
                        shift_imp = calculate_shift(sif_inp_iris, inp_mask_polar_iris.to(device, non_blocking=True), sif_imp_unrot, imp_mask_polar_unrot.to(device, non_blocking=True))
                        
                        del inp_mask_polar_iris, inp_polar_iris_norm, sif_inp_iris
                        del tar_mask_polar_unrot, tar_polar_unrot_norm, sif_tar_unrot
                        del tar2_mask_polar_unrot, tar2_polar_unrot_norm, sif_tar2_unrot
                        del imp_mask_polar_unrot, imp_polar_unrot_norm, sif_imp_unrot
                        
                        inp_norm, inp_masks, inp_img_pxyr, inp_img_ixyr = crop_iris(inp_norm, inp_masks, inp_img_pxyr, inp_img_ixyr)
                        inp_img_means = torch.mean(inp_norm, dim=(1,2,3)).view(-1, 1, 1, 1)
                        inp_img_stds = torch.std(inp_norm, dim=(1,2,3)).view(-1, 1, 1, 1)
                        inp = torch.clamp(torch.nan_to_num(((inp_norm - inp_img_means) / inp_img_stds), nan=4, posinf=4, neginf=-4), -4, 4).to(device)
                    
                        inp = inp.to(device, non_blocking=True)
                        inp_norm = inp_norm.to(device, non_blocking=True)
                        inp_masks = inp_masks.to(device, non_blocking=True)
                        inp_img_pxyr = inp_img_pxyr.to(device, non_blocking=True)
                        inp_img_ixyr = inp_img_ixyr.to(device, non_blocking=True)
                        inp_img_means = inp_img_means.to(device, non_blocking=True)
                        inp_img_stds = inp_img_stds.to(device, non_blocking=True)
                        inp_polar, inp_mask_polar = sifLossModel.cartToPol(inp, ((inp_masks+1)/2), inp_img_pxyr, inp_img_ixyr)
                        inp_polar_norm = torch.clamp(((inp_polar * inp_img_stds) + inp_img_means), 0, 255)
                        sif_inp = sifLossModel.getCodes(inp_polar_norm)
                        
                        tar_norm, tar_masks, tar_img_pxyr, tar_img_ixyr = rotate_tensor_batch(tar_norm, tar_masks, shift_tar, tar_img_pxyr, tar_img_ixyr)
                        tar_norm, tar_masks, tar_img_pxyr, tar_img_ixyr = crop_iris(tar_norm, tar_masks, tar_img_pxyr, tar_img_ixyr)
                        tar_img_means = torch.mean(tar_norm, dim=(1,2,3)).view(-1, 1, 1, 1)
                        tar_img_stds = torch.std(tar_norm, dim=(1,2,3)).view(-1, 1, 1, 1)
                        tar = torch.clamp(torch.nan_to_num(((tar_norm - tar_img_means) / tar_img_stds), nan=4, posinf=4, neginf=-4), -4, 4).to(device)
                        
                        tar = tar.to(device, non_blocking=True)
                        tar_norm = tar_norm.to(device, non_blocking=True)
                        tar_masks = tar_masks.to(device, non_blocking=True)
                        tar_img_pxyr = tar_img_pxyr.to(device, non_blocking=True)
                        tar_img_ixyr = tar_img_ixyr.to(device, non_blocking=True)
                        tar_img_means = tar_img_means.to(device, non_blocking=True)
                        tar_img_stds = tar_img_stds.to(device, non_blocking=True)
                        tar_polar, tar_mask_polar = sifLossModel.cartToPol(tar, ((tar_masks+1)/2), tar_img_pxyr, tar_img_ixyr)
                        tar_polar_norm = torch.clamp(((tar_polar * tar_img_stds) + tar_img_means), 0, 255)
                        sif_tar = sifLossModel.getCodes(tar_polar_norm)
                        
                        tar2_norm, tar2_masks, tar2_img_pxyr, tar2_img_ixyr = rotate_tensor_batch(tar2_norm, tar2_masks, shift_tar2, tar2_img_pxyr, tar2_img_ixyr)
                        tar2_norm, tar2_masks, tar2_img_pxyr, tar2_img_ixyr = crop_iris(tar2_norm, tar2_masks, tar2_img_pxyr, tar2_img_ixyr)
                        tar2_img_means = torch.mean(tar2_norm, dim=(1,2,3)).view(-1, 1, 1, 1)
                        tar2_img_stds = torch.std(tar2_norm, dim=(1,2,3)).view(-1, 1, 1, 1)
                        tar2 = torch.clamp(torch.nan_to_num(((tar2_norm - tar2_img_means) / tar2_img_stds), nan=4, posinf=4, neginf=-4), -4, 4).to(device)
                        
                        tar2 = tar2.to(device, non_blocking=True)
                        tar2_norm = tar2_norm.to(device, non_blocking=True)
                        tar2_masks = tar2_masks.to(device, non_blocking=True)
                        tar2_img_pxyr = tar2_img_pxyr.to(device, non_blocking=True)
                        tar2_img_ixyr = tar2_img_ixyr.to(device, non_blocking=True)
                        tar2_img_means = tar2_img_means.to(device, non_blocking=True)
                        tar2_img_stds = tar2_img_stds.to(device, non_blocking=True)
                        tar2_polar, tar2_mask_polar = sifLossModel.cartToPol(tar2, ((tar2_masks+1)/2), tar2_img_pxyr, tar2_img_ixyr)
                        tar2_polar_norm = torch.clamp(((tar2_polar * tar2_img_stds) + tar2_img_means), 0, 255)
                        sif_tar2 = sifLossModel.getCodes(tar2_polar_norm)
                        
                        imp_norm, imp_masks, imp_img_pxyr, imp_img_ixyr = rotate_tensor_batch(imp_norm, imp_masks, shift_imp, imp_img_pxyr, imp_img_ixyr)
                        imp_norm, imp_masks, imp_img_pxyr, imp_img_ixyr = crop_iris(imp_norm, imp_masks, imp_img_pxyr, imp_img_ixyr)
                        imp_img_means = torch.mean(imp_norm, dim=(1,2,3)).view(-1, 1, 1, 1)
                        imp_img_stds = torch.std(imp_norm, dim=(1,2,3)).view(-1, 1, 1, 1)
                        imp = torch.clamp(torch.nan_to_num(((imp_norm - imp_img_means) / imp_img_stds), nan=4, posinf=4, neginf=-4), -4, 4).to(device)
                        
                        imp = imp.to(device, non_blocking=True)
                        imp_norm = imp_norm.to(device, non_blocking=True)
                        imp_masks = imp_masks.to(device, non_blocking=True)
                        imp_img_pxyr = imp_img_pxyr.to(device, non_blocking=True)
                        imp_img_ixyr = imp_img_ixyr.to(device, non_blocking=True)
                        imp_img_means = imp_img_means.to(device, non_blocking=True)
                        imp_img_stds = imp_img_stds.to(device, non_blocking=True)
                        imp_polar, imp_mask_polar = sifLossModel.cartToPol(imp, ((imp_masks+1)/2), imp_img_pxyr, imp_img_ixyr)
                        imp_polar_norm = torch.clamp(((imp_polar * imp_img_stds) + imp_img_means), 0, 255)
                        sif_imp = sifLossModel.getCodes(imp_polar_norm)
                        
                        tar_norm_imp = torch.clamp(((tar * imp_img_stds) + imp_img_means), 0, 255)
                        tar_polar_norm_imp = torch.clamp(((tar_polar * imp_img_stds) + imp_img_means), 0, 255)
                        sif_tar_imp = sifLossModel.getCodes(tar_polar_norm_imp)
                        
                        # data alignment, cropping and move to gpu end
                        
                        if not args.no_ld_in)
                            target_alpha = (tar_img_pxyr[:, 2]/tar_img_ixyr[:, 2]).reshape(-1, 1)
                        
                        out_tar_mask_polar = tar_mask_polar.detach()
                        out_tar_mask_polar = torch.cat([out_tar_mask_polar]*sif_tar.shape[1], dim=1)
                        
                        out_imp_mask_polar = (tar_mask_polar * imp_mask_polar)
                        out_imp_mask_polar = torch.cat([out_imp_mask_polar]*sif_imp.shape[1], dim=1)
                        
                        tar_imp_mask_polar = out_imp_mask_polar.clone().detach()
                        
                        tar_inp_mask_polar = (tar_mask_polar * inp_mask_polar)
                        tar_inp_mask_polar = torch.cat([tar_inp_mask_polar]*sif_inp.shape[1], dim=1)
                        
                        if not args.no_ld_in:
                            ld_in, _ = deformer.linear_deform(inp, inp_img_pxyr, inp_img_ixyr, target_alpha)
                            ld_in = ld_in.detach().float()
                            ld_in = torch.clamp(torch.nan_to_num(ld_in, nan=4, posinf=4, neginf=-4), -4, 4)
                            ld_in_masks, _ = deformer.linear_deform(inp_masks.float().requires_grad_(False), inp_img_pxyr, inp_img_ixyr, target_alpha)
                            ld_in_masks = ld_in_masks.detach().float()
                            ld_in_masks = torch.clamp(torch.nan_to_num(ld_in_masks, nan=1.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
                            ld_in_masks = torch.where(ld_in_masks < 0, -1.0, 1.0)
                        
                        if args.ema:  
                            if args.no_ld_in:
                                out = ema(torch.cat([inp, tar_masks], 1))
                                if args.model_type == 'stylegan3':
                                    out = G.synthesis(out, noise_mode='const') 
                            else:
                                out = ema(torch.cat([inp, ld_in, tar_masks], 1))
                                if args.model_type == 'stylegan3':
                                    out = G.synthesis(out, noise_mode='const')
                        else:
                            if args.no_ld_in:
                                out = deform_net(torch.cat([inp, tar_masks], 1))
                                if args.model_type == 'stylegan3':
                                    out = G.synthesis(out, noise_mode='const') 
                            else:
                                out = deform_net(torch.cat([inp, ld_in, tar_masks], 1))
                                if args.model_type == 'stylegan3':
                                    out = G.synthesis(out, noise_mode='const')
                            
                        out_polar, _ = sifLossModel.cartToPol(out, None, tar_img_pxyr, tar_img_ixyr)
                        out_norm = torch.clamp(((out * tar_img_stds) + tar_img_means), 0, 255).to(device, non_blocking=True)
                        out_norm_imp = torch.clamp(((out * imp_img_stds) + imp_img_means), 0, 255).to(device, non_blocking=True)
                        
                        out_polar_tar = torch.clamp(((out_polar * tar_img_stds) + tar_img_means), 0, 255).to(device, non_blocking=True)
                        out_polar_imp = torch.clamp(((out_polar * imp_img_stds) + imp_img_means), 0, 255).to(device, non_blocking=True)
                        
                        sif_out_tar = sifLossModel.getCodes(out_polar_tar)
                        sif_out_imp = sifLossModel.getCodes(out_polar_imp)
                         
                        # debug check  
                        if batch < 2:
                            for i in range(out.shape[0]):
                                inp_img = torch.clamp(inp_norm[i].clone().detach(), 0, 255).cpu().numpy()[0]
                                out_img = torch.clamp(out_norm[i].clone().detach(), 0, 255).cpu().numpy()[0]
                                tar_img = torch.clamp(tar_norm[i].clone().detach(), 0, 255).cpu().numpy()[0]
                                
                                inp_img_vis = np.stack((np.uint8(inp_img),)*3, axis=-1)
                                tar_img_vis = np.stack((np.uint8(tar_img),)*3, axis=-1)
                               
                                get_concat_h(Image.fromarray(np.uint8(inp_img_vis)), get_concat_h(Image.fromarray(np.uint8(out_img)).convert('RGB'), Image.fromarray(np.uint8(tar_img_vis)))).save('./debug_validation' + args.tag + '/'+ str(batch) + '_' + str(i)+'.png')
                        
                        sif_out_tar_continuous_masked = sif_out_tar * out_tar_mask_polar
                        sif_out_imp_continuous_masked = sif_out_imp * out_imp_mask_polar
                        
                        sif_tar_out_continuous_masked = (sif_tar * out_tar_mask_polar)
                        sif_imp_out_continuous_masked = (sif_imp * out_imp_mask_polar)
                        sif_tar_imp_continuous_masked = (sif_tar_imp * tar_imp_mask_polar)
                        sif_imp_tar_continuous_masked = (sif_imp * tar_imp_mask_polar)
                        sif_inp_tar_continuous_masked = (sif_inp * tar_inp_mask_polar)
                        sif_tar_inp_continuous_masked = (sif_tar * tar_inp_mask_polar)
                        
                        if args.sif_tanh:           
                            sif_loss_tar = torch.mean(smooth_l1(torch.tanh(sif_out_tar_continuous_masked), torch.tanh(sif_tar_out_continuous_masked)), dim=(1,2,3))
                            sif_loss_imp = torch.mean(smooth_l1(torch.tanh(sif_out_imp_continuous_masked), torch.tanh(sif_imp_out_continuous_masked)), dim=(1,2,3))
                            with torch.no_grad():
                                sif_loss_margin = torch.mean(smooth_l1(torch.tanh(sif_imp_tar_continuous_masked), torch.tanh(sif_tar_imp_continuous_masked)), dim=(1,2,3))
                            sif_loss = torch.mean(sif_loss_tar + torch.maximum(sif_loss_margin - sif_loss_imp, torch.zeros(sif_loss_tar.shape).to(device)))
                        else:
                            sif_loss_tar = torch.mean(smooth_l1(sif_out_tar_continuous_masked / 255.0, sif_tar_out_continuous_masked / 255.0), dim=(1,2,3))
                            sif_loss_imp = torch.mean(smooth_l1(sif_out_imp_continuous_masked / 255.0, sif_imp_out_continuous_masked / 255.0), dim=(1,2,3))
                            with torch.no_grad():
                                sif_loss_margin = torch.mean(smooth_l1(sif_imp_tar_continuous_masked / 255.0, sif_tar_imp_continuous_masked / 255.0), dim=(1,2,3))
                            sif_loss = torch.mean(sif_loss_tar + torch.maximum(sif_loss_margin - sif_loss_imp, torch.zeros(sif_loss_tar.shape).to(device)))
                        

                        sif_out_binary = torch.where(sif_out_tar_continuous_masked.cpu() > 0, 1.0, 0.0)
                        sif_tar_binary = torch.where(sif_tar_out_continuous_masked.cpu() > 0, 1.0, 0.0)
                        bit_count = torch.sum(out_tar_mask_polar.int()).item()
                        
                        sif_tar_inp_binary = torch.where(sif_tar_inp_continuous_masked.cpu() > 0, 1.0, 0.0)
                        sif_inp_tar_binary = torch.where(sif_inp_tar_continuous_masked.cpu() > 0, 1.0, 0.0)
                        inp_bit_count = torch.sum(tar_inp_mask_polar.int()).item()
                            
                        val_bit_diff.append(torch.sum(torch.abs(sif_out_binary - sif_tar_binary)).item())
                        val_bit_count.append(bit_count)
                        val_linear_bit_diff.append(torch.sum(torch.abs(sif_tar_inp_binary - sif_inp_tar_binary)).item())
                        val_linear_bit_count.append(inp_bit_count)
                        
                        if not args.no_direct_loss:
                            mse_loss_tar = torch.mean(smooth_l1(out, tar.requires_grad_(False)), dim=(1,2,3))
                            mse_loss_imp = torch.mean(smooth_l1(out, imp.requires_grad_(False)), dim=(1,2,3))
                            with torch.no_grad():
                                mse_margin = torch.mean(smooth_l1(tar, imp), dim=(1,2,3))
                            mse_loss = torch.mean(mse_loss_tar + torch.maximum(mse_margin - mse_loss_imp, torch.zeros(mse_loss_tar.shape).to(device)))
                        else:
                            mse_loss = 0.
                        
                        if args.use_lpips_loss:
                            out_norm_lpips = torch.clamp((out_norm - 127.5) / 127.5, -1.0, 1.0)
                            tar_norm_lpips = torch.clamp((tar_norm - 127.5) / 127.5, -1.0, 1.0)
                            imp_norm_lpips = torch.clamp((imp_norm - 127.5) / 127.5, -1.0, 1.0)
                            lpips_loss_tar = lpipsLossModel(out_norm_lpips, tar_norm_lpips.requires_grad_(False))
                            lpips_loss_imp = lpipsLossModel(out_norm_lpips, imp_norm_lpips.requires_grad_(False))
                            with torch.no_grad():
                                lpips_margin = lpipsLossModel(imp_norm_lpips.requires_grad_(False), tar_norm_lpips.requires_grad_(False))
                            lpips_loss = torch.mean(lpips_loss_tar + torch.maximum(lpips_margin - lpips_loss_imp, torch.zeros(lpips_loss_imp.shape).to(device)))                           
                        else:
                            lpips_loss = 0.
                        
                        if args.use_tv_loss:
                            tv_out = tvLossModel(out)
                            tv_tar = tvLossModel(tar).detach().requires_grad_(False)
                            tv_tar2 = tvLossModel(tar2).detach().requires_grad_(False)
                            tv_max = torch.maximum(tv_tar, tv_tar2)
                            tv_loss = torch.mean(torch.maximum(tv_out - tv_max, torch.zeros(tv_tar.shape).to(device)))
                        else:
                            tv_loss = 0.
                        
                        if args.use_dists_loss:
                            dists_loss_tar = DISTSModel(out_norm, tar_norm, require_grad=True, batch_average=False)
                            dists_loss_imp = DISTSModel(out_norm, imp_norm, require_grad=True, batch_average=False)
                            with torch.no_grad():
                                dists_margin = DISTSModel(imp_norm, tar_norm, require_grad=False, batch_average=False)
                            dists_loss = torch.mean(dists_loss_tar + torch.maximum(dists_margin - dists_loss_imp, torch.zeros(dists_loss_imp.shape).to(device)))
                        else:
                            dists_loss = 0.
                            
                        if args.use_nn_id_loss:
                            normalize = Normalize(mean=(0.5,), std=(0.5,))
                            out_polar_tar_inp = normalize(out_polar_tar/255)
                            out_polar_imp_inp = normalize(out_polar_imp/255)
                            tar_polar_inp = normalize(tar_polar/255).requires_grad_(False)
                            tar_polar_imp_inp = normalize(tar_polar_imp/255).requires_grad_(False)
                            imp_img_polar_inp = normalize(imp_img_polar/255).requires_grad_(False)
                            nn_id_loss_tar = NNIdentityModel(out_polar_tar_inp, tar_polar_inp)
                            nn_id_loss_imp = NNIdentityModel(out_polar_imp_inp, imp_img_polar_inp)
                            with torch.no_grad():
                                nn_id_loss_margin = NNIdentityModel(tar_polar_imp_inp, imp_img_polar_inp)
                            nn_id_loss = torch.mean(nn_id_loss_tar + torch.maximum(nn_id_loss_margin - nn_id_loss_imp, torch.zeros(nn_id_loss_tar.shape).to(device)))
                        else:
                            nn_id_loss = 0.
                            
                        if args.use_iso_loss:
                            inp_sharpness = SharpLoss(inp_norm)
                            out_sharpness = SharpLoss(out_norm)
                            sharpness_loss = torch.mean(torch.maximum(inp_sharpness - out_sharpness, torch.zeros(inp_sharpness.shape).to(device)))
                        else:
                            sharpness_loss = 0.
                        
                        if args.use_msssim_loss:
                            msssim_loss_tar = MSSSIMModel(out_norm, tar_norm.requires_grad_(False))
                            msssim_loss_imp = MSSSIMModel(out_norm_imp, imp_norm.requires_grad_(False))
                            with torch.no_grad():
                                msssim_margin = MSSSIMModel(imp_norm.requires_grad_(False), tar_norm_imp.requires_grad_(False))
                            msssim_loss = torch.mean(msssim_loss_tar + torch.maximum(msssim_margin - msssim_loss_imp, torch.zeros(msssim_loss_imp.shape).to(device)))
                        else:
                            msssim_loss = 0.
                            
                        if args.use_tv_loss:
                            tv_out = tvLossModel(out)
                            tv_tar = tvLossModel(tar).detach().requires_grad_(False)
                            tv_tar2 = tvLossModel(tar2).detach().requires_grad_(False)
                            tv_max = torch.maximum(tv_tar, tv_tar2)
                            tv_loss = torch.mean(torch.maximum(tv_out - tv_max, torch.zeros(tv_tar.shape).to(device)))
                        else:
                            tv_loss = 0.
                        
                        loss = mse_loss + sif_loss + nn_id_loss + msssim_loss + dists_loss + lpips_loss + sharpness_loss + tv_loss                    
                        val_epoch_loss.append(loss.item())
                        
                    val_loss_average.append(sum(val_epoch_loss) / len(val_epoch_loss))                     
                    val_bit_diff_average.append(sum(val_bit_diff) * 100 / sum(val_bit_count))
                    val_linear_bit_diff_average.append(sum(val_linear_bit_diff) * 100 / sum(val_linear_bit_count))
                
                val_loss_mean = np.mean(np.array(val_loss_average))
                val_loss_std = np.std(np.array(val_loss_average))
                
                val_bit_match_mean = np.mean(100 - np.array(val_bit_diff_average))
                val_bit_match_std = np.std(100 - np.array(val_bit_diff_average))
                val_linear_bit_match_mean = np.mean(100 - np.array(val_linear_bit_diff_average))
                val_linear_bit_match_std = np.std(100 - np.array(val_linear_bit_diff_average))
                
                print("{epoch:04}: val_loss: {loss}+-{loss_std} val_bit_match: {bit_match}+-{bit_std} val_linear_bit_match: {lin_bit_match}+-{lin_bit_std}".format(epoch = epoch, loss=val_loss_mean, loss_std = val_loss_std, bit_match = val_bit_match_mean, bit_std = val_bit_match_std,  lin_bit_match = val_linear_bit_match_mean, lin_bit_std = val_linear_bit_match_std))                    
                
                if val_loss_mean < best_val_loss:
                    best_val_loss = val_loss_mean
                    if not os.path.exists(checkpoint_dir):
                        os.mkdir(checkpoint_dir)
                    if args.model_type == 'multiresolutionnestednet':
                        filename = checkpoint_dir + "{epoch:04}-val_loss-{loss}+-{loss_std}-val_bit_match-{bit_match}+-{bit_std}-val_linear_bit_match-{lin_bit_match}+-{lin_bit_std}-stage-{stage}.pth".format(epoch = epoch, loss=val_loss_mean, loss_std = val_loss_std, bit_match = val_bit_match_mean, bit_std = val_bit_match_std,  lin_bit_match = val_linear_bit_match_mean, lin_bit_std = val_linear_bit_match_std, stage=stage)
                    else:
                        filename = checkpoint_dir + "{epoch:04}-val_loss-{loss}+-{loss_std}-val_bit_match-{bit_match}+-{bit_std}-val_linear_bit_match-{lin_bit_match}+-{lin_bit_std}.pth".format(epoch = epoch, loss=val_loss_mean, loss_std = val_loss_std, bit_match = val_bit_match_mean, bit_std = val_bit_match_std,  lin_bit_match = val_linear_bit_match_mean, lin_bit_std = val_linear_bit_match_std)
                    
                    if args.multi_gpu:
                        models = [deform_net.module]
                        if args.use_patch_adv_loss:
                            models.append(discriminator.module)
                    else:
                        if args.ema:
                            models = [ema]
                        else:
                            models = [deform_net]
                        if args.use_patch_adv_loss:
                            models.append(discriminator)
                            
                    torch.save(models, filename)
                    if args.only_val:
                        return True
                    
    return True       
        
if __name__ == '__main__':
    #print('Its running')
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--parent_dir_wsd', type=str, default='')
    parser.add_argument('--train_bins_path_wsd', type=str, default='')
    parser.add_argument('--val_bins_path_wsd', type=str, default='')
    parser.add_argument('--test_bins_path_wsd', type=str, default='')
    parser.add_argument('--parent_dir_csoispad', type=str, default='')
    parser.add_argument('--train_bins_path_csoispad', type=str, default='')
    parser.add_argument('--val_bins_path_csoispad', type=str, default='')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--sif_filter_path1', type=str, default='./filters/ICAtextureFilters_17x17_5bit.mat')
    parser.add_argument('--sif_filter_path2', type=str, default='./filters/ICAtextureFilters_15x15_7bit.mat')
    parser.add_argument('--osiris_filters', type=str, default='./filters/osiris_filters.txt')
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--d_lr', type=float, default=0.0001)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--log_batch', type=int, default=50)
    parser.add_argument('--weight_path', type=str, default=None)
    parser.add_argument('--cfg_path', type=str, default="./configs/cfg.yaml", help="path of the iris recognition module configuration file.")
    parser.add_argument('--optim_type', type=str, default='clipped_prodigy')
    parser.add_argument('--optim_type_D', type=str, default='adamw')
    parser.add_argument("--clip", type=float, default=0.4, help="Clipping Gradients. Default=0.4")
    parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
    parser.add_argument("--weight_decay", "--wd", default=0, type=float, help="Weight decay for the autoencoder, Default: 0")
    parser.add_argument("--step", type=int, default=5, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
    parser.add_argument('--use_tv_loss', action='store_true')
    parser.add_argument('--use_vgg_loss', action='store_true')
    parser.add_argument('--use_lpips_loss', action='store_true')
    parser.add_argument('--res_mult', type=int, default=1)
    parser.add_argument('--model_type', type=str, default='nestedresunetmask')
    parser.add_argument('--max_pairs_inp', type=int, default=5)
    parser.add_argument('--dynamic_batch', type=int, default=5, help='Double the batch size if validation accuracy doesnt decrease for <dynamic_batch> epoch')
    parser.add_argument('--max_batch_size',type=int, default=512)
    parser.add_argument('--lpips_net_type', type=str, default='alex')
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--cudnn', action='store_true')
    parser.add_argument('--no_direct_loss', action='store_true')
    parser.add_argument('--use_msssim_loss', action='store_true')
    parser.add_argument('--use_dists_loss', action='store_true')
    parser.add_argument('--use_cycle_loss', action='store_true')
    parser.add_argument('--sif_hinge', action='store_true')
    parser.add_argument('--sif_msssim', action='store_true')
    parser.add_argument('--sif_tanh', action='store_true')
    parser.add_argument('--sif_sigmoid', action='store_true')
    parser.add_argument('--sif_bce', action='store_true')
    parser.add_argument('--sif_label_smoothing', action='store_true')
    parser.add_argument('--log_file', type=str, default=None)
    parser.add_argument('--val_repeats', type=int, default=1)
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--deform_mult', type=float, default=0.6)
    parser.add_argument('--max_deform_mult', type=float, default=0.6)
    parser.add_argument('--only_val', action='store_true')
    parser.add_argument('--use_nn_id_loss', action='store_true')
    parser.add_argument('--nn_id_path', type=str, default="")
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--use_iso_loss', action='store_true')
    parser.add_argument('--use_patch_adv_loss', action='store_true')
    parser.add_argument('--disc_weight_decay', type=float, default=0.1, help="Weight decay for the discriminator, Default: 0.1")
    parser.add_argument('--D_margin', type=float, default=0.1)
    parser.add_argument('--disc_model_type', type=str, default='vectordiscriminator')
    parser.add_argument('--load_only_dnet', action='store_true')
    parser.add_argument('--no_ld_in', action='store_true')
    parser.add_argument('--virtual_batch_mult', type=int, default=1)
    parser.add_argument('--D_clip', type=float, default=1.0)
    parser.add_argument('--ema', action='store_true')
    parser.add_argument('--load_ema', action='store_true')
    parser.add_argument('--r1_reg_param', type=float, default=10.)
    parser.add_argument('--model_wo_ld_in', action='store_true')
    
    args = parser.parse_args()
    
    if args.cudnn:
        print('Using CUDNN')
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
          
    if not args.sample:
        train(args)
    else:
        sample(args)
    
    sys.stdout.close()
