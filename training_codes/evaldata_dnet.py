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
from torchvision.transforms import Normalize, ToTensor
from torch.nn.functional import interpolate
import torch.nn.functional as F
from torch.autograd import Function 

from kornia.geometry.transform import translate as tensor_translate
from kornia.geometry.transform import rotate as tensor_rotate

from network import *

from scipy import io
import datetime
import os
from typing import List, Optional, Tuple, Union
from modules.utils import get_cfg
from modules.irisRecognition import irisRecognition

from torchvision import models, transforms

from tqdm import tqdm

import math
from math import pi

import math
import sys
from DISTS_pytorch import DISTS
import shutil
import madgrad
import random
from PIL import Image, ImageDraw, ImageFont, ImageChops

from torch.autograd import Variable

import pickle as pkl
from biomech import Biomech

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst
    
def minmaximage(t):
    t_min, t_max = t.min(), t.max()
    t_p = ((t - t_min)/(t_max - t_min))*255
    return t_p

def process_image_for_input(image_inp, mask_inp, pxyr_inp, ixyr_inp, input_size=256):
    ir_ratio = (input_size/2) / ((input_size/2) - (input_size/16))
    image = image_inp.copy()
    mask = mask_inp.copy()
    pxyr = np.copy(pxyr_inp)
    ixyr = np.copy(ixyr_inp)
    image = image.crop((ixyr[0] - int(round(ir_ratio*ixyr[2])), ixyr[1] - int(round(ir_ratio*ixyr[2])), ixyr[0] + int(round(ir_ratio*ixyr[2])), ixyr[1] + int(round(ir_ratio*ixyr[2]))))
    mask = mask.crop((ixyr[0] - int(round(ir_ratio*ixyr[2])), ixyr[1] - int(round(ir_ratio*ixyr[2])), ixyr[0] + int(round(ir_ratio*ixyr[2])), ixyr[1] + int(round(ir_ratio*ixyr[2]))))
    w_crop, h_crop = image.size
    pxyr[0] = pxyr[0] - ixyr[0] + w_crop/2
    pxyr[1] = pxyr[1] - ixyr[1] + h_crop/2
    ixyr[0] = w_crop/2
    ixyr[1] = h_crop/2
    h_mult = input_size/h_crop
    w_mult = input_size/w_crop
    image = image.resize((input_size, input_size))
    mask = mask.resize((input_size, input_size))
    pxyr[0] = input_size/2 - w_mult * (w_crop/2 - pxyr[0])
    pxyr[1] = input_size/2 - h_mult * (h_crop/2 - pxyr[1])
    pxyr[2] = max(w_mult, h_mult) * pxyr[2]
    ixyr[0] = input_size/2
    ixyr[1] = input_size/2
    ixyr[2] = max(w_mult, h_mult) * ixyr[2]
    return image, mask, pxyr, ixyr

    
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
    
def polarize(image, mask, pxyr, ixyr, sifLossModel, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5, ))
    ])
    inp_t = torch.from_numpy(np.float32(image)).float().unsqueeze(0).unsqueeze(0).to(device)
    mask_t = transform(mask).unsqueeze(0).to(device)
    mask_t[mask_t<0] = 0
    mask_t[mask_t>=0] = 1
    
    pxyr = torch.tensor(pxyr).unsqueeze(0).float().to(device)
    ixyr = torch.tensor(ixyr).unsqueeze(0).float().to(device)
    
    inp_polar_t, mask_polar_t = sifLossModel.cartToPol(inp_t, mask_t, pxyr, ixyr)
    
    inp_polar_np = torch.clamp(inp_polar_t[0].clone().detach(), 0, 255).cpu().numpy()[0]
    inp_polar_image = Image.fromarray(np.uint8(inp_polar_np))
    
    mask_polar_np = torch.where(mask_polar_t[0].clone().detach() < 0.5, 0, 1).cpu().numpy()[0]
    mask_polar_image = Image.fromarray(np.uint8(mask_polar_np * 255))
    
    return inp_polar_image, mask_polar_image

def deform_v1(image, target_mask, deform_net, device): #Update to remove mean and std and add it later, input processing is wrong here
    def img_transform(img_tensor):
        img_t = img_tensor[0]
        img_t = np.clip(img_t.clone().detach().cpu().numpy() * 255, 0, 255)
        img = Image.fromarray(img_t.astype(np.uint8))
        return img
    tensor_transform = transforms.Compose([
        transforms.Resize(size=(240, 320)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
    ])
    with torch.inference_mode():
        s_img_t = tensor_transform(image).unsqueeze(0)
        b_mask_t = tensor_transform(target_mask).unsqueeze(0)

        inp = Variable(torch.cat([s_img_t, b_mask_t], dim=1)).to(device)
        
        out = deform_net(inp)

        out_im = img_transform((out[0]+1)/2)
        
        return out_im
        
        
    
def deform(image, image_mask, image_pxyr, image_ixyr, target_image, target_mask, target_pxyr, target_ixyr, sifLossModel, deformer, deform_net, model_type, device, G=None): #Update to remove mean and std and add it later, input processing is wrong here
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5, ))
    ])
    with torch.inference_mode():
        inp_t = torch.from_numpy(np.float32(image)).float()
        inp_img_mean = torch.mean(inp_t)
        inp_img_std = torch.std(inp_t)
        
        xform_inp = torch.clamp(torch.nan_to_num((inp_t - inp_img_mean)/inp_img_std, nan=4, posinf=4, neginf=-4), -4, 4)
        xform_inp = xform_inp.unsqueeze(0).unsqueeze(0)
        xform_inp_mask = transform(image_mask)
        xform_inp_mask[xform_inp_mask<0] = -1.0
        xform_inp_mask[xform_inp_mask>=0] = 1.0
        xform_inp_mask = xform_inp_mask.unsqueeze(0)
        
        image_pxyr = torch.tensor(image_pxyr).unsqueeze(0).float()
        image_ixyr = torch.tensor(image_ixyr).unsqueeze(0).float()
        
        tar_t = torch.from_numpy(np.float32(target_image)).float()
        tar_img_mean = torch.mean(tar_t)
        tar_img_std = torch.std(tar_t)
        
        xform_tar = torch.clamp(torch.nan_to_num((tar_t - tar_img_mean)/tar_img_std, nan=4, posinf=4, neginf=-4), -4, 4)
        xform_tar = xform_tar.unsqueeze(0).unsqueeze(0)
        xform_tar_mask = transform(target_mask)
        xform_tar_mask[xform_tar_mask<0] = -1.0
        xform_tar_mask[xform_tar_mask>=0] = 1.0
        xform_tar_mask = xform_tar_mask.unsqueeze(0)
        
        target_pxyr = torch.tensor(target_pxyr).unsqueeze(0).float()
        target_ixyr = torch.tensor(target_ixyr).unsqueeze(0).float()
        
        inp_polar_iris, inp_mask_polar_iris = sifLossModel.cartToPolIrisCenter(inp_t.unsqueeze(0).unsqueeze(0), ((xform_inp_mask+1)/2), image_pxyr, image_ixyr)
        sif_inp_iris = sifLossModel.getCodesCPU(inp_polar_iris)
        
        tar_polar_unrot, tar_mask_polar_unrot = sifLossModel.cartToPolIrisCenter(tar_t.unsqueeze(0).unsqueeze(0), ((xform_tar_mask+1)/2), target_pxyr, target_ixyr)
        sif_tar_unrot = sifLossModel.getCodesCPU(tar_polar_unrot)
        
        shift_tar = calculate_shift(sif_inp_iris, inp_mask_polar_iris, sif_tar_unrot, tar_mask_polar_unrot)
        
        del inp_polar_iris, inp_mask_polar_iris, sif_inp_iris
        del tar_polar_unrot, tar_mask_polar_unrot, sif_tar_unrot
        
        xform_inp, xform_inp_mask, image_pxyr, image_ixyr = crop_iris(xform_inp.to(device), xform_inp_mask.to(device), image_pxyr.to(device), image_ixyr.to(device))
        xform_tar, xform_tar_mask, target_pxyr, target_ixyr = rotate_tensor_batch(xform_tar, xform_tar_mask, shift_tar, target_pxyr, target_ixyr)
        xform_tar, xform_tar_mask, target_pxyr, target_ixyr = crop_iris(xform_tar.to(device), xform_tar_mask.to(device), target_pxyr.to(device), target_ixyr.to(device))
        
        target_alpha = (target_pxyr[:, 2]/target_ixyr[:, 2]).reshape(-1, 1).to(device)
        ld_in, _ = deformer.linear_deform(xform_inp, image_pxyr, image_ixyr, target_alpha)
        ld_in = ld_in.detach().requires_grad_(False).float()
        ld_in = torch.clamp(torch.nan_to_num(ld_in, nan=4, posinf=4, neginf=-4), -4, 4)
        
        out = deform_net(torch.cat([xform_inp, ld_in, xform_tar_mask], 1))
        if model_type == 'stylegan3':
            out = G.synthesis(out, noise_mode='const')
        out_norm = ((out * inp_img_std) + inp_img_mean).to(device, non_blocking=True)
        out_image_np = torch.clamp(out_norm[0].clone().detach(), 0, 255).cpu().numpy()[0]
        out_image = Image.fromarray(np.uint8(out_image_np))
        
        out_mask = ((xform_tar_mask+1)/2)
        out_mask_np = torch.where(out_mask[0].clone().detach() < 0.5, 0, 1).cpu().numpy()[0]
        out_mask_image = Image.fromarray(np.uint8(out_mask_np * 255))
        
        out_polar, out_polar_mask = sifLossModel.cartToPol(out_norm, out_mask, target_pxyr, target_ixyr)
        
        out_polar_np = torch.clamp(out_polar[0].clone().detach(), 0, 255).cpu().numpy()[0]
        out_polar_image = Image.fromarray(np.uint8(out_polar_np))
        
        out_polar_mask_np = torch.where(out_polar_mask[0].clone().detach() < 0.5, 0, 1).cpu().numpy()[0]
        out_polar_mask_image = Image.fromarray(np.uint8(out_polar_mask_np * 255))
        
        return out_image, out_mask_image, out_polar_image, out_polar_mask_image

def swap(img1, mask1, pxyr1, ixyr1, seqid1, alpha1, img2, mask2, pxyr2, ixyr2, seqid2, alpha2):
    return img2, mask2, pxyr2, ixyr2, seqid2, alpha2, img1, mask1, pxyr1, ixyr1, seqid1, alpha1
    
def evaluate_graph_new_wbpd(args):
    folderpath = os.path.join(args.parent_dir_wsd, 'WarsawData2') + '/61/'
    dest_dir = './new_wbpd_eval/'
    
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
        os.mkdir(dest_dir + 'deformed_images_polar')
        os.mkdir(dest_dir + 'v1_deformed_images_polar')
        os.mkdir(dest_dir + 'biomech_images_polar')
        os.mkdir(dest_dir + 'biomech_masks_polar')
        os.mkdir(dest_dir + 'images_polar')
        
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    deform_net_v1 = torch.load('./0015-val_bit_diff-14.7574462890625.pth', map_location=device).to(device)
    deform_net_v1.eval()
    
    filter_mat1 = io.loadmat(args.sif_filter_path1)['ICAtextureFilters']
    filter_mat2 = io.loadmat(args.sif_filter_path2)['ICAtextureFilters']
    sifLossModel = SIFLayerMaskOSIRIS(polar_height = 64, polar_width = 512, filter_mat1 = filter_mat1, filter_mat2 = filter_mat2, osiris_filters = args.osiris_filters, device=device).to(device)
    
    deformer = LinearDeformer(device)
    
    if args.model_type == 'stylegan3':
        deform_net_v2 = torch.load(args.weight_path)[0].to(device)
        with dnnlib.util.open_url(args.sg3_weight) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    else:
        if args.load_ema:
            ema = torch.load(args.weight_path)[0]
            deform_net_v2 = ema.ema_model.to(device)
        else:
            deform_net_v2 = torch.load(args.weight_path)[0].to(device)
    deform_net_v2.eval()
    
    with open(os.path.join(args.parent_dir_wsd, 'pupil_iris_xyrs_new.pkl'), 'rb') as circleFile:
        pupil_iris_xyrs = pkl.load(circleFile)
    
    irisRec = irisRecognition(get_cfg(args.cfg), use_hough=False)
    
    alpha_image_pairs = []
    
    for imagename in os.listdir(folderpath):
        if imagename.startswith('61_P'):
            pr = pupil_iris_xyrs['61/'+imagename.split('.')[0]]['pxyr'][2]
            ir = pupil_iris_xyrs['61/'+imagename.split('.')[0]]['ixyr'][2]
            alpha = float(pr) / float(ir)
            alpha_image_pairs.append((alpha, '61/'+imagename.split('.')[0]))
    
    alpha_image_pairs.sort(reverse=True)
    
    img1name = alpha_image_pairs[0][1]
    alpha1 = alpha_image_pairs[0][0]
    img1 = Image.open(os.path.join(args.parent_dir_wsd, 'WarsawData2') + '/' + img1name + '.bmp')
    mask1 = Image.open(os.path.join(args.parent_dir_wsd, 'WarsawMasksCoarse') + '/' + img1name + '.bmp')
    pxyr1 = pupil_iris_xyrs[img1name]['pxyr']
    ixyr1 = pupil_iris_xyrs[img1name]['ixyr']
        
    shutil.copyfile(os.path.join(args.parent_dir_wsd, 'WarsawDataPolar') + '/' + img1name + '.bmp', dest_dir + img1name.split('/')[1] + '.bmp')
    
    for alpha_image_pair in tqdm(alpha_image_pairs):
        img2name = alpha_image_pair[1]
        shutil.copyfile(os.path.join(args.parent_dir_wsd, 'WarsawDataPolar') + '/' + img2name + '.bmp', dest_dir + 'images_polar/' + img2name.split('/')[1] + '.bmp')
        
        alpha2 = alpha_image_pair[0]
        img2 = Image.open(os.path.join(args.parent_dir_wsd, 'WarsawData2') + '/' + img2name + '.bmp')
        mask2 = Image.open(os.path.join(args.parent_dir_wsd, 'WarsawMasksCoarse') + '/' + img2name + '.bmp')
        pxyr2 = pupil_iris_xyrs[img2name]['pxyr']
        ixyr2 = pupil_iris_xyrs[img2name]['ixyr']
        
        v1_deformed_image = deform_v1(img1, mask2, deform_net_v1, device)
        v1_deformed_mask, v1_deformed_pxyr, v1_deformed_ixyr = irisRec.segment_and_circApprox(v1_deformed_image)
        v1_deformed_polar_image, v1_deformed_mask_polar = irisRec.cartToPol_torch(v1_deformed_image, v1_deformed_mask, v1_deformed_pxyr, v1_deformed_ixyr)
        
        Image.fromarray(v1_deformed_polar_image, 'L').save(dest_dir + 'v1_deformed_images_polar/' + img2name.split('/')[1] + '.bmp')
        
        deformed_image, deformed_mask, deformed_polar_image, deformed_polar_mask = deform(img1, mask1, pxyr1, ixyr1, img2, mask2, pxyr2, ixyr2, sifLossModel, deformer, deform_net_v2, args.model_type, device)
        
        deformed_polar_image.save(dest_dir + 'deformed_images_polar/' + img2name.split('/')[1] + '.bmp')
        
        bpr = float(alpha1 * 0.006)
        spr = float(alpha2 * 0.006)
        biomech_radii = np.float32(Biomech(spr, bpr, 65).flatten()[1:])
        
        bm_deformed_polar_image, bm_deformed_polar_mask = irisRec.bioMechCartToPol_torch(img2, mask2, pxyr2, ixyr2, biomech_radii)
        
        Image.fromarray(bm_deformed_polar_image, 'L').save(dest_dir + 'biomech_images_polar/' + img2name.split('/')[1] + '.bmp')     
        Image.fromarray(bm_deformed_polar_mask, 'L').save(dest_dir + 'biomech_masks_polar/' + img2name.split('/')[1] + '.bmp')  
                                
        
        

        
        
        
        
        
        
        
        
    
    
    
        

def evaluate_hollingsworth_deformirisnet(args):
    # Declare Models
    
    debug_constrict = 0
    debug_dilate = 0
    debug_total = 20
    
    if debug_total > 0:
        if not os.path.exists('./debug/'):
            os.mkdir('./debug/')
        else:
            if not os.path.exists('./debug/dilation/'):
                os.mkdir('./debug/dilation/')
            if not os.path.exists('./debug/constriction/'):
                os.mkdir('./debug/constriction/')
    
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    deform_net = torch.load('./0015-val_bit_diff-14.7574462890625.pth', map_location=device).to(device)
    deform_net.eval()
    
    with open(os.path.join(args.hollingsworth_dir, 'pupil_iris_xyrs.pkl'), 'rb') as circleFile:
        pupil_iris_xyrs = pkl.load(circleFile)
        
    irisRec = irisRecognition(get_cfg(args.cfg), use_hough=False)
    
    for pairFileName in tqdm(['genuine_pairs.txt', 'imposter_pairs.txt']):
        with open(os.path.join(args.hollingsworth_dir, 'v1_dnet_bs_' + pairFileName), 'w') as bsdngpFile:
            with open(os.path.join(args.hollingsworth_dir, 'v1_dnet_sb_' + pairFileName), 'w') as sbdngpFile:
                with open(os.path.join(args.hollingsworth_dir, pairFileName), 'r') as pFile: 
                    for line in tqdm(pFile):
                        lineparts = line.strip().split(',')
                        
                        seqid1 = lineparts[0]
                        seqid2 = lineparts[1]
                        
                        img1 = Image.open(os.path.join(os.path.join(args.hollingsworth_dir, 'recordings'), seqid1 + '.tiff')).convert('L')
                        img2 = Image.open(os.path.join(os.path.join(args.hollingsworth_dir, 'recordings'), seqid2 + '.tiff')).convert('L')
                        
                        mask1 = Image.open(os.path.join(os.path.join(args.hollingsworth_dir, 'masks_coarse'), seqid1 + '.tiff')).convert('L')
                        mask2 = Image.open(os.path.join(os.path.join(args.hollingsworth_dir, 'masks_coarse'), seqid2 + '.tiff')).convert('L')
                        
                        pxyr1 = np.copy(pupil_iris_xyrs[seqid1]['pxyr'])
                        ixyr1 = np.copy(pupil_iris_xyrs[seqid1]['ixyr'])
                        pxyr2 = np.copy(pupil_iris_xyrs[seqid2]['pxyr'])
                        ixyr2 = np.copy(pupil_iris_xyrs[seqid2]['ixyr'])
                        
                        #img1_polar, mask1_polar = polarize(img1, mask1, pxyr1, ixyr1, sifLossModel, device)
                        #img2_polar, mask2_polar = polarize(img2, mask2, pxyr2, ixyr2, sifLossModel, device)
                        
                        #img1_polar.save(os.path.join(os.path.join(args.hollingsworth_dir, 'images_polar'), seqid1 + '.tiff'))
                        #img2_polar.save(os.path.join(os.path.join(args.hollingsworth_dir, 'images_polar'), seqid2 + '.tiff'))
                        
                        #mask1_polar.save(os.path.join(os.path.join(args.hollingsworth_dir, 'masks_polar'), seqid1 + '.tiff'))
                        #mask2_polar.save(os.path.join(os.path.join(args.hollingsworth_dir, 'masks_polar'), seqid2 + '.tiff'))
                        
                        #img1_crop, mask1_crop, pxyr1_crop, ixyr1_crop = process_image_for_input(img1, mask1, pxyr1, ixyr1, 256)
                        #img2_crop, mask2_crop, pxyr2_crop, ixyr2_crop = process_image_for_input(img2, mask2, pxyr2, ixyr2, 256)
                        
                        alpha1 = float(pxyr1[2]) / float(ixyr1[2])
                        alpha2 = float(pxyr2[2]) / float(ixyr2[2])
                        '''
                        if abs(alpha1 - alpha2) > args.min_alpha_deform:
                            
                            mask1_rowwisesum = np.sum(np.array(mask1), axis=1)
                            mask2_rowwisesum = np.sum(np.array(mask2), axis=1)
                            mask1_nonemptyrows = np.where(mask1_rowwisesum > 0, 1, 0)
                            mask2_nonemptyrows = np.where(mask2_rowwisesum > 0, 1, 0)
                            mask1_height = np.sum(mask1_nonemptyrows)
                            mask2_height = np.sum(mask2_nonemptyrows)
                            if mask1_height < mask2_height:
                                if alpha1 < alpha2:
                                    mask1_binary = mask1.convert("1")
                                    mask2_binary = mask2.convert("1")
                                    mask2_binary_new = ImageChops.logical_and(mask1_binary, mask2_binary)
                                    mask2_new = mask2_binary_new.convert("L")
                                else:
                                    pxyr1_t = torch.tensor(pxyr1).unsqueeze(0).to(device)
                                    ixyr1_t = torch.tensor(ixyr1).unsqueeze(0).to(device)
                                    mask1_t = torch.tensor(np.array(mask1).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
                                    newalpha1_t = torch.tensor(alpha2-0.05).reshape(-1,1).to(device)
                                    mask1_deform_t, _ = deformer.linear_deform(mask1_t, pxyr1_t, ixyr1_t, newalpha1_t, mode='nearest')
                                    mask1_deform_np = mask1_deform_t[0][0].to(torch.uint8).detach().cpu().numpy()
                                    mask1_deform = Image.fromarray(mask1_deform_np, "L")
                                    mask1_d_binary = mask1_deform.convert("1")
                                    mask2_binary = mask2.convert("1")
                                    mask2_binary_new = ImageChops.logical_and(mask1_d_binary, mask2_binary)
                                    mask2_new = mask2_binary_new.convert("L")
                            else:
                                mask2_new = mask2
                            
                            if args.model_type != 'stylegan3':
                                deformed_image, deformed_mask, deformed_polar_image, deformed_polar_mask = deform(img1, mask1, pxyr1, ixyr1, img2, mask2, pxyr2, ixyr2, sifLossModel, deformer, deform_net, args.model_type, device)
                            else:
                                deformed_image, deformed_mask, deformed_polar_image, deformed_polar_mask = deform(img1, mask1, pxyr1, ixyr1, img2, mask2, pxyr2, ixyr2, sifLossModel, deformer, deform_net, args.model_type, device, G)
                        
                            
                            if alpha1 < alpha2:
                                if debug_dilate < debug_total and mask1_height < mask2_height:
                                    image_for_saving = get_concat_h(img1, get_concat_h(mask1, get_concat_h(img2, get_concat_h(mask2, get_concat_h(deformed_image, mask2_new)))))
                                    image_for_saving.save('./debug/dilation/'+str(debug_dilate)+'.tiff')       
                                    debug_dilate += 1
                            else:
                                if debug_constrict < debug_total and mask1_height < mask2_height:
                                    image_for_saving = get_concat_h(img1, get_concat_h(mask1, get_concat_h(img2, get_concat_h(mask2, get_concat_h(deformed_image, mask2_new)))))
                                    image_for_saving.save('./debug/constriction/'+str(debug_constrict)+'.tiff')       
                                    debug_constrict += 1 
                                                  
                        else:
                            deformed_image = img1
                            deformed_mask = mask1
                        '''
                        
                        deformed_image = deform_v1(img1, mask2, deform_net, device)  
                        deformed_image.save(os.path.join(os.path.join(args.hollingsworth_dir, 'recordings'), 'v1_' + seqid1 + '-' + seqid2 + '.tiff'))
                        dnet_pair = 'v1_' + seqid1 + '-' + seqid2 + ',' + seqid2
                        if alpha1 < alpha2:
                            sbdngpFile.write(dnet_pair + '\n')
                        else:
                            bsdngpFile.write(dnet_pair + '\n')
                        
                        deformed_mask, deformed_pxyr, deformed_ixyr = irisRec.segment_and_circApprox(deformed_image)
                        deformed_polar_image, deformed_mask_polar = irisRec.cartToPol_torch(deformed_image, deformed_mask, deformed_pxyr, deformed_ixyr)
                        Image.fromarray(deformed_polar_image, 'L').save(os.path.join(os.path.join(args.hollingsworth_dir, 'images_polar'), 'v1_' + seqid1 + '-' + seqid2 + '.tiff'))
                        
                        img1, mask1, pxyr1, ixyr1, seqid1, alpha1, img2, mask2, pxyr2, ixyr2, seqid2, alpha2 = swap(img1, mask1, pxyr1, ixyr1, seqid1, alpha1, img2, mask2, pxyr2, ixyr2, seqid2, alpha2)
                        '''
                        if abs(alpha1 - alpha2) > args.min_alpha_deform:
                            
                            mask1_rowwisesum = np.sum(np.array(mask1), axis=1)
                            mask2_rowwisesum = np.sum(np.array(mask2), axis=1)
                            mask1_nonemptyrows = np.where(mask1_rowwisesum > 0, 1, 0)
                            mask2_nonemptyrows = np.where(mask2_rowwisesum > 0, 1, 0)
                            mask1_height = np.sum(mask1_nonemptyrows)
                            mask2_height = np.sum(mask2_nonemptyrows)
                            if mask1_height < mask2_height:
                                if alpha1 < alpha2:
                                    mask1_binary = mask1.convert("1")
                                    mask2_binary = mask2.convert("1")
                                    mask2_binary_new = ImageChops.logical_and(mask1_binary, mask2_binary)
                                    mask2_new = mask2_binary_new.convert("L")
                                else:
                                    pxyr1_t = torch.tensor(pxyr1).unsqueeze(0).to(device)
                                    ixyr1_t = torch.tensor(ixyr1).unsqueeze(0).to(device)
                                    mask1_t = torch.tensor(np.array(mask1).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
                                    newalpha1_t = torch.tensor(alpha2-0.05).reshape(-1,1).to(device)
                                    mask1_deform_t, _ = deformer.linear_deform(mask1_t, pxyr1_t, ixyr1_t, newalpha1_t, mode='nearest')
                                    mask1_deform_np = mask1_deform_t[0][0].to(torch.uint8).detach().cpu().numpy()
                                    mask1_deform = Image.fromarray(mask1_deform_np, "L")
                                    mask1_d_binary = mask1_deform.convert("1")
                                    mask2_binary = mask2.convert("1")
                                    mask2_binary_new = ImageChops.logical_and(mask1_d_binary, mask2_binary)
                                    mask2_new = mask2_binary_new.convert("L")
                            else:
                                mask2_new = mask2
                            
                            mask2_new = mask2     
                            if args.model_type != 'stylegan3':
                                deformed_image = deform(img1, pxyr1, ixyr1, mask2_new, pxyr2, ixyr2, deformer, deform_net, args.model_type, device)
                            else:
                                deformed_image = deform(img1, pxyr1, ixyr1, mask2_new, pxyr2, ixyr2, deformer, deform_net, args.model_type, device, G)
                            deformed_mask = mask2
                            
                            if alpha1 < alpha2:
                                if debug_dilate < debug_total and mask1_height < mask2_height:
                                    image_for_saving = get_concat_h(img1, get_concat_h(mask1, get_concat_h(img2, get_concat_h(mask2, get_concat_h(deformed_image, mask2_new)))))
                                    image_for_saving.save('./debug/dilation/'+str(debug_dilate)+'.tiff')       
                                    debug_dilate += 1
                            else:
                                if debug_constrict < debug_total and mask1_height < mask2_height:
                                    image_for_saving = get_concat_h(img1, get_concat_h(mask1, get_concat_h(img2, get_concat_h(mask2, get_concat_h(deformed_image, mask2_new)))))
                                    image_for_saving.save('./debug/constriction/'+str(debug_constrict)+'.tiff')       
                                    debug_constrict += 1
                            
                        else:
                            deformed_image = img1
                            deformed_mask = mask1
                        '''
                        
                        deformed_image = deform_v1(img1, mask2, deform_net, device)  
                        deformed_image.save(os.path.join(os.path.join(args.hollingsworth_dir, 'recordings'), 'v1_' + seqid1 + '-' + seqid2 + '.tiff'))
                        dnet_pair = 'v1_' + seqid1 + '-' + seqid2 + ',' + seqid2
                        if alpha1 < alpha2:
                            sbdngpFile.write(dnet_pair + '\n')
                        else:
                            bsdngpFile.write(dnet_pair + '\n')
                        
                        deformed_mask, deformed_pxyr, deformed_ixyr = irisRec.segment_and_circApprox(deformed_image)
                        deformed_polar_image, deformed_mask_polar = irisRec.cartToPol_torch(deformed_image, deformed_mask, deformed_pxyr, deformed_ixyr)
                        Image.fromarray(deformed_polar_image, 'L').save(os.path.join(os.path.join(args.hollingsworth_dir, 'images_polar'), 'v1_' + seqid1 + '-' + seqid2 + '.tiff'))
                
    
    with open(os.path.join(args.hollingsworth_dir, 'pupil_iris_xyrs_full.pkl'), 'wb') as fullCircleFile:
        pkl.dump(pupil_iris_xyrs, fullCircleFile)
    
def evaluate_hollingsworth(args):
    # Declare Models
    
    debug_constrict = 0
    debug_dilate = 0
    debug_total = 20
    
    if debug_total > 0:
        if not os.path.exists('./debug/'):
            os.mkdir('./debug/')
        else:
            if not os.path.exists('./debug/dilation/'):
                os.mkdir('./debug/dilation/')
            if not os.path.exists('./debug/constriction/'):
                os.mkdir('./debug/constriction/')
    
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    filter_mat1 = io.loadmat(args.sif_filter_path1)['ICAtextureFilters']
    filter_mat2 = io.loadmat(args.sif_filter_path2)['ICAtextureFilters']
    sifLossModel = SIFLayerMaskOSIRIS(polar_height = 64, polar_width = 512, filter_mat1 = filter_mat1, filter_mat2 = filter_mat2, osiris_filters = args.osiris_filters, device=device).to(device)
    
    deformer = LinearDeformer(device)
    
    if args.model_type == 'stylegan3':
        deform_net = torch.load(args.weight_path)[0].to(device)
        with dnnlib.util.open_url(args.sg3_weight) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    else:
        if args.load_ema:
            ema = torch.load(args.weight_path)[0]
            deform_net = ema.ema_model.to(device)
        else:
            deform_net = torch.load(args.weight_path)[0].to(device)
    deform_net.eval()
    
    with open(os.path.join(args.hollingsworth_dir, 'pupil_iris_xyrs.pkl'), 'rb') as circleFile:
        pupil_iris_xyrs = pkl.load(circleFile)
        
    irisRec = irisRecognition(get_cfg(args.cfg), use_hough=False)
    
    for pairFileName in tqdm(['genuine_pairs.txt', 'imposter_pairs.txt']):
        with open(os.path.join(args.hollingsworth_dir, 'dnet_bs_' + pairFileName), 'w') as bsdngpFile:
            with open(os.path.join(args.hollingsworth_dir, 'dnet_sb_' + pairFileName), 'w') as sbdngpFile:
                with open(os.path.join(args.parent_dir_wsd, 'biomech_' + pairFileName), 'w') as bmFile:
                    with open(os.path.join(args.hollingsworth_dir, pairFileName), 'r') as pFile: 
                        for line in tqdm(pFile):
                            lineparts = line.strip().split(',')
                            
                            seqid1 = lineparts[0]
                            seqid2 = lineparts[1]
                            
                            img1 = Image.open(os.path.join(os.path.join(args.hollingsworth_dir, 'recordings'), seqid1 + '.tiff')).convert('L')
                            img2 = Image.open(os.path.join(os.path.join(args.hollingsworth_dir, 'recordings'), seqid2 + '.tiff')).convert('L')
                            
                            mask1 = Image.open(os.path.join(os.path.join(args.hollingsworth_dir, 'masks_coarse'), seqid1 + '.tiff')).convert('L')
                            mask2 = Image.open(os.path.join(os.path.join(args.hollingsworth_dir, 'masks_coarse'), seqid2 + '.tiff')).convert('L')
                            
                            pxyr1 = np.copy(pupil_iris_xyrs[seqid1]['pxyr'])
                            ixyr1 = np.copy(pupil_iris_xyrs[seqid1]['ixyr'])
                            pxyr2 = np.copy(pupil_iris_xyrs[seqid2]['pxyr'])
                            ixyr2 = np.copy(pupil_iris_xyrs[seqid2]['ixyr'])
                            
                            #img1_polar, mask1_polar = polarize(img1, mask1, pxyr1, ixyr1, sifLossModel, device)
                            #img2_polar, mask2_polar = polarize(img2, mask2, pxyr2, ixyr2, sifLossModel, device)
                            
                            #img1_polar.save(os.path.join(os.path.join(args.hollingsworth_dir, 'images_polar'), seqid1 + '.tiff'))
                            #img2_polar.save(os.path.join(os.path.join(args.hollingsworth_dir, 'images_polar'), seqid2 + '.tiff'))
                            
                            #mask1_polar.save(os.path.join(os.path.join(args.hollingsworth_dir, 'masks_polar'), seqid1 + '.tiff'))
                            #mask2_polar.save(os.path.join(os.path.join(args.hollingsworth_dir, 'masks_polar'), seqid2 + '.tiff'))
                            
                            #img1_crop, mask1_crop, pxyr1_crop, ixyr1_crop = process_image_for_input(img1, mask1, pxyr1, ixyr1, 256)
                            #img2_crop, mask2_crop, pxyr2_crop, ixyr2_crop = process_image_for_input(img2, mask2, pxyr2, ixyr2, 256)
                            
                            alpha1 = float(pxyr1[2]) / float(ixyr1[2])
                            alpha2 = float(pxyr2[2]) / float(ixyr2[2])
                            '''
                            if abs(alpha1 - alpha2) > args.min_alpha_deform:
                                
                                mask1_rowwisesum = np.sum(np.array(mask1), axis=1)
                                mask2_rowwisesum = np.sum(np.array(mask2), axis=1)
                                mask1_nonemptyrows = np.where(mask1_rowwisesum > 0, 1, 0)
                                mask2_nonemptyrows = np.where(mask2_rowwisesum > 0, 1, 0)
                                mask1_height = np.sum(mask1_nonemptyrows)
                                mask2_height = np.sum(mask2_nonemptyrows)
                                if mask1_height < mask2_height:
                                    if alpha1 < alpha2:
                                        mask1_binary = mask1.convert("1")
                                        mask2_binary = mask2.convert("1")
                                        mask2_binary_new = ImageChops.logical_and(mask1_binary, mask2_binary)
                                        mask2_new = mask2_binary_new.convert("L")
                                    else:
                                        pxyr1_t = torch.tensor(pxyr1).unsqueeze(0).to(device)
                                        ixyr1_t = torch.tensor(ixyr1).unsqueeze(0).to(device)
                                        mask1_t = torch.tensor(np.array(mask1).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
                                        newalpha1_t = torch.tensor(alpha2-0.05).reshape(-1,1).to(device)
                                        mask1_deform_t, _ = deformer.linear_deform(mask1_t, pxyr1_t, ixyr1_t, newalpha1_t, mode='nearest')
                                        mask1_deform_np = mask1_deform_t[0][0].to(torch.uint8).detach().cpu().numpy()
                                        mask1_deform = Image.fromarray(mask1_deform_np, "L")
                                        mask1_d_binary = mask1_deform.convert("1")
                                        mask2_binary = mask2.convert("1")
                                        mask2_binary_new = ImageChops.logical_and(mask1_d_binary, mask2_binary)
                                        mask2_new = mask2_binary_new.convert("L")
                                else:
                                    mask2_new = mask2
                                
                                if args.model_type != 'stylegan3':
                                    deformed_image, deformed_mask, deformed_polar_image, deformed_polar_mask = deform(img1, mask1, pxyr1, ixyr1, img2, mask2, pxyr2, ixyr2, sifLossModel, deformer, deform_net, args.model_type, device)
                                else:
                                    deformed_image, deformed_mask, deformed_polar_image, deformed_polar_mask = deform(img1, mask1, pxyr1, ixyr1, img2, mask2, pxyr2, ixyr2, sifLossModel, deformer, deform_net, args.model_type, device, G)
                            
                                
                                if alpha1 < alpha2:
                                    if debug_dilate < debug_total and mask1_height < mask2_height:
                                        image_for_saving = get_concat_h(img1, get_concat_h(mask1, get_concat_h(img2, get_concat_h(mask2, get_concat_h(deformed_image, mask2_new)))))
                                        image_for_saving.save('./debug/dilation/'+str(debug_dilate)+'.tiff')       
                                        debug_dilate += 1
                                else:
                                    if debug_constrict < debug_total and mask1_height < mask2_height:
                                        image_for_saving = get_concat_h(img1, get_concat_h(mask1, get_concat_h(img2, get_concat_h(mask2, get_concat_h(deformed_image, mask2_new)))))
                                        image_for_saving.save('./debug/constriction/'+str(debug_constrict)+'.tiff')       
                                        debug_constrict += 1 
                                                      
                            else:
                                deformed_image = img1
                                deformed_mask = mask1
                            '''
                            
                            if args.model_type != 'stylegan3':
                                deformed_image, deformed_mask, deformed_polar_image, deformed_polar_mask, deformed_pxyr, deformed_ixyr = deform(img1, mask1, pxyr1, ixyr1, img2, mask2, pxyr2, ixyr2, sifLossModel, deformer, deform_net, args.model_type, device)
                            else:
                                deformed_image, deformed_mask, deformed_polar_image, deformed_polar_mask = deform(img1, mask1, pxyr1, ixyr1, img2, mask2, pxyr2, ixyr2, sifLossModel, deformer, deform_net, args.model_type, device, G)
                                
                            deformed_image.save(os.path.join(os.path.join(args.hollingsworth_dir, 'recordings'), seqid1 + '-' + seqid2 + '.tiff'))
                            dnet_pair = seqid1 + '-' + seqid2 + ',' + seqid2
                            if alpha1 < alpha2:
                                sbdngpFile.write(dnet_pair + '\n')
                                
                                spr = float(alpha1 * 0.006)
                                bpr = float(alpha2 * 0.006)
                                biomech_radii = np.float32(Biomech(spr, bpr, 65).flatten()[1:])
                                
                                bm_deformed_polar_image, bm_deformed_polar_mask = irisRec.bioMechCartToPol_torch(img2, mask2, pxyr2, ixyr2, biomech_radii)
                                bm_pair = 'biomech_' + seqid1 + '-' + seqid2 + ',' + seqid2
                                Image.fromarray(bm_deformed_polar_image, 'L').save(os.path.join(os.path.join(args.hollingsworth_dir, 'masks_coarse'), 'biomech_' + seqid1 + '-' + seqid2 + '.tiff'))     
                                Image.fromarray(bm_deformed_polar_mask, 'L').save(os.path.join(os.path.join(args.hollingsworth_dir, 'masks_polar'), 'biomech_' + seqid1 + '-' + seqid2 + '.tiff'))  
                                
                                bmFile.write(bm_pair + '\n')   
                            else:
                                bsdngpFile.write(dnet_pair + '\n')
                            
                            pupil_iris_xyrs[seqid1 + '-' + seqid2] = {}
                            pupil_iris_xyrs[seqid1 + '-' + seqid2]['pxyr'] = deformed_pxyr
                            pupil_iris_xyrs[seqid1 + '-' + seqid2]['ixyr'] = deformed_ixyr
                            
                            deformed_mask.save(os.path.join(os.path.join(args.hollingsworth_dir, 'masks_coarse'), seqid1 + '-' + seqid2 + '.tiff'))
                            deformed_polar_image.save(os.path.join(os.path.join(args.hollingsworth_dir, 'images_polar'), seqid1 + '-' + seqid2 + '.tiff'))
                            deformed_polar_mask.save(os.path.join(os.path.join(args.hollingsworth_dir, 'masks_polar'), seqid1 + '-' + seqid2 + '.tiff'))
                            
                            img1, mask1, pxyr1, ixyr1, seqid1, alpha1, img2, mask2, pxyr2, ixyr2, seqid2, alpha2 = swap(img1, mask1, pxyr1, ixyr1, seqid1, alpha1, img2, mask2, pxyr2, ixyr2, seqid2, alpha2)
                            '''
                            if abs(alpha1 - alpha2) > args.min_alpha_deform:
                                
                                mask1_rowwisesum = np.sum(np.array(mask1), axis=1)
                                mask2_rowwisesum = np.sum(np.array(mask2), axis=1)
                                mask1_nonemptyrows = np.where(mask1_rowwisesum > 0, 1, 0)
                                mask2_nonemptyrows = np.where(mask2_rowwisesum > 0, 1, 0)
                                mask1_height = np.sum(mask1_nonemptyrows)
                                mask2_height = np.sum(mask2_nonemptyrows)
                                if mask1_height < mask2_height:
                                    if alpha1 < alpha2:
                                        mask1_binary = mask1.convert("1")
                                        mask2_binary = mask2.convert("1")
                                        mask2_binary_new = ImageChops.logical_and(mask1_binary, mask2_binary)
                                        mask2_new = mask2_binary_new.convert("L")
                                    else:
                                        pxyr1_t = torch.tensor(pxyr1).unsqueeze(0).to(device)
                                        ixyr1_t = torch.tensor(ixyr1).unsqueeze(0).to(device)
                                        mask1_t = torch.tensor(np.array(mask1).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
                                        newalpha1_t = torch.tensor(alpha2-0.05).reshape(-1,1).to(device)
                                        mask1_deform_t, _ = deformer.linear_deform(mask1_t, pxyr1_t, ixyr1_t, newalpha1_t, mode='nearest')
                                        mask1_deform_np = mask1_deform_t[0][0].to(torch.uint8).detach().cpu().numpy()
                                        mask1_deform = Image.fromarray(mask1_deform_np, "L")
                                        mask1_d_binary = mask1_deform.convert("1")
                                        mask2_binary = mask2.convert("1")
                                        mask2_binary_new = ImageChops.logical_and(mask1_d_binary, mask2_binary)
                                        mask2_new = mask2_binary_new.convert("L")
                                else:
                                    mask2_new = mask2
                                
                                mask2_new = mask2     
                                if args.model_type != 'stylegan3':
                                    deformed_image = deform(img1, pxyr1, ixyr1, mask2_new, pxyr2, ixyr2, deformer, deform_net, args.model_type, device)
                                else:
                                    deformed_image = deform(img1, pxyr1, ixyr1, mask2_new, pxyr2, ixyr2, deformer, deform_net, args.model_type, device, G)
                                deformed_mask = mask2
                                
                                if alpha1 < alpha2:
                                    if debug_dilate < debug_total and mask1_height < mask2_height:
                                        image_for_saving = get_concat_h(img1, get_concat_h(mask1, get_concat_h(img2, get_concat_h(mask2, get_concat_h(deformed_image, mask2_new)))))
                                        image_for_saving.save('./debug/dilation/'+str(debug_dilate)+'.tiff')       
                                        debug_dilate += 1
                                else:
                                    if debug_constrict < debug_total and mask1_height < mask2_height:
                                        image_for_saving = get_concat_h(img1, get_concat_h(mask1, get_concat_h(img2, get_concat_h(mask2, get_concat_h(deformed_image, mask2_new)))))
                                        image_for_saving.save('./debug/constriction/'+str(debug_constrict)+'.tiff')       
                                        debug_constrict += 1
                                
                            else:
                                deformed_image = img1
                                deformed_mask = mask1
                            '''
                            
                            if args.model_type != 'stylegan3':
                                deformed_image, deformed_mask, deformed_polar_image, deformed_polar_mask, deformed_pxyr, deformed_ixyr = deform(img1, mask1, pxyr1, ixyr1, img2, mask2, pxyr2, ixyr2, sifLossModel, deformer, deform_net, args.model_type, device)
                            else:
                                deformed_image, deformed_mask, deformed_polar_image, deformed_polar_mask, deformed_pxyr, deformed_ixyr = deform(img1, mask1, pxyr1, ixyr1, img2, mask2, pxyr2, ixyr2, sifLossModel, deformer, deform_net, args.model_type, device, G)
                                
                            deformed_image.save(os.path.join(os.path.join(args.hollingsworth_dir, 'recordings'), seqid1 + '-' + seqid2 + '.tiff'))
                            dnet_pair = seqid1 + '-' + seqid2 + ',' + seqid2
                            if alpha1 < alpha2:
                                sbdngpFile.write(dnet_pair + '\n')
                                
                                spr = float(alpha1 * 0.006)
                                bpr = float(alpha2 * 0.006)
                                biomech_radii = np.float32(Biomech(spr, bpr, 65).flatten()[1:])
                                
                                bm_deformed_polar_image, bm_deformed_polar_mask = irisRec.bioMechCartToPol_torch(img2, mask2, pxyr2, ixyr2, biomech_radii)
                                bm_pair = 'biomech_' + seqid1 + '-' + seqid2 + ',' + seqid2
                                Image.fromarray(bm_deformed_polar_image, 'L').save(os.path.join(os.path.join(args.hollingsworth_dir, 'masks_coarse'), 'biomech_' + seqid1 + '-' + seqid2 + '.tiff'))     
                                Image.fromarray(bm_deformed_polar_mask, 'L').save(os.path.join(os.path.join(args.hollingsworth_dir, 'masks_polar'), 'biomech_' + seqid1 + '-' + seqid2 + '.tiff'))  
                                
                                bmFile.write(bm_pair + '\n')   
                            else:
                                bsdngpFile.write(dnet_pair + '\n')
                            
                            pupil_iris_xyrs[seqid1 + '-' + seqid2] = {}
                            pupil_iris_xyrs[seqid1 + '-' + seqid2]['pxyr'] = deformed_pxyr
                            pupil_iris_xyrs[seqid1 + '-' + seqid2]['ixyr'] = deformed_ixyr
                            
                            deformed_mask.save(os.path.join(os.path.join(args.hollingsworth_dir, 'masks_coarse'), seqid1 + '-' + seqid2 + '.tiff'))
                            deformed_polar_image.save(os.path.join(os.path.join(args.hollingsworth_dir, 'images_polar'), seqid1 + '-' + seqid2 + '.tiff'))
                            deformed_polar_mask.save(os.path.join(os.path.join(args.hollingsworth_dir, 'masks_polar'), seqid1 + '-' + seqid2 + '.tiff'))
                    
    
    with open(os.path.join(args.hollingsworth_dir, 'pupil_iris_xyrs_full.pkl'), 'wb') as fullCircleFile:
        pkl.dump(pupil_iris_xyrs, fullCircleFile)

def evaluate_hollingsworth_only_biomech(args):
    
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    with open(os.path.join(args.hollingsworth_dir, 'pupil_iris_xyrs.pkl'), 'rb') as circleFile:
        pupil_iris_xyrs = pkl.load(circleFile)
        
    irisRec = irisRecognition(get_cfg(args.cfg), use_hough=False)
    
    for pairFileName in tqdm(['genuine_pairs.txt', 'imposter_pairs.txt']):
        with open(os.path.join(args.hollingsworth_dir, 'biomech_' + pairFileName), 'w') as bmFile:
            with open(os.path.join(args.hollingsworth_dir, pairFileName), 'r') as pFile: 
                for line in tqdm(pFile):
                    lineparts = line.strip().split(',')
                    
                    seqid1 = lineparts[0]
                    seqid2 = lineparts[1]
                    
                    img1 = Image.open(os.path.join(os.path.join(args.hollingsworth_dir, 'recordings'), seqid1 + '.tiff')).convert('L')
                    img2 = Image.open(os.path.join(os.path.join(args.hollingsworth_dir, 'recordings'), seqid2 + '.tiff')).convert('L')
                    
                    mask1 = Image.open(os.path.join(os.path.join(args.hollingsworth_dir, 'masks_coarse'), seqid1 + '.tiff')).convert('L')
                    mask2 = Image.open(os.path.join(os.path.join(args.hollingsworth_dir, 'masks_coarse'), seqid2 + '.tiff')).convert('L')
                    
                    pxyr1 = np.copy(pupil_iris_xyrs[seqid1]['pxyr'])
                    ixyr1 = np.copy(pupil_iris_xyrs[seqid1]['ixyr'])
                    pxyr2 = np.copy(pupil_iris_xyrs[seqid2]['pxyr'])
                    ixyr2 = np.copy(pupil_iris_xyrs[seqid2]['ixyr'])
                    
                    #img1_polar, mask1_polar = polarize(img1, mask1, pxyr1, ixyr1, sifLossModel, device)
                    #img2_polar, mask2_polar = polarize(img2, mask2, pxyr2, ixyr2, sifLossModel, device)
                    
                    #img1_polar.save(os.path.join(os.path.join(args.hollingsworth_dir, 'images_polar'), seqid1 + '.tiff'))
                    #img2_polar.save(os.path.join(os.path.join(args.hollingsworth_dir, 'images_polar'), seqid2 + '.tiff'))
                    
                    #mask1_polar.save(os.path.join(os.path.join(args.hollingsworth_dir, 'masks_polar'), seqid1 + '.tiff'))
                    #mask2_polar.save(os.path.join(os.path.join(args.hollingsworth_dir, 'masks_polar'), seqid2 + '.tiff'))
                    
                    #img1_crop, mask1_crop, pxyr1_crop, ixyr1_crop = process_image_for_input(img1, mask1, pxyr1, ixyr1, 256)
                    #img2_crop, mask2_crop, pxyr2_crop, ixyr2_crop = process_image_for_input(img2, mask2, pxyr2, ixyr2, 256)
                    
                    alpha1 = float(pxyr1[2]) / float(ixyr1[2])
                    alpha2 = float(pxyr2[2]) / float(ixyr2[2])
    
                    if alpha1 < alpha2:
                        
                        spr = float(alpha1 * 0.006)
                        bpr = float(alpha2 * 0.006)
                        biomech_radii = np.float32(Biomech(spr, bpr, 65).flatten()[1:])
                        
                        bm_deformed_polar_image, bm_deformed_polar_mask = irisRec.bioMechCartToPol_torch(img2, mask2, pxyr2, ixyr2, biomech_radii)
                        bm_pair = 'biomech_' + seqid1 + '-' + seqid2 + ',' + seqid2
                        Image.fromarray(bm_deformed_polar_image, 'L').save(os.path.join(os.path.join(args.hollingsworth_dir, 'images_polar'), 'biomech_' + seqid1 + '-' + seqid2 + '.tiff'))     
                        Image.fromarray(bm_deformed_polar_mask, 'L').save(os.path.join(os.path.join(args.hollingsworth_dir, 'masks_polar'), 'biomech_' + seqid1 + '-' + seqid2 + '.tiff'))  
                        
                        bmFile.write(bm_pair + '\n')                                   
                    
                    
                    img1, mask1, pxyr1, ixyr1, seqid1, alpha1, img2, mask2, pxyr2, ixyr2, seqid2, alpha2 = swap(img1, mask1, pxyr1, ixyr1, seqid1, alpha1, img2, mask2, pxyr2, ixyr2, seqid2, alpha2)
                    
                    if alpha1 < alpha2:
                        
                        spr = float(alpha1 * 0.006)
                        bpr = float(alpha2 * 0.006)
                        biomech_radii = np.float32(Biomech(spr, bpr, 65).flatten()[1:])
                        
                        bm_deformed_polar_image, bm_deformed_polar_mask = irisRec.bioMechCartToPol_torch(img2, mask2, pxyr2, ixyr2, biomech_radii)
                        bm_pair = 'biomech_' + seqid1 + '-' + seqid2 + ',' + seqid2
                        Image.fromarray(bm_deformed_polar_image, 'L').save(os.path.join(os.path.join(args.hollingsworth_dir, 'images_polar'), 'biomech_' + seqid1 + '-' + seqid2 + '.tiff'))     
                        Image.fromarray(bm_deformed_polar_mask, 'L').save(os.path.join(os.path.join(args.hollingsworth_dir, 'masks_polar'), 'biomech_' + seqid1 + '-' + seqid2 + '.tiff'))  
                        
                        bmFile.write(bm_pair + '\n')

def evaluate_qfire(args):
    # Declare Models
    
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    deformer = LinearDeformer(device)
    
    filter_mat1 = io.loadmat(args.sif_filter_path1)['ICAtextureFilters']
    filter_mat2 = io.loadmat(args.sif_filter_path2)['ICAtextureFilters']
    sifLossModel = SIFLayerMaskOSIRIS(polar_height = 64, polar_width = 512, filter_mat1 = filter_mat1, filter_mat2 = filter_mat2, osiris_filters = args.osiris_filters, device=device).to(device)
    
    if args.model_type == 'stylegan3':
        deform_net = torch.load(args.weight_path)[0].to(device)
        with dnnlib.util.open_url(args.sg3_weight) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    else:
        if args.load_ema:
            ema = torch.load(args.weight_path)[0]
            deform_net = ema.ema_model.to(device)
        else:
            deform_net = torch.load(args.weight_path)[0].to(device)
    deform_net.eval()
    
    with open(os.path.join(args.qfire_dir, 'pupil_iris_xyrs.pkl'), 'rb') as circleFile:
        pupil_iris_xyrs = pkl.load(circleFile)
    
    for pairFileName in tqdm(['genuine_pairs.csv', 'imposter_pairs.csv']):
        with open(os.path.join(args.qfire_dir, 'dnet_bs_' + pairFileName), 'w') as bsdngpFile:
            with open(os.path.join(args.qfire_dir, 'dnet_sb_' + pairFileName), 'w') as sbdngpFile:
                with open(os.path.join(args.qfire_dir, pairFileName), 'r') as pFile: 
                    for line in tqdm(pFile):
                        lineparts = line.strip().split(',')
                        
                        file1 = lineparts[0].strip()
                        file2 = lineparts[1].strip()
                        
                        img1 = Image.open(os.path.join(os.path.join(args.qfire_dir, 'images'), file1)).convert('L')
                        img2 = Image.open(os.path.join(os.path.join(args.qfire_dir, 'images'), file2)).convert('L')
                        
                        mask1 = Image.open(os.path.join(os.path.join(args.qfire_dir, 'masks'), file1)).convert('L')
                        mask2 = Image.open(os.path.join(os.path.join(args.qfire_dir, 'masks'), file2)).convert('L')
                        
                        pxyr1 = np.copy(pupil_iris_xyrs[file1]['pxyr'])
                        ixyr1 = np.copy(pupil_iris_xyrs[file1]['ixyr'])
                        pxyr2 = np.copy(pupil_iris_xyrs[file2]['pxyr'])
                        ixyr2 = np.copy(pupil_iris_xyrs[file2]['ixyr'])
                        
                        #img1, mask1, pxyr1, ixyr1 = process_image_for_input(img1, mask1, pxyr1, ixyr1, 256)
                        #img2, mask2, pxyr2, ixyr2 = process_image_for_input(img2, mask2, pxyr2, ixyr2, 256)
                        
                        alpha1 = float(pxyr1[2]) / float(ixyr1[2])
                        alpha2 = float(pxyr2[2]) / float(ixyr2[2])
                        
                        #print(alpha1, alpha2)
                        '''
                        if abs(alpha1 - alpha2) > args.min_alpha_deform:
                            
                            mask1_rowwisesum = np.sum(np.array(mask1), axis=1)
                            mask2_rowwisesum = np.sum(np.array(mask2), axis=1)
                            mask1_nonemptyrows = np.where(mask1_rowwisesum > 0, 1, 0)
                            mask2_nonemptyrows = np.where(mask2_rowwisesum > 0, 1, 0)
                            mask1_height = np.sum(mask1_nonemptyrows)
                            mask2_height = np.sum(mask2_nonemptyrows)
                            if mask1_height < mask2_height:
                                if alpha1 < alpha2:
                                    mask1_binary = mask1.convert("1")
                                    mask2_binary = mask2.convert("1")
                                    mask2_binary_new = ImageChops.logical_and(mask1_binary, mask2_binary)
                                    mask2_new = mask2_binary_new.convert("L")
                                else:
                                    pxyr1_t = torch.tensor(pxyr1).unsqueeze(0).to(device)
                                    ixyr1_t = torch.tensor(ixyr1).unsqueeze(0).to(device)
                                    mask1_t = torch.tensor(np.array(mask1).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
                                    newalpha1_t = torch.tensor(alpha2-0.05).reshape(-1,1).to(device)
                                    mask1_deform_t, _ = deformer.linear_deform(mask1_t, pxyr1_t, ixyr1_t, newalpha1_t, mode='nearest')
                                    mask1_deform_np = mask1_deform_t[0][0].to(torch.uint8).detach().cpu().numpy()
                                    mask1_deform = Image.fromarray(mask1_deform_np, "L")
                                    mask1_d_binary = mask1_deform.convert("1")
                                    mask2_binary = mask2.convert("1")
                                    mask2_binary_new = ImageChops.logical_and(mask1_d_binary, mask2_binary)
                                    mask2_new = mask2_binary_new.convert("L")
                            else:
                                mask2_new = mask2
                            
                            mask2_new = mask2 
                            if args.model_type != 'stylegan3':
                                deformed_image = deform(img1, pxyr1, ixyr1, mask2_new, pxyr2, ixyr2, deformer, deform_net, args.model_type, device)
                            else:
                                deformed_image = deform(img1, pxyr1, ixyr1, mask2_new, pxyr2, ixyr2, deformer, deform_net, args.model_type, device, G)
                            deformed_mask = mask2
                        else:
                            deformed_image = img1
                            deformed_mask = mask1
                        '''
                        if args.model_type != 'stylegan3':
                            deformed_image, deformed_mask, deformed_polar_image, deformed_polar_mask = deform(img1, mask1, pxyr1, ixyr1, img2, mask2, pxyr2, ixyr2, sifLossModel, deformer, deform_net, args.model_type, device)
                        else:
                            deformed_image, deformed_mask, deformed_polar_image, deformed_polar_mask = deform(img1, mask1, pxyr1, ixyr1, img2, mask2, pxyr2, ixyr2, sifLossModel, deformer, deform_net, args.model_type, device, G)
                         
                        deformed_image.save(os.path.join(os.path.join(args.qfire_dir, 'images'), file1.split('.')[0] + '_' + file2.split('/')[1].split('.')[0] + '.png'))
                        dnet_pair = file1.split('.')[0] + '_' + file2.split('/')[1].split('.')[0] + '.png' + ',' + file2
                        if alpha1 < alpha2:
                            sbdngpFile.write(dnet_pair + '\n')
                        else:
                            bsdngpFile.write(dnet_pair + '\n')
                        
                        pupil_iris_xyrs[file1.split('.')[0] + '_' + file2.split('/')[1].split('.')[0] + '.png'] = {}
                        pupil_iris_xyrs[file1.split('.')[0] + '_' + file2.split('/')[1].split('.')[0] + '.png']['pxyr'] = pxyr2
                        pupil_iris_xyrs[file1.split('.')[0] + '_' + file2.split('/')[1].split('.')[0] + '.png']['ixyr'] = ixyr2
                        
                        deformed_mask.save(os.path.join(os.path.join(args.qfire_dir, 'masks'), file1.split('.')[0] + '_' + file2.split('/')[1].split('.')[0] + '.png'))
                        deformed_polar_image.save(os.path.join(os.path.join(args.qfire_dir, 'images_polar'), file1.split('.')[0] + '_' + file2.split('/')[1].split('.')[0] + '.png'))
                        deformed_polar_mask.save(os.path.join(os.path.join(args.qfire_dir, 'masks_polar'), file1.split('.')[0] + '_' + file2.split('/')[1].split('.')[0] + '.png'))
                        
                        img1, mask1, pxyr1, ixyr1, file1, alpha1, img2, mask2, pxyr2, ixyr2, file2, alpha2 = swap(img1, mask1, pxyr1, ixyr1, file1, alpha1, img2, mask2, pxyr2, ixyr2, file2, alpha2)
                        '''
                        if abs(alpha1 - alpha2) > args.min_alpha_deform:
                            
                            mask1_rowwisesum = np.sum(np.array(mask1), axis=1)
                            mask2_rowwisesum = np.sum(np.array(mask2), axis=1)
                            mask1_nonemptyrows = np.where(mask1_rowwisesum > 0, 1, 0)
                            mask2_nonemptyrows = np.where(mask2_rowwisesum > 0, 1, 0)
                            mask1_height = np.sum(mask1_nonemptyrows)
                            mask2_height = np.sum(mask2_nonemptyrows)
                            if mask1_height < mask2_height:
                                if alpha1 < alpha2:
                                    mask1_binary = mask1.convert("1")
                                    mask2_binary = mask2.convert("1")
                                    mask2_binary_new = ImageChops.logical_and(mask1_binary, mask2_binary)
                                    mask2_new = mask2_binary_new.convert("L")
                                else:
                                    pxyr1_t = torch.tensor(pxyr1).unsqueeze(0).to(device)
                                    ixyr1_t = torch.tensor(ixyr1).unsqueeze(0).to(device)
                                    mask1_t = torch.tensor(np.array(mask1).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
                                    newalpha1_t = torch.tensor(alpha2-0.05).reshape(-1,1).to(device)
                                    mask1_deform_t, _ = deformer.linear_deform(mask1_t, pxyr1_t, ixyr1_t, newalpha1_t, mode='nearest')
                                    mask1_deform_np = mask1_deform_t[0][0].to(torch.uint8).detach().cpu().numpy()
                                    mask1_deform = Image.fromarray(mask1_deform_np, "L")
                                    mask1_d_binary = mask1_deform.convert("1")
                                    mask2_binary = mask2.convert("1")
                                    mask2_binary_new = ImageChops.logical_and(mask1_d_binary, mask2_binary)
                                    mask2_new = mask2_binary_new.convert("L")
                            else:
                                mask2_new = mask2
                            
                            mask2_new = mask2
                            if args.model_type != 'stylegan3':
                                deformed_image = deform(img1, pxyr1, ixyr1, mask2_new, pxyr2, ixyr2, deformer, deform_net, args.model_type, device)
                            else:
                                deformed_image = deform(img1, pxyr1, ixyr1, mask2_new, pxyr2, ixyr2, deformer, deform_net, args.model_type, device, G)
                            deformed_mask = mask2
                        else:
                            deformed_image = img1
                            deformed_mask = mask1
                        '''
                        if args.model_type != 'stylegan3':
                            deformed_image, deformed_mask, deformed_polar_image, deformed_polar_mask = deform(img1, mask1, pxyr1, ixyr1, img2, mask2, pxyr2, ixyr2, sifLossModel, deformer, deform_net, args.model_type, device)
                        else:
                            deformed_image, deformed_mask, deformed_polar_image, deformed_polar_mask = deform(img1, mask1, pxyr1, ixyr1, img2, mask2, pxyr2, ixyr2, sifLossModel, deformer, deform_net, args.model_type, device, G)
                        deformed_image.save(os.path.join(os.path.join(args.qfire_dir, 'images'), file1.split('.')[0] + '_' + file2.split('/')[1].split('.')[0] + '.png'))
                        dnet_pair = file1.split('.')[0] + '_' + file2.split('/')[1].split('.')[0] + '.png' + ',' + file2
                        if alpha1 < alpha2:
                            sbdngpFile.write(dnet_pair + '\n')
                        else:
                            bsdngpFile.write(dnet_pair + '\n')
                        
                        pupil_iris_xyrs[file1.split('.')[0] + '_' + file2.split('/')[1].split('.')[0] + '.png'] = {}
                        pupil_iris_xyrs[file1.split('.')[0] + '_' + file2.split('/')[1].split('.')[0] + '.png']['pxyr'] = pxyr2
                        pupil_iris_xyrs[file1.split('.')[0] + '_' + file2.split('/')[1].split('.')[0] + '.png']['ixyr'] = ixyr2
                        
                        deformed_mask.save(os.path.join(os.path.join(args.qfire_dir, 'masks'), file1.split('.')[0] + '_' + file2.split('/')[1].split('.')[0] + '.png'))
                        deformed_polar_image.save(os.path.join(os.path.join(args.qfire_dir, 'images_polar'), file1.split('.')[0] + '_' + file2.split('/')[1].split('.')[0] + '.png'))
                        deformed_polar_mask.save(os.path.join(os.path.join(args.qfire_dir, 'masks_polar'), file1.split('.')[0] + '_' + file2.split('/')[1].split('.')[0] + '.png'))
                    
    
    with open(os.path.join(args.qfire_dir, 'pupil_iris_xyrs_full.pkl'), 'wb') as fullCircleFile:
        pkl.dump(pupil_iris_xyrs, fullCircleFile)

def evaluate_casia(args):
    # Declare Models
    
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    deformer = LinearDeformer(device)
    
    filter_mat1 = io.loadmat(args.sif_filter_path1)['ICAtextureFilters']
    filter_mat2 = io.loadmat(args.sif_filter_path2)['ICAtextureFilters']
    sifLossModel = SIFLayerMaskOSIRIS(polar_height = 64, polar_width = 512, filter_mat1 = filter_mat1, filter_mat2 = filter_mat2, osiris_filters = args.osiris_filters, device=device).to(device)
    
    if args.model_type == 'stylegan3':
        deform_net = torch.load(args.weight_path)[0].to(device)
        with dnnlib.util.open_url(args.sg3_weight) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    else:
        if args.load_ema:
            ema = torch.load(args.weight_path)[0]
            deform_net = ema.ema_model.to(device)
        else:
            deform_net = torch.load(args.weight_path)[0].to(device)
    deform_net.eval()
    
    with open(os.path.join(args.casia_dir, 'pupil_iris_xyrs.pkl'), 'rb') as circleFile:
        pupil_iris_xyrs = pkl.load(circleFile)
    
    for pairFileName in tqdm(['genuine_pairs.csv', 'imposter_pairs.csv']):
        with open(os.path.join(args.casia_dir, 'dnet_bs_' + pairFileName), 'w') as bsdngpFile:
            with open(os.path.join(args.casia_dir, 'dnet_sb_' + pairFileName), 'w') as sbdngpFile:
                with open(os.path.join(args.casia_dir, pairFileName), 'r') as pFile: 
                    for line in tqdm(pFile):
                        lineparts = line.strip().split(',')
                        
                        file1 = lineparts[0].strip()
                        file2 = lineparts[1].strip()
                        
                        img1 = Image.open(os.path.join(os.path.join(args.casia_dir, 'images_iso'), file1)).convert('L')
                        img2 = Image.open(os.path.join(os.path.join(args.casia_dir, 'images_iso'), file2)).convert('L')
                        
                        mask1 = Image.open(os.path.join(os.path.join(args.casia_dir, 'masks_coarse'), file1)).convert('L')
                        mask2 = Image.open(os.path.join(os.path.join(args.casia_dir, 'masks_coarse'), file2)).convert('L')
                        
                        pxyr1 = np.copy(pupil_iris_xyrs[file1]['pxyr'])
                        ixyr1 = np.copy(pupil_iris_xyrs[file1]['ixyr'])
                        pxyr2 = np.copy(pupil_iris_xyrs[file2]['pxyr'])
                        ixyr2 = np.copy(pupil_iris_xyrs[file2]['ixyr'])
                        
                        alpha1 = float(pxyr1[2]) / float(ixyr1[2])
                        alpha2 = float(pxyr2[2]) / float(ixyr2[2])
                        
                        if args.model_type != 'stylegan3':
                            deformed_image, deformed_mask, deformed_polar_image, deformed_polar_mask = deform(img1, mask1, pxyr1, ixyr1, img2, mask2, pxyr2, ixyr2, sifLossModel, deformer, deform_net, args.model_type, device)
                        else:
                            deformed_image, deformed_mask, deformed_polar_image, deformed_polar_mask = deform(img1, mask1, pxyr1, ixyr1, img2, mask2, pxyr2, ixyr2, sifLossModel, deformer, deform_net, args.model_type, device, G)
                        
                        filename1 = file1.split('/')[-1]
                        filename2 = file2.split('/')[-1]
                        d_filename = filename1.split('.')[0] + '_' + filename2
                        
                        d_folder = '/'.join(file1.split('/')[:-1])
                        
                        deformed_image.save(os.path.join(os.path.join(os.path.join(args.casia_dir, 'images_iso'), d_folder), d_filename))
                        
                        dnet_pair = d_folder + '/' + d_filename  + ',' + file2
                        if alpha1 < alpha2:
                            sbdngpFile.write(dnet_pair + '\n')
                        else:
                            bsdngpFile.write(dnet_pair + '\n')
                        
                        pupil_iris_xyrs[d_folder + '/' + d_filename] = {}
                        pupil_iris_xyrs[d_folder + '/' + d_filename]['pxyr'] = pxyr2
                        pupil_iris_xyrs[d_folder + '/' + d_filename]['ixyr'] = ixyr2
                        
                        deformed_mask.save(os.path.join(os.path.join(os.path.join(args.casia_dir, 'masks_fine'), d_folder), d_filename))
                        deformed_polar_image.save(os.path.join(os.path.join(os.path.join(args.casia_dir, 'images_polar'), d_folder), d_filename))
                        deformed_polar_mask.save(os.path.join(os.path.join(os.path.join(args.casia_dir, 'masks_polar'), d_folder), d_filename))
                        
                        img1, mask1, pxyr1, ixyr1, file1, alpha1, img2, mask2, pxyr2, ixyr2, file2, alpha2 = swap(img1, mask1, pxyr1, ixyr1, file1, alpha1, img2, mask2, pxyr2, ixyr2, file2, alpha2)
                        
                        if args.model_type != 'stylegan3':
                            deformed_image, deformed_mask, deformed_polar_image, deformed_polar_mask = deform(img1, mask1, pxyr1, ixyr1, img2, mask2, pxyr2, ixyr2, sifLossModel, deformer, deform_net, args.model_type, device)
                        else:
                            deformed_image, deformed_mask, deformed_polar_image, deformed_polar_mask = deform(img1, mask1, pxyr1, ixyr1, img2, mask2, pxyr2, ixyr2, sifLossModel, deformer, deform_net, args.model_type, device, G)
                        
                        filename1 = file1.split('/')[-1]
                        filename2 = file2.split('/')[-1]
                        d_filename = filename1.split('.')[0] + '_' + filename2
                        
                        d_folder = '/'.join(file1.split('/')[:-1])
                        
                        deformed_image.save(os.path.join(os.path.join(os.path.join(args.casia_dir, 'images_iso'), d_folder), d_filename))
                        
                        dnet_pair = d_folder + '/' + d_filename  + ',' + file2
                        if alpha1 < alpha2:
                            sbdngpFile.write(dnet_pair + '\n')
                        else:
                            bsdngpFile.write(dnet_pair + '\n')
                        
                        pupil_iris_xyrs[d_folder + '/' + d_filename] = {}
                        pupil_iris_xyrs[d_folder + '/' + d_filename]['pxyr'] = pxyr2
                        pupil_iris_xyrs[d_folder + '/' + d_filename]['ixyr'] = ixyr2
                        
                        deformed_mask.save(os.path.join(os.path.join(os.path.join(args.casia_dir, 'masks_fine'), d_folder), d_filename))
                        deformed_polar_image.save(os.path.join(os.path.join(os.path.join(args.casia_dir, 'images_polar'), d_folder), d_filename))
                        deformed_polar_mask.save(os.path.join(os.path.join(os.path.join(args.casia_dir, 'masks_polar'), d_folder), d_filename))
                        
    
    with open(os.path.join(args.casia_dir, 'pupil_iris_xyrs_full.pkl'), 'wb') as fullCircleFile:
        pkl.dump(pupil_iris_xyrs, fullCircleFile)

def evaluate_wsd_graph(args):
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    deformer = LinearDeformer(device)
    
    filter_mat1 = io.loadmat(args.sif_filter_path1)['ICAtextureFilters']
    filter_mat2 = io.loadmat(args.sif_filter_path2)['ICAtextureFilters']
    sifLossModel = SIFLayerMaskOSIRIS(polar_height = 64, polar_width = 512, filter_mat1 = filter_mat1, filter_mat2 = filter_mat2, osiris_filters = args.osiris_filters, device=device).to(device)
    
    if args.model_type == 'stylegan3':
        deform_net = torch.load(args.weight_path)[0].to(device)
        with dnnlib.util.open_url(args.sg3_weight) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    else:
        if args.load_ema:
            ema = torch.load(args.weight_path)[0]
            deform_net = ema.ema_model.to(device)
        else:
            deform_net = torch.load(args.weight_path)[0].to(device)
    deform_net.eval()
    
    with open(args.test_bins_path_wsd, 'rb') as testbinsfile:
        test_bins = pkl.load(testbinsfile)
        
    with open(os.path.join(args.parent_dir_wsd, 'pupil_iris_xyrs_new.pkl'), 'rb') as circleFile:
        pupil_iris_xyrs = pkl.load(circleFile)
    
    #print(pupil_iris_xyrs.keys())
    
    test_identities = list(test_bins.keys())
    if not os.path.exists('./wpd_eval/'):
        os.mkdir('./wpd_eval/')
    
    frame_range = [381, 440]
    
    for identity in tqdm(test_identities):
        for session in ['1', '2']:
            if not os.path.exists('./wpd_eval/'+identity):
                os.mkdir('./wpd_eval/'+identity+'_'+session)
                os.mkdir('./wpd_eval/'+identity+'_'+session+'/original_images/')
                os.mkdir('./wpd_eval/'+identity+'_'+session+'/original_images_polar/')
                os.mkdir('./wpd_eval/'+identity+'_'+session+'/original_masks/')
                os.mkdir('./wpd_eval/'+identity+'_'+session+'/original_masks_polar/')
                os.mkdir('./wpd_eval/'+identity+'_'+session+'/deformed_images/')
                os.mkdir('./wpd_eval/'+identity+'_'+session+'/deformed_images_polar/')
                os.mkdir('./wpd_eval/'+identity+'_'+session+'/deformed_masks/')
                os.mkdir('./wpd_eval/'+identity+'_'+session+'/deformed_masks_polar/')
            
            image_list = []
            initial_image_path = args.image_dir_wsd + identity.split('_')[0] + '/' + identity + '_'+session+'_380.bmp'
            if not os.path.exists(initial_image_path):
                print(initial_image_path)
                break
            initial_image_polar_path = args.image_polar_dir_wsd + identity.split('_')[0] + '/' + identity + '_' + session + '_380.bmp'
            initial_mask_path = args.mask_dir_wsd + identity.split('_')[0] + '/' + identity + '_' + session +'_380.bmp'
            initial_mask_polar_path = args.mask_polar_dir_wsd + identity.split('_')[0] + '/' + identity + '_' + session +'_380.bmp'
            shutil.copyfile(initial_image_polar_path, './wpd_eval/'+identity+'_'+session+ '/' + identity + '_' + session +'_380_image_polar.bmp')
            shutil.copyfile(initial_mask_polar_path, './wpd_eval/'+identity+'_'+session+ '/' + identity + '_' + session +'_380_mask_polar.bmp')
            
            initial_image = Image.open(initial_image_path).convert('L')
            initial_mask = Image.open(initial_mask_path).convert('L')
            initial_pxyr = np.copy(pupil_iris_xyrs[identity.split('_')[0] + '/' + identity + '_'+session+'_380']['pxyr'])
            initial_ixyr = np.copy(pupil_iris_xyrs[identity.split('_')[0] + '/' + identity + '_'+session+'_380']['ixyr'])
            
            initial_image_crop, initial_mask_crop, initial_pxyr_crop, initial_ixyr_crop = process_image_for_input(initial_image, initial_mask, initial_pxyr, initial_ixyr, 256)
            initial_image_crop.save('./wpd_eval/'+identity+'_'+session+ '/' + identity + '_' + session +'_380.bmp')
            initial_mask_crop.save('./wpd_eval/'+identity+'_'+session+ '/' + identity + '_' + session +'_380_mask.bmp')
            initial_alpha = float(initial_pxyr[2]) / float(initial_ixyr[2])
            
            for frame_no in tqdm(range(frame_range[0], frame_range[1]+1)):
                image_path = args.image_dir_wsd + identity.split('_')[0] + '/' + identity + '_'+session+'_' + str(frame_no) + '.bmp'
                if not os.path.exists(image_path):
                    print(image_path)
                    break
                image_polar_path = args.image_polar_dir_wsd + identity.split('_')[0] + '/' + identity + '_' + session + '_' + str(frame_no) + '.bmp'
                mask_path = args.mask_dir_wsd + identity.split('_')[0] + '/' + identity + '_' + session + '_' + str(frame_no) + '.bmp'
                mask_polar_path = args.mask_polar_dir_wsd + identity.split('_')[0] + '/' + identity + '_' + session + '_' + str(frame_no) + '.bmp'
                shutil.copyfile(image_polar_path, './wpd_eval/'+identity+'_'+session+'/original_images_polar/'+ identity + '_' + session + '_' + str(frame_no) + '.bmp')
                shutil.copyfile(mask_polar_path, './wpd_eval/'+identity+'_'+session+'/original_masks_polar/'+ identity + '_' + session + '_' + str(frame_no) + '.bmp')
                
                image = Image.open(image_path).convert('L')
                mask = Image.open(mask_path).convert('L')
                pxyr = np.copy(pupil_iris_xyrs[identity.split('_')[0] + '/' + identity + '_' + session + '_' + str(frame_no)]['pxyr'])
                ixyr = np.copy(pupil_iris_xyrs[identity.split('_')[0] + '/' + identity + '_' + session + '_' + str(frame_no)]['ixyr'])
                
                image_crop, mask_crop, pxyr_crop, ixyr_crop = process_image_for_input(image, mask, pxyr, ixyr, 256)
                image_crop.save('./wpd_eval/'+identity+'_'+session+'/original_images/'+ identity + '_' + session + '_' + str(frame_no) + '.bmp')
                mask_crop.save( './wpd_eval/'+identity+'_'+session+'/original_masks/'+ identity + '_' + session + '_' + str(frame_no) + '.bmp')
                alpha = float(pxyr[2]) / float(ixyr[2])
                
                if args.model_type != 'stylegan3':
                    deformed_image, deformed_mask, deformed_image_polar, deformed_mask_polar = deform(initial_image, initial_mask, initial_pxyr, initial_ixyr, image, mask, pxyr, ixyr, sifLossModel, deformer, deform_net, args.model_type, device)
                else:
                    deformed_image, deformed_mask, deformed_image_polar, deformed_mask_polar = deform(initial_image, initial_mask, initial_pxyr, initial_ixyr, image, mask, pxyr, ixyr, sifLossModel, deformer, deform_net, args.model_type, device, G)
                                
                deformed_image.save('./wpd_eval/'+identity+'_'+session+'/deformed_images/'+ identity + '_' + session + '_' + str(frame_no) + '.bmp')
                deformed_mask.save('./wpd_eval/'+identity+'_'+session+'/deformed_masks/'+ identity + '_' + session + '_' + str(frame_no) + '.bmp')
                deformed_image_polar.save('./wpd_eval/'+identity+'_'+session+'/deformed_images_polar/'+ identity + '_' + session + '_' + str(frame_no) + '.bmp')
                deformed_mask_polar.save('./wpd_eval/'+identity+'_'+session+'/deformed_masks_polar/'+ identity + '_' + session + '_' + str(frame_no) + '.bmp')
                
def evaluate_wsd(args):
    # Declare Models
    
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    deformer = LinearDeformer(device)
    
    filter_mat1 = io.loadmat(args.sif_filter_path1)['ICAtextureFilters']
    filter_mat2 = io.loadmat(args.sif_filter_path2)['ICAtextureFilters']
    sifLossModel = SIFLayerMaskOSIRIS(polar_height = 64, polar_width = 512, filter_mat1 = filter_mat1, filter_mat2 = filter_mat2, osiris_filters = args.osiris_filters, device=device).to(device)
    
    if args.model_type == 'stylegan3':
        deform_net = torch.load(args.weight_path)[0].to(device)
        with dnnlib.util.open_url(args.sg3_weight) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    else:
        if args.load_ema:
            ema = torch.load(args.weight_path)[0]
            deform_net = ema.ema_model.to(device)
        else:
            deform_net = torch.load(args.weight_path)[0].to(device)
    deform_net.eval()
    
    with open(os.path.join(args.parent_dir_wsd, 'pupil_iris_xyrs_new.pkl'), 'rb') as circleFile:
        pupil_iris_xyrs = pkl.load(circleFile)
        
    irisRec = irisRecognition(get_cfg(args.cfg), use_hough=False)
    
    for delta in ['0.1', '0.2', '0.3', '0.4']:
        for pairFileName in tqdm(['sampled_filtered_' + delta +'_genuine_pairs.txt', 'sampled_filtered_' + delta +'_imposter_pairs.txt']):
            with open(os.path.join(args.parent_dir_wsd, 'dnet_bs_' + pairFileName), 'w') as bsdngpFile:
                with open(os.path.join(args.parent_dir_wsd, 'dnet_sb_' + pairFileName), 'w') as sbdngpFile:
                    with open(os.path.join(args.parent_dir_wsd, 'biomech_' + pairFileName), 'w') as bmFile:
                        with open(os.path.join(args.parent_dir_wsd, pairFileName), 'r') as pFile: 
                            for line in tqdm(pFile):
                                lineparts = line.strip().split(',')
                                
                                file1 = lineparts[0].strip()
                                file2 = lineparts[1].strip()
                                
                                img1 = Image.open(os.path.join(os.path.join(args.parent_dir_wsd, 'WarsawData2'), file1)).convert('L')
                                img2 = Image.open(os.path.join(os.path.join(args.parent_dir_wsd, 'WarsawData2'), file2)).convert('L')
                                
                                mask1 = Image.open(os.path.join(os.path.join(args.parent_dir_wsd, 'WarsawMasksCoarse'), file1)).convert('L')
                                mask2 = Image.open(os.path.join(os.path.join(args.parent_dir_wsd, 'WarsawMasksCoarse'), file2)).convert('L')
                                
                                pxyr1 = np.copy(pupil_iris_xyrs[file1.split('.')[0]]['pxyr'])
                                ixyr1 = np.copy(pupil_iris_xyrs[file1.split('.')[0]]['ixyr'])
                                pxyr2 = np.copy(pupil_iris_xyrs[file2.split('.')[0]]['pxyr'])
                                ixyr2 = np.copy(pupil_iris_xyrs[file2.split('.')[0]]['ixyr'])
                                
                                alpha1 = float(pxyr1[2]) / float(ixyr1[2])
                                alpha2 = float(pxyr2[2]) / float(ixyr2[2])
                                
                                if args.model_type != 'stylegan3':
                                    deformed_image, deformed_mask, deformed_polar_image, deformed_polar_mask = deform(img1, mask1, pxyr1, ixyr1, img2, mask2, pxyr2, ixyr2, sifLossModel, deformer, deform_net, args.model_type, device)
                                else:
                                    deformed_image, deformed_mask, deformed_polar_image, deformed_polar_mask = deform(img1, mask1, pxyr1, ixyr1, img2, mask2, pxyr2, ixyr2, sifLossModel, deformer, deform_net, args.model_type, device, G)
                                
                                filename1 = file1.split('/')[-1]
                                filename2 = file2.split('/')[-1]
                                d_filename = filename1.split('.')[0] + '_' + filename2
                                
                                d_folder = '/'.join(file1.split('/')[:-1])
                                
                                deformed_image.save(os.path.join(os.path.join(os.path.join(args.parent_dir_wsd, 'WarsawDataDeformed'), d_folder), d_filename))
                                
                                dnet_pair = d_folder + '/' + d_filename  + ',' + file2
                                if alpha1 < alpha2:
                                    sbdngpFile.write(dnet_pair + '\n')
                                    
                                    spr = float(alpha1 * 0.006)
                                    bpr = float(alpha2 * 0.006)
                                    biomech_radii = np.float32(Biomech(spr, bpr, 65).flatten()[1:])
                                    
                                    bm_deformed_polar_image, bm_deformed_polar_mask = irisRec.bioMechCartToPol_torch(img2, mask2, pxyr2, ixyr2, biomech_radii)
                                    bm_pair = d_folder + '/' + 'biomech_' + d_filename  + ',' + file2
                                    Image.fromarray(bm_deformed_polar_image, 'L').save(os.path.join(os.path.join(os.path.join(args.parent_dir_wsd, 'WarsawDataDeformedPolar'), d_folder), 'biomech_' + d_filename))     
                                    Image.fromarray(bm_deformed_polar_mask, 'L').save(os.path.join(os.path.join(os.path.join(args.parent_dir_wsd, 'WarsawMasksDeformedPolar'), d_folder), 'biomech_' + d_filename))  
                                    
                                    bmFile.write(bm_pair + '\n')  
                                    
                                else:
                                    bsdngpFile.write(dnet_pair + '\n')
                                
                                pupil_iris_xyrs[d_folder + '/' + d_filename] = {}
                                pupil_iris_xyrs[d_folder + '/' + d_filename]['pxyr'] = pxyr2
                                pupil_iris_xyrs[d_folder + '/' + d_filename]['ixyr'] = ixyr2
                                
                                deformed_mask.save(os.path.join(os.path.join(os.path.join(args.parent_dir_wsd, 'WarsawMasksDeformedFine'), d_folder), d_filename))
                                deformed_polar_image.save(os.path.join(os.path.join(os.path.join(args.parent_dir_wsd, 'WarsawDataDeformedPolar'), d_folder), d_filename))
                                deformed_polar_mask.save(os.path.join(os.path.join(os.path.join(args.parent_dir_wsd, 'WarsawMasksDeformedPolar'), d_folder), d_filename))
                                
                                img1, mask1, pxyr1, ixyr1, file1, alpha1, img2, mask2, pxyr2, ixyr2, file2, alpha2 = swap(img1, mask1, pxyr1, ixyr1, file1, alpha1, img2, mask2, pxyr2, ixyr2, file2, alpha2)
                                
                                if args.model_type != 'stylegan3':
                                    deformed_image, deformed_mask, deformed_polar_image, deformed_polar_mask = deform(img1, mask1, pxyr1, ixyr1, img2, mask2, pxyr2, ixyr2, sifLossModel, deformer, deform_net, args.model_type, device)
                                else:
                                    deformed_image, deformed_mask, deformed_polar_image, deformed_polar_mask = deform(img1, mask1, pxyr1, ixyr1, img2, mask2, pxyr2, ixyr2, sifLossModel, deformer, deform_net, args.model_type, device, G)
                                
                                filename1 = file1.split('/')[-1]
                                filename2 = file2.split('/')[-1]
                                d_filename = filename1.split('.')[0] + '_' + filename2
                                
                                d_folder = '/'.join(file1.split('/')[:-1])
                                
                                deformed_image.save(os.path.join(os.path.join(os.path.join(args.parent_dir_wsd, 'WarsawDataDeformed'), d_folder), d_filename))
                                
                                dnet_pair = d_folder + '/' + d_filename  + ',' + file2
                                if alpha1 < alpha2:
                                    sbdngpFile.write(dnet_pair + '\n')
                                    
                                    spr = float(alpha1 * 0.006)
                                    bpr = float(alpha2 * 0.006)
                                    biomech_radii = np.float32(Biomech(spr, bpr, 65).flatten()[1:])
                                    
                                    bm_deformed_polar_image, bm_deformed_polar_mask = irisRec.bioMechCartToPol_torch(img2, mask2, pxyr2, ixyr2, biomech_radii)
                                    bm_pair = d_folder + '/' + 'biomech_' + d_filename  + ',' + file2
                                    Image.fromarray(bm_deformed_polar_image, 'L').save(os.path.join(os.path.join(os.path.join(args.parent_dir_wsd, 'WarsawDataDeformedPolar'), d_folder), 'biomech_' + d_filename))     
                                    Image.fromarray(bm_deformed_polar_mask, 'L').save(os.path.join(os.path.join(os.path.join(args.parent_dir_wsd, 'WarsawMasksDeformedPolar'), d_folder), 'biomech_' + d_filename))  
                                    
                                    bmFile.write(bm_pair + '\n') 
                                      
                                else:
                                    bsdngpFile.write(dnet_pair + '\n')
                                
                                pupil_iris_xyrs[d_folder + '/' + d_filename] = {}
                                pupil_iris_xyrs[d_folder + '/' + d_filename]['pxyr'] = pxyr2
                                pupil_iris_xyrs[d_folder + '/' + d_filename]['ixyr'] = ixyr2
                                
                                deformed_mask.save(os.path.join(os.path.join(os.path.join(args.parent_dir_wsd, 'WarsawMasksDeformedFine'), d_folder), d_filename))
                                deformed_polar_image.save(os.path.join(os.path.join(os.path.join(args.parent_dir_wsd, 'WarsawDataDeformedPolar'), d_folder), d_filename))
                                deformed_polar_mask.save(os.path.join(os.path.join(os.path.join(args.parent_dir_wsd, 'WarsawMasksDeformedPolar'), d_folder), d_filename))
                                
            
        with open(os.path.join(args.casia_dir, 'pupil_iris_xyrs_full.pkl'), 'wb') as fullCircleFile:
            pkl.dump(pupil_iris_xyrs, fullCircleFile)

def evaluate_wsd_filtered(args):
    # Declare Models
    
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    deformer = LinearDeformer(device)
    
    filter_mat1 = io.loadmat(args.sif_filter_path1)['ICAtextureFilters']
    filter_mat2 = io.loadmat(args.sif_filter_path2)['ICAtextureFilters']
    sifLossModel = SIFLayerMaskOSIRIS(polar_height = 64, polar_width = 512, filter_mat1 = filter_mat1, filter_mat2 = filter_mat2, osiris_filters = args.osiris_filters, device=device).to(device)
    
    if args.model_type == 'stylegan3':
        deform_net = torch.load(args.weight_path)[0].to(device)
        with dnnlib.util.open_url(args.sg3_weight) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    else:
        if args.load_ema:
            ema = torch.load(args.weight_path)[0]
            deform_net = ema.ema_model.to(device)
        else:
            deform_net = torch.load(args.weight_path)[0].to(device)
    deform_net.eval()
    
    with open(os.path.join(args.parent_dir_wsd, 'pupil_iris_xyrs_new.pkl'), 'rb') as circleFile:
        pupil_iris_xyrs = pkl.load(circleFile)
    
    irisRec = irisRecognition(get_cfg(args.cfg), use_hough=False)
   
    for pairFileName in tqdm(['genuine_pairs.txt', 'imposter_pairs.txt']):
        with open(os.path.join(args.parent_dir_wsd, 'dnet_bs_' + pairFileName), 'w') as bsdngpFile:
            with open(os.path.join(args.parent_dir_wsd, 'dnet_sb_' + pairFileName), 'w') as sbdngpFile:
                with open(os.path.join(args.parent_dir_wsd, 'biomech_' + pairFileName), 'w') as bmFile:
                    with open(os.path.join(args.parent_dir_wsd, pairFileName), 'r') as pFile: 
                        for line in tqdm(pFile):
                            lineparts = line.strip().split(',')
                            
                            file1 = lineparts[0].strip()
                            file2 = lineparts[1].strip()
                            
                            img1 = Image.open(os.path.join(os.path.join(args.parent_dir_wsd, 'WarsawData2'), file1)).convert('L')
                            img2 = Image.open(os.path.join(os.path.join(args.parent_dir_wsd, 'WarsawData2'), file2)).convert('L')
                            
                            mask1 = Image.open(os.path.join(os.path.join(args.parent_dir_wsd, 'WarsawMasksCoarse'), file1)).convert('L')
                            mask2 = Image.open(os.path.join(os.path.join(args.parent_dir_wsd, 'WarsawMasksCoarse'), file2)).convert('L')
                            
                            pxyr1 = np.copy(pupil_iris_xyrs[file1.split('.')[0]]['pxyr'])
                            ixyr1 = np.copy(pupil_iris_xyrs[file1.split('.')[0]]['ixyr'])
                            pxyr2 = np.copy(pupil_iris_xyrs[file2.split('.')[0]]['pxyr'])
                            ixyr2 = np.copy(pupil_iris_xyrs[file2.split('.')[0]]['ixyr'])
                            
                            alpha1 = float(pxyr1[2]) / float(ixyr1[2])
                            alpha2 = float(pxyr2[2]) / float(ixyr2[2])
                            
                            if args.model_type != 'stylegan3':
                                deformed_image, deformed_mask, deformed_polar_image, deformed_polar_mask = deform(img1, mask1, pxyr1, ixyr1, img2, mask2, pxyr2, ixyr2, sifLossModel, deformer, deform_net, args.model_type, device)
                            else:
                                deformed_image, deformed_mask, deformed_polar_image, deformed_polar_mask = deform(img1, mask1, pxyr1, ixyr1, img2, mask2, pxyr2, ixyr2, sifLossModel, deformer, deform_net, args.model_type, device, G)
                            
                            filename1 = file1.split('/')[-1]
                            filename2 = file2.split('/')[-1]
                            d_filename = filename1.split('.')[0] + '_' + filename2
                            
                            d_folder = '/'.join(file1.split('/')[:-1])
                            
                            deformed_image.save(os.path.join(os.path.join(os.path.join(args.parent_dir_wsd, 'WarsawDataDeformed'), d_folder), d_filename))
                            
                            dnet_pair = d_folder + '/' + d_filename  + ',' + file2
                            if alpha1 < alpha2:
                                sbdngpFile.write(dnet_pair + '\n')
                
                                spr = float(alpha1 * 0.006)
                                bpr = float(alpha2 * 0.006)
                                biomech_radii = np.float32(Biomech(spr, bpr, 65).flatten()[1:])
                                
                                bm_deformed_polar_image, bm_deformed_polar_mask = irisRec.bioMechCartToPol_torch(img2, mask2, pxyr2, ixyr2, biomech_radii)
                                bm_pair = d_folder + '/' + 'biomech_' + d_filename  + ',' + file2
                                Image.fromarray(bm_deformed_polar_image, 'L').save(os.path.join(os.path.join(os.path.join(args.parent_dir_wsd, 'WarsawDataDeformedPolar'), d_folder), 'biomech_' + d_filename))     
                                Image.fromarray(bm_deformed_polar_mask, 'L').save(os.path.join(os.path.join(os.path.join(args.parent_dir_wsd, 'WarsawMasksDeformedPolar'), d_folder), 'biomech_' + d_filename))  
                                
                                bmFile.write(bm_pair + '\n')                          
                                
                            else:
                                bsdngpFile.write(dnet_pair + '\n')
                            
                            pupil_iris_xyrs[d_folder + '/' + d_filename] = {}
                            pupil_iris_xyrs[d_folder + '/' + d_filename]['pxyr'] = pxyr2
                            pupil_iris_xyrs[d_folder + '/' + d_filename]['ixyr'] = ixyr2
                            
                            deformed_mask.save(os.path.join(os.path.join(os.path.join(args.parent_dir_wsd, 'WarsawMasksDeformedFine'), d_folder), d_filename))
                            deformed_polar_image.save(os.path.join(os.path.join(os.path.join(args.parent_dir_wsd, 'WarsawDataDeformedPolar'), d_folder), d_filename))
                            deformed_polar_mask.save(os.path.join(os.path.join(os.path.join(args.parent_dir_wsd, 'WarsawMasksDeformedPolar'), d_folder), d_filename))
                            
                            img1, mask1, pxyr1, ixyr1, file1, alpha1, img2, mask2, pxyr2, ixyr2, file2, alpha2 = swap(img1, mask1, pxyr1, ixyr1, file1, alpha1, img2, mask2, pxyr2, ixyr2, file2, alpha2)
                            
                            if args.model_type != 'stylegan3':
                                deformed_image, deformed_mask, deformed_polar_image, deformed_polar_mask = deform(img1, mask1, pxyr1, ixyr1, img2, mask2, pxyr2, ixyr2, sifLossModel, deformer, deform_net, args.model_type, device)
                            else:
                                deformed_image, deformed_mask, deformed_polar_image, deformed_polar_mask = deform(img1, mask1, pxyr1, ixyr1, img2, mask2, pxyr2, ixyr2, sifLossModel, deformer, deform_net, args.model_type, device, G)
                            
                            filename1 = file1.split('/')[-1]
                            filename2 = file2.split('/')[-1]
                            d_filename = filename1.split('.')[0] + '_' + filename2
                            
                            d_folder = '/'.join(file1.split('/')[:-1])
                            
                            deformed_image.save(os.path.join(os.path.join(os.path.join(args.parent_dir_wsd, 'WarsawDataDeformed'), d_folder), d_filename))
                            
                            dnet_pair = d_folder + '/' + d_filename  + ',' + file2
                            if alpha1 < alpha2:
                                sbdngpFile.write(dnet_pair + '\n')
                                
                                spr = float(alpha1 * 0.006)
                                bpr = float(alpha2 * 0.006)
                                biomech_radii = np.float32(Biomech(spr, bpr, 65).flatten()[1:])
                                
                                bm_deformed_polar_image, bm_deformed_polar_mask = irisRec.bioMechCartToPol_torch(img2, mask2, pxyr2, ixyr2, biomech_radii)
                                bm_pair = d_folder + '/' + 'biomech_' + d_filename  + ',' + file2
                                Image.fromarray(bm_deformed_polar_image, 'L').save(os.path.join(os.path.join(os.path.join(args.parent_dir_wsd, 'WarsawDataDeformedPolar'), d_folder), 'biomech_' + d_filename))     
                                Image.fromarray(bm_deformed_polar_mask, 'L').save(os.path.join(os.path.join(os.path.join(args.parent_dir_wsd, 'WarsawMasksDeformedPolar'), d_folder), 'biomech_' + d_filename))  
                                
                                bmFile.write(bm_pair + '\n')
                            else:
                                bsdngpFile.write(dnet_pair + '\n')
                            
                            pupil_iris_xyrs[d_folder + '/' + d_filename] = {}
                            pupil_iris_xyrs[d_folder + '/' + d_filename]['pxyr'] = pxyr2
                            pupil_iris_xyrs[d_folder + '/' + d_filename]['ixyr'] = ixyr2
                            
                            deformed_mask.save(os.path.join(os.path.join(os.path.join(args.parent_dir_wsd, 'WarsawMasksDeformedFine'), d_folder), d_filename))
                            deformed_polar_image.save(os.path.join(os.path.join(os.path.join(args.parent_dir_wsd, 'WarsawDataDeformedPolar'), d_folder), d_filename))
                            deformed_polar_mask.save(os.path.join(os.path.join(os.path.join(args.parent_dir_wsd, 'WarsawMasksDeformedPolar'), d_folder), d_filename))
                            
        
    with open(os.path.join(args.casia_dir, 'pupil_iris_xyrs_full.pkl'), 'wb') as fullCircleFile:
        pkl.dump(pupil_iris_xyrs, fullCircleFile)


def evaluate_wsd_filtered_only_biomech(args):
    # Declare Models
    
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    with open(os.path.join(args.parent_dir_wsd, 'pupil_iris_xyrs_new.pkl'), 'rb') as circleFile:
        pupil_iris_xyrs = pkl.load(circleFile)
    
    irisRec = irisRecognition(get_cfg(args.cfg), use_hough=False)
   
    for pairFileName in tqdm(['genuine_pairs.txt', 'imposter_pairs.txt']):
        with open(os.path.join(args.parent_dir_wsd, 'biomech_' + pairFileName), 'w') as bmFile:
            with open(os.path.join(args.parent_dir_wsd, pairFileName), 'r') as pFile: 
                for line in tqdm(pFile):
                    lineparts = line.strip().split(',')
                    
                    file1 = lineparts[0].strip()
                    file2 = lineparts[1].strip()
                    
                    img1 = Image.open(os.path.join(os.path.join(args.parent_dir_wsd, 'WarsawData2'), file1)).convert('L')
                    img2 = Image.open(os.path.join(os.path.join(args.parent_dir_wsd, 'WarsawData2'), file2)).convert('L')
                    
                    mask1 = Image.open(os.path.join(os.path.join(args.parent_dir_wsd, 'WarsawMasksCoarse'), file1)).convert('L')
                    mask2 = Image.open(os.path.join(os.path.join(args.parent_dir_wsd, 'WarsawMasksCoarse'), file2)).convert('L')
                    
                    pxyr1 = np.copy(pupil_iris_xyrs[file1.split('.')[0]]['pxyr'])
                    ixyr1 = np.copy(pupil_iris_xyrs[file1.split('.')[0]]['ixyr'])
                    pxyr2 = np.copy(pupil_iris_xyrs[file2.split('.')[0]]['pxyr'])
                    ixyr2 = np.copy(pupil_iris_xyrs[file2.split('.')[0]]['ixyr'])
                    
                    alpha1 = float(pxyr1[2]) / float(ixyr1[2])
                    alpha2 = float(pxyr2[2]) / float(ixyr2[2])
                    
                    filename1 = file1.split('/')[-1]
                    filename2 = file2.split('/')[-1]
                    d_filename = filename1.split('.')[0] + '_' + filename2
                    
                    d_folder = '/'.join(file1.split('/')[:-1])
                    
                    if alpha1 < alpha2:        
                        spr = float(alpha1 * 0.006)
                        bpr = float(alpha2 * 0.006)
                        biomech_radii = np.float32(Biomech(spr, bpr, 65).flatten()[1:])
                        
                        bm_deformed_polar_image, bm_deformed_polar_mask = irisRec.bioMechCartToPol_torch(img2, mask2, pxyr2, ixyr2, biomech_radii)
                        bm_pair = d_folder + '/' + 'biomech_' + d_filename  + ',' + file2
                        Image.fromarray(bm_deformed_polar_image, 'L').save(os.path.join(os.path.join(os.path.join(args.parent_dir_wsd, 'WarsawDataDeformedPolar'), d_folder), 'biomech_' + d_filename))     
                        Image.fromarray(bm_deformed_polar_mask, 'L').save(os.path.join(os.path.join(os.path.join(args.parent_dir_wsd, 'WarsawMasksDeformedPolar'), d_folder), 'biomech_' + d_filename))  
                        
                        bmFile.write(bm_pair + '\n')                          
                    
                    img1, mask1, pxyr1, ixyr1, file1, alpha1, img2, mask2, pxyr2, ixyr2, file2, alpha2 = swap(img1, mask1, pxyr1, ixyr1, file1, alpha1, img2, mask2, pxyr2, ixyr2, file2, alpha2)
                    
                    filename1 = file1.split('/')[-1]
                    filename2 = file2.split('/')[-1]
                    d_filename = filename1.split('.')[0] + '_' + filename2
                    
                    d_folder = '/'.join(file1.split('/')[:-1])
                    
                    if alpha1 < alpha2:                        
                        spr = float(alpha1 * 0.006)
                        bpr = float(alpha2 * 0.006)
                        biomech_radii = np.float32(Biomech(spr, bpr, 65).flatten()[1:])
                        
                        bm_deformed_polar_image, bm_deformed_polar_mask = irisRec.bioMechCartToPol_torch(img2, mask2, pxyr2, ixyr2, biomech_radii)
                        bm_pair = d_folder + '/' + 'biomech_' + d_filename  + ',' + file2
                        Image.fromarray(bm_deformed_polar_image, 'L').save(os.path.join(os.path.join(os.path.join(args.parent_dir_wsd, 'WarsawDataDeformedPolar'), d_folder), 'biomech_' + d_filename))     
                        Image.fromarray(bm_deformed_polar_mask, 'L').save(os.path.join(os.path.join(os.path.join(args.parent_dir_wsd, 'WarsawMasksDeformedPolar'), d_folder), 'biomech_' + d_filename))  
                        
                        bmFile.write(bm_pair + '\n')
            

if __name__ == '__main__':
    #print('Its running')
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--cudnn', action='store_true')
    parser.add_argument('--cfg', type=str, default='./configs/cfg.yaml')
    parser.add_argument('--parent_dir_wsd', type=str, default='')
    parser.add_argument('--image_dir_wsd', type=str, default="")
    parser.add_argument('--image_polar_dir_wsd', type=str, default="")
    parser.add_argument('--mask_dir_wsd', type=str, default="")
    parser.add_argument('--mask_polar_dir_wsd', type=str, default="")
    parser.add_argument('--test_bins_path_wsd', type=str, default='')
    parser.add_argument('--hollingsworth_dir', type=str, default='')
    parser.add_argument('--qfire_dir', type=str, default='')
    parser.add_argument('--casia_dir', type=str, default='')
    parser.add_argument('--model_type', type=str, default='not_stylegan3')
    parser.add_argument('--weight_path', type=str, default="")
    parser.add_argument('--evaluation_dataset', type=str, default='hollingsworth')
    parser.add_argument('--min_alpha_deform', type=float, default=0.01)
    parser.add_argument('--sg3_weight', type=str, default='')
    parser.add_argument('--sif_filter_path1', type=str, default='./filters/ICAtextureFilters_17x17_5bit.mat')
    parser.add_argument('--sif_filter_path2', type=str, default='./filters/ICAtextureFilters_15x15_7bit.mat')
    parser.add_argument('--osiris_filters', type=str, default='./filters/osiris_filters.txt')
    parser.add_argument('--load_ema', action='store_true')
    
    args = parser.parse_args()
    
    if args.cudnn:
        print('Using CUDNN')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = True
        
    if args.evaluation_dataset == 'hollingsworth':
        evaluate_hollingsworth(args)
    elif args.evaluation_dataset == 'hollingsworth_biomech':
        evaluate_hollingsworth_only_biomech(args)
    elif args.evaluation_dataset == 'hollingsworth_deformirisnet':
        evaluate_hollingsworth_deformirisnet(args)
    elif args.evaluation_dataset == 'qfire':
        evaluate_qfire(args)
    elif args.evaluation_dataset == 'wsd_graph':
        #evaluate_wsd_graph(args)
        evaluate_graph_new_wbpd(args)
    elif args.evaluation_dataset == 'wsd':
        evaluate_wsd(args)
    elif args.evaluation_dataset == 'wsd_filtered':
        evaluate_wsd_filtered(args)
    elif args.evaluation_dataset == 'wsd_filtered_biomech':
        evaluate_wsd_filtered_only_biomech(args)
    elif args.evaluation_dataset == 'casia':
        evaluate_casia(args)
    
        
    
    
    