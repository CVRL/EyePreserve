import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageDraw, ImageFilter
import torchvision.transforms as transforms
from tqdm import tqdm
import os
from PIL import Image
import random
import os
import pickle as pkl
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchvision import models, transforms
from math import pi
import os
import csv
import math
import random
from tqdm import tqdm

from PIL import Image
from argparse import ArgumentParser
import torch.nn as nn

from torch.optim import SGD, Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor, ToPILImage
import scipy
from scipy import io
#from modules.layers import ConvOffset2D

import math 
import numpy as np
import lpips
import json
import cv2

import imgaug.augmenters as iaa

'''
with open(os.path.join(hollingsworth_dir, 'infos.json')) as json_file:
            self.info = json.load(json_file)
        
        print('Finding all pairs...')
        self.pairs = []
        
        ids = list(self.info.keys())
        
        for i in tqdm(range(len(ids))):
            for j in range(i+1, len(ids)):
                id1 = ids[i]
                id2 = ids[j]
                alpha_id1 = float(self.info[id1]['pupil_xyr'][2]) / float(self.info[id1]['iris_xyr'][2])
                alpha_id2 = float(self.info[id2]['pupil_xyr'][2]) / float(self.info[id2]['ixyr'][2])
                if alpha_id1 <= alpha_id2:
                    self.pairs.append([id1, id2])
                else:
                    self.pairs.append([id2, id1])
'''

class PairFromBinDatasetSBWSD(Dataset):
    def __init__(self, bins_path_wsd, parent_dir_wsd, flip_data = True, res_mult=1):
        super().__init__()
        
        self.flip_data = flip_data   
        self.input_size=(int(240*res_mult),int(320*res_mult))
        self.res_mult = res_mult
        
        self.image_dir_wsd = os.path.join(parent_dir_wsd, 'WarsawData2')
        self.mask_dir_wsd = os.path.join(parent_dir_wsd, 'WarsawMask2')            
        
        with open(bins_path_wsd, 'rb') as wsdbinsFile:
            self.bins_wsd = pkl.load(wsdbinsFile)
        
        self.all_ids = list(self.bins_wsd.keys())
        
        print('Initializing pairs for Warsaw Pupil Dynamics...')
        self.pairs = []
        self.pair_ids = []
        
        for identifier in tqdm(self.bins_wsd.keys()):
            for bin_num_1 in self.bins_wsd[identifier].keys():
                for bin_num_2 in self.bins_wsd[identifier].keys():
                    if bin_num_1 < bin_num_2:
                        random.shuffle(self.bins_wsd[identifier][bin_num_1])
                        random.shuffle(self.bins_wsd[identifier][bin_num_2])
                        max_pairs = min(len(self.bins_wsd[identifier][bin_num_1]), len(self.bins_wsd[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            imagename1 = self.bins_wsd[identifier][bin_num_1][img_ind]
                            imagename2 = self.bins_wsd[identifier][bin_num_2][img_ind]
                            indiv_id1 = imagename1.split('_')[0]
                            indiv_id2 = imagename2.split('_')[0]
                            imagepath1 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id1), imagename1)
                            maskpath1 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id1), imagename1)
                            imagepath2 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id2), imagename2)
                            maskpath2 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id2), imagename2)
                            self.pairs.append([imagepath1, maskpath1, imagepath2, maskpath2])  
                            self.pair_ids.append(identifier)
        
        print('Total pairs: ', len(self.pairs))
        
        self.transform = transforms.Compose([
            transforms.Resize([self.input_size[0], self.input_size[1]]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ])
        
        self.pupil_iris_xyrs_path_wsd = os.path.join(parent_dir_wsd, 'pupil_iris_xyrs2.pkl')
        with open(self.pupil_iris_xyrs_path_wsd, 'rb') as pupilirisxyrsFile:
            self.pupil_iris_xyrs_wsd = pkl.load(pupilirisxyrsFile)
        
        self.pupil_xyrs = {}
        self.iris_xyrs = {}
        
        for imagename in self.pupil_iris_xyrs_wsd.keys():
            indiv_id = imagename.split('_')[0]
            imagepath = os.path.join(os.path.join(self.image_dir_wsd, indiv_id), imagename)
            self.pupil_xyrs[imagepath] = self.pupil_iris_xyrs_wsd[imagename]['pxyr']
            self.iris_xyrs[imagepath] = self.pupil_iris_xyrs_wsd[imagename]['ixyr']        
       
    
    def load_image(self, file):
        return Image.open(file).convert('L')
        
    def __getitem__(self, index):
    
        pair = self.pairs[index]
        
        inp_img_path = pair[0].strip()
        inp_mask_path = pair[1].strip()
        tar_img_path = pair[2].strip()
        tar_mask_path = pair[3].strip()
        
        inp_img = self.load_image(inp_img_path)
        inp_mask = self.load_image(inp_mask_path)
        
        tar_img = self.load_image(tar_img_path)
        tar_mask = self.load_image(tar_mask_path)
        
        inp_img_pupil_xyr = torch.FloatTensor(self.pupil_xyrs[inp_img_path])
        inp_img_iris_xyr = torch.FloatTensor(self.iris_xyrs[inp_img_path])
        
        tar_img_pupil_xyr = torch.FloatTensor(self.pupil_xyrs[tar_img_path])
        tar_img_iris_xyr = torch.FloatTensor(self.iris_xyrs[tar_img_path])
        
        sw, sh = inp_img.size
        sw_mult = (self.input_size[1]/sw)
        sh_mult = (self.input_size[0]/sh)
        
        bw, bh = tar_img.size
        bw_mult = (self.input_size[1]/bw)
        bh_mult = (self.input_size[0]/bh)
        
        inp_img_pupil_xyr[0] = (inp_img_pupil_xyr[0] * sw_mult)
        inp_img_pupil_xyr[1] = (inp_img_pupil_xyr[1] * sh_mult)
        inp_img_pupil_xyr[2] = (inp_img_pupil_xyr[2] * max(sw_mult, sh_mult))
        
        tar_img_pupil_xyr[0] = (tar_img_pupil_xyr[0] * bw_mult)
        tar_img_pupil_xyr[1] = (tar_img_pupil_xyr[1] * bh_mult)
        tar_img_pupil_xyr[2] = (tar_img_pupil_xyr[2] * max(bw_mult, bh_mult))
    
        inp_img_iris_xyr[0] = (inp_img_iris_xyr[0] * sw_mult)
        inp_img_iris_xyr[1] = (inp_img_iris_xyr[1] * sh_mult)
        inp_img_iris_xyr[2] = (inp_img_iris_xyr[2] * max(sw_mult, sh_mult))
        
        tar_img_iris_xyr[0] = (tar_img_iris_xyr[0] * bw_mult)
        tar_img_iris_xyr[1] = (tar_img_iris_xyr[1] * bh_mult)
        tar_img_iris_xyr[2] = (tar_img_iris_xyr[2] * max(bw_mult, bh_mult))
        
        if self.flip_data and random.random() < 0.5:
            inp_img = inp_img.transpose(Image.FLIP_LEFT_RIGHT) 
            inp_mask = inp_mask.transpose(Image.FLIP_LEFT_RIGHT)
            tar_img = tar_img.transpose(Image.FLIP_LEFT_RIGHT) 
            tar_mask = tar_mask.transpose(Image.FLIP_LEFT_RIGHT)
            inp_img_pupil_xyr[0] = self.input_size[1] - inp_img_pupil_xyr[0]
            tar_img_pupil_xyr[0] = self.input_size[1] - tar_img_pupil_xyr[0]
            inp_img_iris_xyr[0] = self.input_size[1] - inp_img_iris_xyr[0]
            tar_img_iris_xyr[0] = self.input_size[1] - tar_img_iris_xyr[0]

        xform_inp_img = self.transform(inp_img)
        xform_inp_mask = self.transform(inp_mask)
        xform_inp_mask[xform_inp_mask<0] = -1
        xform_inp_mask[xform_inp_mask>=0] = 1
        
        xform_tar_img = self.transform(tar_img)
        xform_tar_mask = self.transform(tar_mask)
        xform_tar_mask[xform_tar_mask<0] = -1
        xform_tar_mask[xform_tar_mask>=0] = 1
        
        return {'inp_img': xform_inp_img, 'tar_img' : xform_tar_img, 'inp_mask': xform_inp_mask, 'tar_mask' : xform_tar_mask, 'inp_img_pxyr' : inp_img_pupil_xyr, 'tar_img_pxyr' : tar_img_pupil_xyr, 'inp_img_ixyr': inp_img_iris_xyr, 'tar_img_ixyr' :  tar_img_iris_xyr, 'identifier':self.pair_ids[index] }
    
    def __len__(self):
        return len(self.pairs)
    
    def reset(self):
        print('Shuffling pairs for Warsaw Pupil Dynamics...')
        self.pairs = []
        self.pair_ids = []
        for identifier in tqdm(self.bins_wsd.keys()):
            for bin_num_1 in self.bins_wsd[identifier].keys():
                for bin_num_2 in self.bins_wsd[identifier].keys():
                    if bin_num_1 < bin_num_2:
                        random.shuffle(self.bins_wsd[identifier][bin_num_1])
                        random.shuffle(self.bins_wsd[identifier][bin_num_2])
                        max_pairs = min(len(self.bins_wsd[identifier][bin_num_1]), len(self.bins_wsd[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            imagename1 = self.bins_wsd[identifier][bin_num_1][img_ind]
                            imagename2 = self.bins_wsd[identifier][bin_num_2][img_ind]
                            indiv_id1 = imagename1.split('_')[0]
                            indiv_id2 = imagename2.split('_')[0]
                            imagepath1 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id1), imagename1)
                            maskpath1 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id1), imagename1)
                            imagepath2 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id2), imagename2)
                            maskpath2 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id2), imagename2)
                            self.pairs.append([imagepath1, maskpath1, imagepath2, maskpath2])  
                            self.pair_ids.append(identifier)                        
                            
        print('Total pairs: ', len(self.pairs))

class EvalPairMinMaxBinDataset(Dataset):
    def __init__(self, bins_path, parent_dir, max_pairs_inp = None, res_mult = 1):
        super().__init__()
        
        self.image_dir = os.path.join(parent_dir, 'WarsawData')
        self.mask_dir = os.path.join(parent_dir, 'WarsawMask')
        self.pupil_xyrs_path = os.path.join(parent_dir, 'WarsawPupilCircles.pkl')
        self.iris_xyrs_path = os.path.join(parent_dir, 'WarsawIrisCircles.pkl')
        self.max_pairs_inp = max_pairs_inp
        self.input_size=(int(240*res_mult),int(320*res_mult))
        self.res_mult = res_mult
           
        with open(bins_path, 'rb') as binsFile:
            self.bins = pkl.load(binsFile)
            
        with open(self.pupil_xyrs_path, 'rb') as pupilFile:
            self.pupil_xyrs = pkl.load(pupilFile)
        
        with open(self.iris_xyrs_path, 'rb') as irisFile:
            self.iris_xyrs = pkl.load(irisFile)
        
        print('Setting Dataset for Min-max pairs...')
        self.pairs = []
        self.pair_ids = []
        for identifier in tqdm(self.bins.keys()):
            min_bin_num = min(list(self.bins[identifier].keys()))
            max_bin_num = max(list(self.bins[identifier].keys()))
            random.shuffle(self.bins[identifier][min_bin_num])
            random.shuffle(self.bins[identifier][max_bin_num])
            if self.max_pairs_inp is None:
                max_pairs = min(len(self.bins[identifier][min_bin_num]), len(self.bins[identifier][max_bin_num]))
            else:
                max_pairs = min(self.max_pairs_inp, min(len(self.bins[identifier][min_bin_num]), len(self.bins[identifier][max_bin_num])))
            for img_ind in range(max_pairs):
                self.pairs.append([self.bins[identifier][min_bin_num][img_ind], self.bins[identifier][max_bin_num][img_ind]])
                self.pair_ids.append('_'.join(identifier.split('_')[:-1]))
    
    
    def load_image(self, file):
        return Image.open(file).convert('L')
        
    def __getitem__(self, index):
        
        pair = self.pairs[index]
    
        inp_img_name = pair[0].strip()
        tar_img_name = pair[1].strip()
        
        inp_img_pupil_xyr = torch.tensor(self.pupil_xyrs[inp_img_name]) * self.res_mult
        inp_img_iris_xyr = torch.tensor(self.iris_xyrs[inp_img_name]) * self.res_mult
        
        tar_img_pupil_xyr = torch.tensor(self.pupil_xyrs[tar_img_name]) * self.res_mult
        tar_img_iris_xyr = torch.tensor(self.iris_xyrs[tar_img_name]) * self.res_mult
        
        return {'inp_img_name':inp_img_name, 'tar_img_name':tar_img_name, 'inp_img_pxyr': inp_img_pupil_xyr, 'inp_img_ixyr' : inp_img_iris_xyr, 'tar_img_pxyr' : tar_img_pupil_xyr, 'tar_img_ixyr' :  tar_img_iris_xyr, 'identifier':self.pair_ids[index]}
        
    def __len__(self):
        return len(self.pairs)
    
    def reset(self):
        print('Setting Dataset for Min-max pairs...')
        self.pairs = []
        self.pair_ids = []
        for identifier in tqdm(self.bins.keys()):
            min_bin_num = min(list(self.bins[identifier].keys()))
            max_bin_num = max(list(self.bins[identifier].keys()))
            random.shuffle(self.bins[identifier][min_bin_num])
            random.shuffle(self.bins[identifier][max_bin_num])
            if self.max_pairs_inp is None:
                max_pairs = min(len(self.bins[identifier][min_bin_num]), len(self.bins[identifier][max_bin_num]))
            else:
                max_pairs = min(self.max_pairs_inp, min(len(self.bins[identifier][min_bin_num]), len(self.bins[identifier][max_bin_num])))
            for img_ind in range(max_pairs):
                self.pairs.append([self.bins[identifier][min_bin_num][img_ind], self.bins[identifier][max_bin_num][img_ind]])
                self.pair_ids.append('_'.join(identifier.split('_')[:-1]))

class EvalCASIADataset(Dataset):
    def __init__(self, casia_dir, res_mult=1):
        super().__init__()
        self.image_dir = os.path.join(casia_dir, 'images')
        self.mask_dir = os.path.join(casia_dir, 'masks')
        with open(os.path.join(casia_dir, 'infos.json')) as json_file:
            self.info = json.load(json_file)
        self.res_mult = res_mult
        self.pairs = []
        self.pair_ids = []
        
        print('Finding genuine pairs...')
        ids = list(self.info.keys())
        for i in tqdm(range(len(ids))):
            for j in range(i+1, len(ids)):
                id1 = ids[i]
                id2 = ids[j]
                if id1.split('_')[0] == id2.split('_')[0]:
                    
                    pupil_xyr_id1 = torch.tensor(self.info[id1]['pxyr']) * self.res_mult
                    iris_xyr_id1 = torch.tensor(self.info[id1]['ixyr']) * self.res_mult
                    
                    pupil_xyr_id2 =  torch.tensor(self.info[id2]['pxyr']) * self.res_mult
                    iris_xyr_id2 = torch.tensor(self.info[id2]['ixyr']) * self.res_mult
                    
                    id1_pxyr = (pupil_xyr_id1.clone().detach().cpu().numpy())
                    id1_ixyr = (iris_xyr_id1.clone().detach().cpu().numpy())
                    id2_pxyr = (pupil_xyr_id2.clone().detach().cpu().numpy())
                    id2_ixyr = (iris_xyr_id2.clone().detach().cpu().numpy())
                    
                    alpha_id1 = float(self.info[id1]['pxyr'][2]) / float(self.info[id1]['ixyr'][2]) * 0.006
                    alpha_id2 = float(self.info[id2]['pxyr'][2]) / float(self.info[id2]['ixyr'][2]) * 0.006
                    
                    if alpha_id1 <= alpha_id2:
                        self.pairs.append([id1, id2])
                    else:
                        self.pairs.append([id2, id1])
                    self.pair_ids.append(id1.split('_')[0])
                    
        self.input_size=(int(240*res_mult),int(320*res_mult))
        self.transform = transforms.Compose([
            transforms.Resize([self.input_size[0], self.input_size[1]]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ]) 
    
    def __len__(self):
        return len(self.pairs)
    
    def load_image(self, file):
        return Image.open(file).convert('L')
        
    def __getitem__(self, index):
    
        pair = self.pairs[index]
    
        s_id = pair[0].strip()
        b_id = pair[1].strip()
        
        inp_img = self.load_image(os.path.join(self.image_dir, self.info[s_id]['name']))
        inp_mask = self.load_image(os.path.join(self.mask_dir, self.info[s_id]['name']))
        
        tar_img =  self.load_image(os.path.join(self.image_dir, self.info[b_id]['name']))
        tar_mask = self.load_image(os.path.join(self.mask_dir, self.info[b_id]['name']))
               
        inp_img_pupil_xyr = torch.tensor(self.info[s_id]['pxyr']) * self.res_mult
        inp_img_iris_xyr = torch.tensor(self.info[s_id]['ixyr']) * self.res_mult
        
        tar_img_pupil_xyr =  torch.tensor(self.info[b_id]['pxyr']) * self.res_mult
        tar_img_iris_xyr = torch.tensor(self.info[b_id]['ixyr']) * self.res_mult
        
        xform_inp_img = self.transform(inp_img)
        xform_inp_mask = self.transform(inp_mask)
        xform_inp_mask[xform_inp_mask<0] = -1
        xform_inp_mask[xform_inp_mask>=0] = 1
        
        xform_tar_img = self.transform(tar_img)
        xform_tar_mask = self.transform(tar_mask)
        xform_tar_mask[xform_tar_mask<0] = -1
        xform_tar_mask[xform_tar_mask>=0] = 1                
        
        return {'inp_img_name':self.info[s_id]['name'], 'inp_img':xform_inp_img, 'inp_mask':xform_inp_mask,  'tar_img_name': self.info[b_id]['name'], 'tar_img':xform_tar_img, 'tar_mask':xform_tar_mask, 'inp_img_pxyr': inp_img_pupil_xyr, 'inp_img_ixyr' : inp_img_iris_xyr, 'tar_img_pxyr' : tar_img_pupil_xyr, 'tar_img_ixyr' :  tar_img_iris_xyr, 'identifier':self.pair_ids[index]}


class EvalHollingsworthDataset(Dataset):
    def __init__(self, hollingsworth_dir, res_mult=1):
        super().__init__()
        self.image_dir = os.path.join(hollingsworth_dir, 'recordings')
        self.mask_dir = os.path.join(hollingsworth_dir, 'masks')
        with open(os.path.join(hollingsworth_dir, 'infos.json')) as json_file:
            self.info = json.load(json_file)
        self.res_mult = res_mult
        self.pairs = []
        self.pair_ids = []
        
        print('Finding genuine pairs...')
        ids = list(self.info.keys())
        for i in tqdm(range(len(ids))):
            for j in range(i+1, len(ids)):
                id1 = ids[i]
                id2 = ids[j]
                if id1.split('_')[0] == id2.split('_')[0]:
                    
                    pupil_xyr_id1 = torch.tensor(self.info[id1]['pxyr']) * self.res_mult
                    iris_xyr_id1 = torch.tensor(self.info[id1]['ixyr']) * self.res_mult
                    
                    pupil_xyr_id2 =  torch.tensor(self.info[id2]['pxyr']) * self.res_mult
                    iris_xyr_id2 = torch.tensor(self.info[id2]['ixyr']) * self.res_mult
                    
                    id1_pxyr = (pupil_xyr_id1.clone().detach().cpu().numpy())
                    id1_ixyr = (iris_xyr_id1.clone().detach().cpu().numpy())
                    id2_pxyr = (pupil_xyr_id2.clone().detach().cpu().numpy())
                    id2_ixyr = (iris_xyr_id2.clone().detach().cpu().numpy())
                    
                    alpha_id1 = float(self.info[id1]['pxyr'][2]) / float(self.info[id1]['ixyr'][2]) * 0.006
                    alpha_id2 = float(self.info[id2]['pxyr'][2]) / float(self.info[id2]['ixyr'][2]) * 0.006
                    
                    if alpha_id1 <= alpha_id2:
                        self.pairs.append([id1, id2])
                    else:
                        self.pairs.append([id2, id1])
                    self.pair_ids.append(id1.split('_')[0])
                    
        self.input_size=(int(240*res_mult),int(320*res_mult))
        self.transform = transforms.Compose([
            transforms.Resize([self.input_size[0], self.input_size[1]]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ]) 
    
    def __len__(self):
        return len(self.pairs)
    
    def load_image(self, file):
        return Image.open(file).convert('L')
        
    def __getitem__(self, index):
    
        pair = self.pairs[index]
    
        s_id = pair[0].strip()
        b_id = pair[1].strip()
        
        inp_img = self.load_image(os.path.join(self.image_dir, self.info[s_id]['name']))
        inp_mask = self.load_image(os.path.join(self.mask_dir, self.info[s_id]['name']))
        
        tar_img =  self.load_image(os.path.join(self.image_dir, self.info[b_id]['name']))
        tar_mask = self.load_image(os.path.join(self.mask_dir, self.info[b_id]['name']))
               
        inp_img_pupil_xyr = torch.tensor(self.info[s_id]['pxyr']) * self.res_mult
        inp_img_iris_xyr = torch.tensor(self.info[s_id]['ixyr']) * self.res_mult
        
        tar_img_pupil_xyr =  torch.tensor(self.info[b_id]['pxyr']) * self.res_mult
        tar_img_iris_xyr = torch.tensor(self.info[b_id]['ixyr']) * self.res_mult
        
        xform_inp_img = self.transform(inp_img)
        xform_inp_mask = self.transform(inp_mask)
        xform_inp_mask[xform_inp_mask<0] = -1
        xform_inp_mask[xform_inp_mask>=0] = 1
        
        xform_tar_img = self.transform(tar_img)
        xform_tar_mask = self.transform(tar_mask)
        xform_tar_mask[xform_tar_mask<0] = -1
        xform_tar_mask[xform_tar_mask>=0] = 1                
        
        return {'inp_img_name':self.info[s_id]['name'], 's_id':s_id, 'b_id':b_id, 'inp_img':xform_inp_img, 'inp_mask':xform_inp_mask,  'tar_img_name': self.info[b_id]['name'], 'tar_img':xform_tar_img, 'tar_mask':xform_tar_mask, 'inp_img_pxyr': inp_img_pupil_xyr, 'inp_img_ixyr' : inp_img_iris_xyr, 'tar_img_pxyr' : tar_img_pupil_xyr, 'tar_img_ixyr' :  tar_img_iris_xyr, 'identifier':self.pair_ids[index]}

class EvalMinMaxAllPairsDataset(Dataset):
    def __init__(self, parent_dir, pairs, pair_ids, res_mult = 1):
        super().__init__()
        
        self.image_dir = os.path.join(parent_dir, 'WarsawData')
        self.mask_dir = os.path.join(parent_dir, 'WarsawMask')
        self.input_size=(int(240*res_mult),int(320*res_mult))
        self.res_mult = res_mult
           
        self.pairs = pairs
        self.pair_ids = pair_ids
        
        self.transform = transforms.Compose([
            transforms.Resize([self.input_size[0], self.input_size[1]]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ])

    
    def load_image(self, file):
        return Image.open(file).convert('L')
        
    def __getitem__(self, index):
    
        pair = self.pairs[index]
        
        inp_img_name = pair[0].strip()
        tar_img_name = pair[1].strip()
        
        indiv_id1 = inp_img_name.split('_')[0]
        indiv_id2 = tar_img_name.split('_')[0]
        
        inp_img_path = os.path.join(os.path.join(self.image_dir, indiv_id1), inp_img_name)
        
        tar_mask_path = os.path.join(os.path.join(self.mask_dir, indiv_id2), tar_img_name)
        
        inp_img = self.load_image(inp_img_path)
        
        tar_mask = self.load_image(tar_mask_path)
        
        xform_inp_img = self.transform(inp_img)
        
        xform_tar_mask = self.transform(tar_mask)
        xform_tar_mask[xform_tar_mask<0] = -1
        xform_tar_mask[xform_tar_mask>=0] = 1
        
        return {'inp_img_name':inp_img_name, 'inp_img': xform_inp_img, 'tar_img_name':tar_img_name, 'tar_mask' : xform_tar_mask, 'identifier':self.pair_ids[index]}
    
    def __len__(self):
        return len(self.pairs)


class HollingsworthDataset(Dataset):
    def __init__(self, hollingsworth_dir, res_mult=1, flip_data = False):
        super().__init__()
        
        self.input_size=(int(240*res_mult),int(320*res_mult))
        self.transform = transforms.Compose([
            transforms.Resize([self.input_size[0], self.input_size[1]]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ])
        
        self.image_dir = os.path.join(hollingsworth_dir, 'recordings')
        self.mask_dir = os.path.join(hollingsworth_dir, 'masks')
        self.flip_data = flip_data
        self.res_mult = res_mult
        with open(os.path.join(hollingsworth_dir, 'infos.json')) as json_file:
            self.info = json.load(json_file)
        
        print('Finding all pairs...')
        self.pairs = []
        self.identifiers = [] #stores if it's a genuine pair or imposter pair (0 for imposter, 1 for genuine)
        
        ids = list(self.info.keys())
        
        for i in tqdm(range(len(ids))):
            for j in range(i+1, len(ids)):
                id1 = ids[i]
                id2 = ids[j]
                if id1.split('_')[0] == id2.split('_')[0]:
                    alpha_id1 = float(self.info[id1]['pxyr'][2]) / float(self.info[id1]['ixyr'][2])
                    alpha_id2 = float(self.info[id2]['pxyr'][2]) / float(self.info[id2]['ixyr'][2])
                    if alpha_id1 <= alpha_id2:
                        self.pairs.append([id1, id2])
                    else:
                        self.pairs.append([id2, id1])
                    self.identifiers.append(id1.split('_')[0])
    
    def load_image(self, file):
        return Image.open(file).convert('L')
    
    def __len__(self):
        return len(self.pairs)
        
    def __getitem__(self, index):
    
        pair = self.pairs[index]
    
        s_id = pair[0].strip()
        b_id = pair[1].strip()
               
        inp_img = self.load_image(os.path.join(self.image_dir, self.info[s_id]['name']))
        inp_mask = self.load_image(os.path.join(self.mask_dir, self.info[s_id]['name']))
        
        tar_img =  self.load_image(os.path.join(self.image_dir, self.info[b_id]['name']))
        tar_mask = self.load_image(os.path.join(self.mask_dir, self.info[b_id]['name']))
        
        inp_img_pupil_xyr = torch.tensor(self.info[s_id]['pxyr']) * self.res_mult
        inp_img_iris_xyr = torch.tensor(self.info[s_id]['ixyr']) * self.res_mult
        
        tar_img_pupil_xyr =  torch.tensor(self.info[b_id]['pxyr']) * self.res_mult
        tar_img_iris_xyr = torch.tensor(self.info[b_id]['ixyr']) * self.res_mult
        
        if self.flip_data and random.random() < 0.5:
            inp_img = inp_img.transpose(Image.FLIP_LEFT_RIGHT) 
            inp_mask = inp_mask.transpose(Image.FLIP_LEFT_RIGHT)
            tar_img = tar_img.transpose(Image.FLIP_LEFT_RIGHT) 
            tar_mask = tar_mask.transpose(Image.FLIP_LEFT_RIGHT)
            inp_img_pupil_xyr[0] = self.input_size[1] - inp_img_pupil_xyr[0]
            tar_img_pupil_xyr[0] = self.input_size[1] - tar_img_pupil_xyr[0]
            inp_img_iris_xyr[0] = self.input_size[1] - inp_img_iris_xyr[0]
            tar_img_iris_xyr[0] = self.input_size[1] - tar_img_iris_xyr[0]
        
        xform_inp_img = self.transform(inp_img)
        xform_inp_mask = self.transform(inp_mask)
        xform_inp_mask[xform_inp_mask<0] = -1
        xform_inp_mask[xform_inp_mask>=0] = 1
        
        xform_tar_img = self.transform(tar_img)
        xform_tar_mask = self.transform(tar_mask)
        xform_tar_mask[xform_tar_mask<0] = -1
        xform_tar_mask[xform_tar_mask>=0] = 1
        
        return {'inp_img': xform_inp_img, 'inp_mask' : xform_inp_mask, 'tar_img' : xform_tar_img, 'tar_mask' : xform_tar_mask, 'inp_img_pxyr': inp_img_pupil_xyr, 'inp_img_ixyr' : inp_img_iris_xyr, 'tar_img_pxyr' : tar_img_pupil_xyr, 'tar_img_ixyr' :  tar_img_iris_xyr, 'identifier':self.identifiers[index] }
    
                    
        
class AllPairsDatasetSB(Dataset):
    def __init__(self, bins_path, parent_dir, flip_data = True, res_mult = 1):
        super().__init__()
        self.input_size=(int(240*res_mult),int(320*res_mult))
        self.res_mult = res_mult
        self.image_dir = os.path.join(parent_dir, 'WarsawData')
        self.mask_dir = os.path.join(parent_dir, 'WarsawMask')
        self.pupil_xyrs_path = os.path.join(parent_dir, 'WarsawPupilCircles.pkl')
        self.iris_xyrs_path = os.path.join(parent_dir, 'WarsawIrisCircles.pkl') 
        self.flip_data = flip_data
        with open(bins_path, 'rb') as binsFile:
            self.bins = pkl.load(binsFile)
        
        
        print('Finding all pairs...')
        self.pairs = []
        self.pair_identifiers = []
        for identifier in tqdm(self.bins.keys()):
            for bin_num_1 in self.bins[identifier].keys():
                for bin_num_2 in self.bins[identifier].keys():
                    if bin_num_1 < bin_num_2:
                        for img1 in self.bins[identifier][bin_num_1]:
                            for img2 in self.bins[identifier][bin_num_2]:
                                self.pairs.append([img1, img2])   
                                self.pair_identifiers.append(identifier)
        
        self.transform = transforms.Compose([
            transforms.Resize([self.input_size[0], self.input_size[1]]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ])
        
        with open(self.pupil_xyrs_path, 'rb') as pupilFile:
            self.pupil_xyrs = pkl.load(pupilFile)
        
        with open(self.iris_xyrs_path, 'rb') as irisFile:
            self.iris_xyrs = pkl.load(irisFile)

    
    def load_image(self, file):
        return Image.open(file).convert('L')
        
    def __getitem__(self, index):
    
        pair = self.pairs[index]
        
        inp_img_name = pair[0].strip()
        tar_img_name = pair[1].strip()
        
        indiv_id1 = inp_img_name.split('_')[0]
        indiv_id2 = tar_img_name.split('_')[0]
        
        inp_img_path = os.path.join(os.path.join(self.image_dir, indiv_id1), inp_img_name)
        inp_mask_path = os.path.join(os.path.join(self.mask_dir, indiv_id1), inp_img_name)
        
        tar_img_path = os.path.join(os.path.join(self.image_dir, indiv_id2), tar_img_name)
        tar_mask_path = os.path.join(os.path.join(self.mask_dir, indiv_id2), tar_img_name)
        
        inp_img = self.load_image(inp_img_path)
        inp_mask = self.load_image(inp_mask_path)
        
        tar_img = self.load_image(tar_img_path)
        tar_mask = self.load_image(tar_mask_path)
        
        inp_img_pupil_xyr = torch.tensor(self.pupil_xyrs[inp_img_name]) * self.res_mult
        inp_img_iris_xyr = torch.tensor(self.iris_xyrs[inp_img_name]) * self.res_mult
        
        tar_img_pupil_xyr = torch.tensor(self.pupil_xyrs[tar_img_name]) * self.res_mult
        tar_img_iris_xyr = torch.tensor(self.iris_xyrs[tar_img_name]) * self.res_mult
        
        if self.flip_data and random.random() < 0.5:
            inp_img = inp_img.transpose(Image.FLIP_LEFT_RIGHT) 
            inp_mask = inp_mask.transpose(Image.FLIP_LEFT_RIGHT)
            tar_img = tar_img.transpose(Image.FLIP_LEFT_RIGHT) 
            tar_mask = tar_mask.transpose(Image.FLIP_LEFT_RIGHT)
            inp_img_pupil_xyr[0] = self.input_size[1] - inp_img_pupil_xyr[0]
            tar_img_pupil_xyr[0] = self.input_size[1] - tar_img_pupil_xyr[0]
            inp_img_iris_xyr[0] = self.input_size[1] - inp_img_iris_xyr[0]
            tar_img_iris_xyr[0] = self.input_size[1] - tar_img_iris_xyr[0]
            
            
        xform_inp_img = self.transform(inp_img)
        xform_inp_mask = self.transform(inp_mask)
        xform_inp_mask[xform_inp_mask<0] = -1
        xform_inp_mask[xform_inp_mask>=0] = 1
        
        xform_tar_img = self.transform(tar_img)
        xform_tar_mask = self.transform(tar_mask)
        xform_tar_mask[xform_tar_mask<0] = -1
        xform_tar_mask[xform_tar_mask>=0] = 1
        
        return {'inp_img': xform_inp_img, 'inp_mask' : xform_inp_mask, 'tar_img' : xform_tar_img, 'tar_mask' : xform_tar_mask, 'inp_img_pxyr': inp_img_pupil_xyr, 'inp_img_ixyr' : inp_img_iris_xyr, 'tar_img_pxyr' : tar_img_pupil_xyr, 'tar_img_ixyr' :  tar_img_iris_xyr }
    
    def __len__(self):
        return len(self.pairs)
        

class PairMinMaxBinDatasetWSD(Dataset):
    def __init__(self, bins_path, parent_dir, max_pairs_inp = None, res_mult = 1):
        super().__init__()
        
        self.image_dir = os.path.join(parent_dir, 'WarsawData2')
        self.mask_dir = os.path.join(parent_dir, 'WarsawMask2')
        
        self.pupil_iris_xyrs_path = os.path.join(parent_dir, 'pupil_iris_xyrs2.pkl')
        with open(self.pupil_iris_xyrs_path, 'rb') as pupilirisxyrsFile:
            self.pupil_iris_xyrs = pkl.load(pupilirisxyrsFile)
        
        self.pupil_xyrs = {}
        self.iris_xyrs = {}
        
        for imagename in self.pupil_iris_xyrs.keys():
            indiv_id = imagename.split('_')[0]
            imagepath = os.path.join(os.path.join(self.image_dir, indiv_id), imagename)
            self.pupil_xyrs[imagepath] = self.pupil_iris_xyrs[imagename]['pxyr']
            self.iris_xyrs[imagepath] = self.pupil_iris_xyrs[imagename]['ixyr']
        
        self.max_pairs_inp = max_pairs_inp
        self.input_size=(int(240*res_mult),int(320*res_mult))
        self.res_mult = res_mult
           
        with open(bins_path, 'rb') as binsFile:
            self.bins = pkl.load(binsFile)
        
        print('Setting Dataset for Min-max pairs...')
        self.pairs = []
        self.pair_ids = []
        for identifier in tqdm(self.bins.keys()):
            min_bin_num = min(list(self.bins[identifier].keys()))
            max_bin_num = max(list(self.bins[identifier].keys()))
            
            while len(self.bins[identifier][max_bin_num]) < 40:
                max_bin_num -= 1
            
            if max_bin_num == min_bin_num:
                continue
            
            print('Min Bin Length:', len(self.bins[identifier][min_bin_num]))
            print('Max Bin Length:', len(self.bins[identifier][max_bin_num]))
            random.shuffle(self.bins[identifier][min_bin_num])
            random.shuffle(self.bins[identifier][max_bin_num])
            if self.max_pairs_inp is None:
                max_pairs = min(len(self.bins[identifier][min_bin_num]), len(self.bins[identifier][max_bin_num]))
            else:
                max_pairs = min(self.max_pairs_inp, min(len(self.bins[identifier][min_bin_num]), len(self.bins[identifier][max_bin_num])))
            for img_ind in range(max_pairs):
                self.pairs.append([self.bins[identifier][min_bin_num][img_ind], self.bins[identifier][max_bin_num][img_ind]])
                image1_id = '_'.join(self.bins[identifier][min_bin_num][img_ind].split('_')[:2])
                image2_id = '_'.join(self.bins[identifier][max_bin_num][img_ind].split('_')[:2])
                if image1_id != image2_id or image1_id != identifier:
                    print('############################    Something is seriously wrong    #################################################')
                    print(identifier, image1_id, image2_id)
                self.pair_ids.append(identifier)
        
        self.transform = transforms.Compose([
            transforms.Resize([self.input_size[0], self.input_size[1]]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ])
        
        print('Number of pairs: ', len(self.pairs))
    
    def load_image(self, file):
        return Image.open(file).convert('L')
        
    def __getitem__(self, index):
    
        pair = self.pairs[index]
        
        inp_img_name = pair[0].strip()
        tar_img_name = pair[1].strip()
        
        indiv_id1 = inp_img_name.split('_')[0]
        indiv_id2 = tar_img_name.split('_')[0]
        
        inp_img_path = os.path.join(os.path.join(self.image_dir, indiv_id1), inp_img_name)
        inp_mask_path = os.path.join(os.path.join(self.mask_dir, indiv_id1), inp_img_name)
        
        tar_img_path = os.path.join(os.path.join(self.image_dir, indiv_id2), tar_img_name)
        tar_mask_path = os.path.join(os.path.join(self.mask_dir, indiv_id2), tar_img_name)
        
        inp_img = self.load_image(inp_img_path)
        inp_mask = self.load_image(inp_mask_path)
        
        tar_img = self.load_image(tar_img_path)
        tar_mask = self.load_image(tar_mask_path)
        
        inp_img_pupil_xyr = torch.FloatTensor(self.pupil_xyrs[inp_img_path])
        inp_img_iris_xyr = torch.FloatTensor(self.iris_xyrs[inp_img_path])
        
        tar_img_pupil_xyr = torch.FloatTensor(self.pupil_xyrs[tar_img_path])
        tar_img_iris_xyr = torch.FloatTensor(self.iris_xyrs[tar_img_path])
        
        sw, sh = inp_img.size
        sw_mult = (self.input_size[1]/sw)
        sh_mult = (self.input_size[0]/sh)
        
        bw, bh = tar_img.size
        bw_mult = (self.input_size[1]/bw)
        bh_mult = (self.input_size[0]/bh)
        
        inp_img_pupil_xyr[0] = (inp_img_pupil_xyr[0] * sw_mult)
        inp_img_pupil_xyr[1] = (inp_img_pupil_xyr[1] * sh_mult)
        inp_img_pupil_xyr[2] = (inp_img_pupil_xyr[2] * max(sw_mult, sh_mult))
        
        tar_img_pupil_xyr[0] = (tar_img_pupil_xyr[0] * bw_mult)
        tar_img_pupil_xyr[1] = (tar_img_pupil_xyr[1] * bh_mult)
        tar_img_pupil_xyr[2] = (tar_img_pupil_xyr[2] * max(bw_mult, bh_mult))
    
        inp_img_iris_xyr[0] = (inp_img_iris_xyr[0] * sw_mult)
        inp_img_iris_xyr[1] = (inp_img_iris_xyr[1] * sh_mult)
        inp_img_iris_xyr[2] = (inp_img_iris_xyr[2] * max(sw_mult, sh_mult))
        
        tar_img_iris_xyr[0] = (tar_img_iris_xyr[0] * bw_mult)
        tar_img_iris_xyr[1] = (tar_img_iris_xyr[1] * bh_mult)
        tar_img_iris_xyr[2] = (tar_img_iris_xyr[2] * max(bw_mult, bh_mult))
        
        xform_inp_img = self.transform(inp_img)
        xform_inp_mask = self.transform(inp_mask)
        xform_inp_mask[xform_inp_mask<0] = -1
        xform_inp_mask[xform_inp_mask>=0] = 1
        
        xform_tar_img = self.transform(tar_img)
        xform_tar_mask = self.transform(tar_mask)
        xform_tar_mask[xform_tar_mask<0] = -1
        xform_tar_mask[xform_tar_mask>=0] = 1
        
        return {'inp_img_name': inp_img_name, 'tar_img_name':tar_img_name, 'inp_img': xform_inp_img, 'inp_mask' : xform_inp_mask, 'tar_img' : xform_tar_img, 'tar_mask' : xform_tar_mask, 'inp_img_pxyr': inp_img_pupil_xyr, 'inp_img_ixyr' : inp_img_iris_xyr, 'tar_img_pxyr' : tar_img_pupil_xyr, 'tar_img_ixyr' :  tar_img_iris_xyr, 'identifier':self.pair_ids[index]}
        
    
    def __len__(self):
        return len(self.pairs)
    
    def reset(self):
        print('Setting Dataset for Min-max pairs...')
        self.pairs = []
        self.pair_ids = []
        for identifier in tqdm(self.bins.keys()):
            min_bin_num = min(list(self.bins[identifier].keys()))
            max_bin_num = max(list(self.bins[identifier].keys()))
            random.shuffle(self.bins[identifier][min_bin_num])
            random.shuffle(self.bins[identifier][max_bin_num])
            if self.max_pairs_inp is None:
                max_pairs = min(len(self.bins[identifier][min_bin_num]), len(self.bins[identifier][max_bin_num]))
            else:
                max_pairs = min(self.max_pairs_inp, min(len(self.bins[identifier][min_bin_num]), len(self.bins[identifier][max_bin_num])))
            for img_ind in range(max_pairs):
                self.pairs.append([self.bins[identifier][min_bin_num][img_ind], self.bins[identifier][max_bin_num][img_ind]])
                self.pair_ids.append(identifier)
                
class PairMaxMinBinDataset(Dataset):
    def __init__(self, bins_path, parent_dir, max_pairs_inp = None, res_mult=1):
        super().__init__()
        
        self.image_dir = os.path.join(parent_dir, 'WarsawData')
        self.mask_dir = os.path.join(parent_dir, 'WarsawMask')
        self.pupil_xyrs_path = os.path.join(parent_dir, 'WarsawPupilCircles.pkl')
        self.iris_xyrs_path = os.path.join(parent_dir, 'WarsawIrisCircles.pkl')
        self.max_pairs_inp = max_pairs_inp
        self.input_size=(int(240*res_mult),int(320*res_mult))
        self.res_mult = res_mult
        
        with open(bins_path, 'rb') as binsFile:
            self.bins = pkl.load(binsFile)
        
        print('Setting Dataset for Min-max pairs...')
        self.pairs = []
        self.pair_ids = []
        for identifier in tqdm(self.bins.keys()):
            min_bin_num = min(list(self.bins[identifier].keys()))
            max_bin_num = max(list(self.bins[identifier].keys()))
            random.shuffle(self.bins[identifier][min_bin_num])
            random.shuffle(self.bins[identifier][max_bin_num])
            if self.max_pairs_inp is None:
                max_pairs = min(len(self.bins[identifier][max_bin_num]), len(self.bins[identifier][min_bin_num]))
            else:
                max_pairs = min(self.max_pairs_inp, min(len(self.bins[identifier][min_bin_num]), len(self.bins[identifier][max_bin_num])))
            for img_ind in range(max_pairs):
                self.pairs.append([self.bins[identifier][max_bin_num][img_ind], self.bins[identifier][min_bin_num][img_ind]])
                self.pair_ids.append('_'.join(identifier.split('_')[:-1])) 
        
        self.transform = transforms.Compose([
            transforms.Resize([self.input_size[0], self.input_size[1]]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ])
        
        with open(self.pupil_xyrs_path, 'rb') as pupilFile:
            self.pupil_xyrs = pkl.load(pupilFile)
        
        with open(self.iris_xyrs_path, 'rb') as irisFile:
            self.iris_xyrs = pkl.load(irisFile)

    
    def load_image(self, file):
        return Image.open(file).convert('L')
        
    def __getitem__(self, index):
    
        pair = self.pairs[index]
        
        inp_img_name = pair[0].strip()
        tar_img_name = pair[1].strip()
        
        indiv_id1 = inp_img_name.split('_')[0]
        indiv_id2 = tar_img_name.split('_')[0]
        
        inp_img_path = os.path.join(os.path.join(self.image_dir, indiv_id1), inp_img_name)
        inp_mask_path = os.path.join(os.path.join(self.mask_dir, indiv_id1), inp_img_name)
        
        tar_img_path = os.path.join(os.path.join(self.image_dir, indiv_id2), tar_img_name)
        tar_mask_path = os.path.join(os.path.join(self.mask_dir, indiv_id2), tar_img_name)
        
        inp_img = self.load_image(inp_img_path)
        inp_mask = self.load_image(inp_mask_path)
        
        tar_img = self.load_image(tar_img_path)
        tar_mask = self.load_image(tar_mask_path)
        
        inp_img_pupil_xyr = torch.tensor(self.pupil_xyrs[inp_img_name]) * self.res_mult
        inp_img_iris_xyr = torch.tensor(self.iris_xyrs[inp_img_name]) * self.res_mult
        
        tar_img_pupil_xyr = torch.tensor(self.pupil_xyrs[tar_img_name]) * self.res_mult
        tar_img_iris_xyr = torch.tensor(self.iris_xyrs[tar_img_name]) * self.res_mult
        
        xform_inp_img = self.transform(inp_img)
        xform_inp_mask = self.transform(inp_mask)
        xform_inp_mask[xform_inp_mask<0] = -1
        xform_inp_mask[xform_inp_mask>=0] = 1
        
        xform_tar_img = self.transform(tar_img)
        xform_tar_mask = self.transform(tar_mask)
        xform_tar_mask[xform_tar_mask<0] = -1
        xform_tar_mask[xform_tar_mask>=0] = 1
        
        return {'inp_img': xform_inp_img, 'inp_mask' : xform_inp_mask, 'tar_img' : xform_tar_img, 'tar_mask' : xform_tar_mask, 'inp_img_pxyr': inp_img_pupil_xyr, 'inp_img_ixyr' : inp_img_iris_xyr, 'tar_img_pxyr' : tar_img_pupil_xyr, 'tar_img_ixyr' :  tar_img_iris_xyr, 'identifier':self.pair_ids[index]}
    
    def __len__(self):
        return len(self.pairs)
    
    def reset(self):
        print('Setting Dataset for Min-max pairs...')
        self.pairs = []
        self.pair_ids = []
        for identifier in tqdm(self.bins.keys()):
            min_bin_num = min(list(self.bins[identifier].keys()))
            max_bin_num = max(list(self.bins[identifier].keys()))
            random.shuffle(self.bins[identifier][min_bin_num])
            random.shuffle(self.bins[identifier][max_bin_num])
            if self.max_pairs_inp is None:
                max_pairs = min(len(self.bins[identifier][max_bin_num]), len(self.bins[identifier][min_bin_num]))
            else:
                max_pairs = min(self.max_pairs_inp, min(len(self.bins[identifier][min_bin_num]), len(self.bins[identifier][max_bin_num])))
            for img_ind in range(max_pairs):
                self.pairs.append([self.bins[identifier][max_bin_num][img_ind], self.bins[identifier][min_bin_num][img_ind]])
                self.pair_ids.append('_'.join(identifier.split('_')[:-1]))

class PairExtremeBinDataset(Dataset):
    def __init__(self, bins_path, parent_dir, max_pairs_inp = None, res_mult=1):
        super().__init__()
        
        self.image_dir = os.path.join(parent_dir, 'WarsawData')
        self.mask_dir = os.path.join(parent_dir, 'WarsawMask')
        self.pupil_xyrs_path = os.path.join(parent_dir, 'WarsawPupilCircles.pkl')
        self.iris_xyrs_path = os.path.join(parent_dir, 'WarsawIrisCircles.pkl')
        self.max_pairs_inp = max_pairs_inp
        self.input_size=(int(240*res_mult),int(320*res_mult))
        self.res_mult = res_mult
        
        with open(bins_path, 'rb') as binsFile:
            self.bins = pkl.load(binsFile)
        
        print('Setting Dataset for min-max and max-min pairs...')
        self.pairs = []
        self.pair_ids = []
        for identifier in tqdm(self.bins.keys()):
            min_bin_num = min(list(self.bins[identifier].keys()))
            max_bin_num = max(list(self.bins[identifier].keys()))
            random.shuffle(self.bins[identifier][min_bin_num])
            random.shuffle(self.bins[identifier][max_bin_num])
            if self.max_pairs_inp is None:
                max_pairs = min(len(self.bins[identifier][max_bin_num]), len(self.bins[identifier][min_bin_num]))
            else:
                max_pairs = min(self.max_pairs_inp, min(len(self.bins[identifier][max_bin_num]), len(self.bins[identifier][min_bin_num])))
            for img_ind in range(max_pairs):
                self.pairs.append([self.bins[identifier][max_bin_num][img_ind], self.bins[identifier][min_bin_num][img_ind]])
                self.pair_ids.append('_'.join(identifier.split('_')[:-1]))
        
        for identifier in tqdm(self.bins.keys()):
            min_bin_num = min(list(self.bins[identifier].keys()))
            max_bin_num = max(list(self.bins[identifier].keys()))
            random.shuffle(self.bins[identifier][min_bin_num])
            random.shuffle(self.bins[identifier][max_bin_num])
            if self.max_pairs_inp is None:
                max_pairs = min(len(self.bins[identifier][max_bin_num]), len(self.bins[identifier][min_bin_num]))
            else:
                max_pairs = min(self.max_pairs_inp, min(len(self.bins[identifier][max_bin_num]), len(self.bins[identifier][min_bin_num])))
            for img_ind in range(max_pairs):
                self.pairs.append([self.bins[identifier][min_bin_num][img_ind], self.bins[identifier][max_bin_num][img_ind]])
                self.pair_ids.append('_'.join(identifier.split('_')[:-1]))
        
        self.transform = transforms.Compose([
            transforms.Resize([self.input_size[0], self.input_size[1]]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ])
        
        with open(self.pupil_xyrs_path, 'rb') as pupilFile:
            self.pupil_xyrs = pkl.load(pupilFile)
        
        with open(self.iris_xyrs_path, 'rb') as irisFile:
            self.iris_xyrs = pkl.load(irisFile)

    
    def load_image(self, file):
        return Image.open(file).convert('L')
        
    def __getitem__(self, index):
    
        pair = self.pairs[index]
        
        inp_img_name = pair[0].strip()
        tar_img_name = pair[1].strip()
        
        indiv_id1 = inp_img_name.split('_')[0]
        indiv_id2 = tar_img_name.split('_')[0]
        
        inp_img_path = os.path.join(os.path.join(self.image_dir, indiv_id1), inp_img_name)
        inp_mask_path = os.path.join(os.path.join(self.mask_dir, indiv_id1), inp_img_name)
        
        tar_img_path = os.path.join(os.path.join(self.image_dir, indiv_id2), tar_img_name)
        tar_mask_path = os.path.join(os.path.join(self.mask_dir, indiv_id2), tar_img_name)
        
        inp_img = self.load_image(inp_img_path)
        inp_mask = self.load_image(inp_mask_path)
        
        tar_img = self.load_image(tar_img_path)
        tar_mask = self.load_image(tar_mask_path)
        
        inp_img_pupil_xyr = torch.tensor(self.pupil_xyrs[inp_img_name]) * self.res_mult
        inp_img_iris_xyr = torch.tensor(self.iris_xyrs[inp_img_name]) * self.res_mult
        
        tar_img_pupil_xyr = torch.tensor(self.pupil_xyrs[tar_img_name]) * self.res_mult
        tar_img_iris_xyr = torch.tensor(self.iris_xyrs[tar_img_name]) * self.res_mult
        
        xform_inp_img = self.transform(inp_img)
        xform_inp_mask = self.transform(inp_mask)
        xform_inp_mask[xform_inp_mask<0] = -1
        xform_inp_mask[xform_inp_mask>=0] = 1
        
        xform_tar_img = self.transform(tar_img)
        xform_tar_mask = self.transform(tar_mask)
        xform_tar_mask[xform_tar_mask<0] = -1
        xform_tar_mask[xform_tar_mask>=0] = 1
        
        return {'inp_img': xform_inp_img, 'inp_mask' : xform_inp_mask, 'tar_img' : xform_tar_img, 'tar_mask' : xform_tar_mask, 'inp_img_pxyr': inp_img_pupil_xyr, 'inp_img_ixyr' : inp_img_iris_xyr, 'tar_img_pxyr' : tar_img_pupil_xyr, 'tar_img_ixyr' :  tar_img_iris_xyr, 'identifier':self.pair_ids[index] }
    
    def __len__(self):
        return len(self.pairs)
    
    def reset(self):
        print('Setting Dataset for Min-max pairs...')
        self.pairs = []
        self.pair_ids = []
        for identifier in tqdm(self.bins.keys()):
            min_bin_num = min(list(self.bins[identifier].keys()))
            max_bin_num = max(list(self.bins[identifier].keys()))
            random.shuffle(self.bins[identifier][min_bin_num])
            random.shuffle(self.bins[identifier][max_bin_num])
            if self.max_pairs_inp is None:
                max_pairs = min(len(self.bins[identifier][max_bin_num]), len(self.bins[identifier][min_bin_num]))
            else:
                max_pairs = min(self.max_pairs_inp, min(len(self.bins[identifier][max_bin_num]), len(self.bins[identifier][min_bin_num])))
            for img_ind in range(max_pairs):
                self.pairs.append([self.bins[identifier][max_bin_num][img_ind], self.bins[identifier][min_bin_num][img_ind]])
                self.pair_ids.append('_'.join(identifier.split('_')[:-1]))
        
        for identifier in tqdm(self.bins.keys()):
            min_bin_num = min(list(self.bins[identifier].keys()))
            max_bin_num = max(list(self.bins[identifier].keys()))
            random.shuffle(self.bins[identifier][min_bin_num])
            random.shuffle(self.bins[identifier][max_bin_num])
            if self.max_pairs_inp is None:
                max_pairs = min(len(self.bins[identifier][max_bin_num]), len(self.bins[identifier][min_bin_num]))
            else:
                max_pairs = min(self.max_pairs_inp, min(len(self.bins[identifier][max_bin_num]), len(self.bins[identifier][min_bin_num])))
            for img_ind in range(max_pairs):
                self.pairs.append([self.bins[identifier][min_bin_num][img_ind], self.bins[identifier][max_bin_num][img_ind]])
                self.pair_ids.append('_'.join(identifier.split('_')[:-1]))
        

class PairMinMaxBinDatasetPolar(Dataset):
    def __init__(self, bins_path, parent_dir, input_size=(240,320), polar_width=512, polar_height=64):
        super().__init__()
        
        self.image_dir = os.path.join(parent_dir, 'WarsawData')
        self.mask_dir = os.path.join(parent_dir, 'WarsawMask')
        self.pupil_xyrs_path = os.path.join(parent_dir, 'WarsawPupilCircles.pkl')
        self.iris_xyrs_path = os.path.join(parent_dir, 'WarsawIrisCircles.pkl')
        self.max_pairs_inp = max_pairs_inp
        self.input_size = input_size
        self.polar_width = polar_width
        self.polar_height = polar_height
        
        with open(bins_path, 'rb') as binsFile:
            self.bins = pkl.load(binsFile)
        
        print('Setting Dataset for Min-max pairs...')
        self.pairs = []
        for identifier in tqdm(self.bins.keys()):
            min_bin_num = min(list(self.bins[identifier].keys()))
            max_bin_num = max(list(self.bins[identifier].keys()))
            random.shuffle(self.bins[identifier][min_bin_num])
            random.shuffle(self.bins[identifier][max_bin_num])
            max_pairs = min(len(self.bins[identifier][min_bin_num]), len(self.bins[identifier][max_bin_num]))
            for img_ind in range(max_pairs):
                self.pairs.append([self.bins[identifier][min_bin_num][img_ind], self.bins[identifier][max_bin_num][img_ind]])
         
        random.shuffle(self.pairs)  
        
        self.transform = transforms.Compose([
            transforms.Resize([input_size[0], input_size[1]]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ])
        
        with open(self.pupil_xyrs_path, 'rb') as pupilFile:
            self.pupil_xyrs = pkl.load(pupilFile)
        
        with open(self.iris_xyrs_path, 'rb') as irisFile:
            self.iris_xyrs = pkl.load(irisFile)
            
    def grid_sample(self, input, grid, interp_mode='bilinear'):

        # grid: [-1, 1]
        N, C, H, W = input.shape
        gridx = grid[:, :, :, 0]
        gridy = grid[:, :, :, 1]
        gridx = ((gridx + 1) / 2 * W - 0.5) / (W - 1) * 2 - 1
        gridy = ((gridy + 1) / 2 * H - 0.5) / (H - 1) * 2 - 1
        newgrid = torch.stack([gridx, gridy], dim=-1)
        return torch.nn.functional.grid_sample(input, newgrid, mode=interp_mode)
        
    # Rubbersheet model-based Cartesian-to-polar transformation using bilinear interpolation from torch grid sample
    def cartToPol(self, image, mask, pupil_xyr, iris_xyr):
        with torch.no_grad():
            if pupil_xyr is None or iris_xyr is None:
                return None, None
            
            image = image.unsqueeze(0)
            mask = mask.unsqueeze(0)
            width = image.shape[3]
            height = image.shape[2]

            polar_height = self.polar_height
            polar_width = self.polar_width

            pupil_xyr = torch.tensor(pupil_xyr).unsqueeze(0).float()
            iris_xyr = torch.tensor(iris_xyr).unsqueeze(0).float()
            
            theta = (2*pi*torch.linspace(1,polar_width,polar_width)/polar_width)
            pxCirclePoints = pupil_xyr[:, 0].reshape(-1, 1) + pupil_xyr[:, 2].reshape(-1, 1) @ torch.cos(theta).reshape(1, polar_width) #b x 512
            pyCirclePoints = pupil_xyr[:, 1].reshape(-1, 1) + pupil_xyr[:, 2].reshape(-1, 1) @ torch.sin(theta).reshape(1, polar_width)  #b x 512
            
            ixCirclePoints = iris_xyr[:, 0].reshape(-1, 1) + iris_xyr[:, 2].reshape(-1, 1) @ torch.cos(theta).reshape(1, polar_width)  #b x 512
            iyCirclePoints = iris_xyr[:, 1].reshape(-1, 1) + iris_xyr[:, 2].reshape(-1, 1) @ torch.sin(theta).reshape(1, polar_width)  #b x 512

            radius = (torch.linspace(0,polar_height,polar_height)/polar_height).reshape(-1, 1)  #64 x 1
            
            pxCoords = torch.matmul((1-radius), pxCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
            pyCoords = torch.matmul((1-radius), pyCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
            
            ixCoords = torch.matmul(radius, ixCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
            iyCoords = torch.matmul(radius, iyCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512

            x = torch.clamp(pxCoords + ixCoords, 0, width-1).float()
            x_norm = (x/(width-1))*2 - 1 #b x 64 x 512

            y = torch.clamp(pyCoords + iyCoords, 0, height-1).float()
            y_norm = (y/(height-1))*2 - 1  #b x 64 x 512

            grid_sample_mat = torch.cat([x_norm.unsqueeze(-1), y_norm.unsqueeze(-1)], dim=-1)

            image_polar = self.grid_sample(image, grid_sample_mat, interp_mode='bilinear')
            mask_polar = self.grid_sample(mask, grid_sample_mat, interp_mode='nearest')

            return image_polar[0], mask_polar[0]
    
    def load_image(self, file):
        return Image.open(file).convert('L')
        
    def __getitem__(self, index):
    
        pair = self.pairs[index]
        
        inp_img_name = pair[0].strip()
        tar_img_name = pair[1].strip()
        
        indiv_id1 = inp_img_name.split('_')[0]
        indiv_id2 = tar_img_name.split('_')[0]
        
        inp_img_path = os.path.join(os.path.join(self.image_dir, indiv_id1), inp_img_name)
        inp_mask_path = os.path.join(os.path.join(self.mask_dir, indiv_id1), inp_img_name)
        
        tar_img_path = os.path.join(os.path.join(self.image_dir, indiv_id2), tar_img_name)
        tar_mask_path = os.path.join(os.path.join(self.mask_dir, indiv_id2), tar_img_name)
        
        inp_img = self.load_image(inp_img_path)
        inp_mask = self.load_image(inp_mask_path)
        
        tar_img = self.load_image(tar_img_path)
        tar_mask = self.load_image(tar_mask_path)
        
        inp_img_pupil_xyr = torch.tensor(self.pupil_xyrs[inp_img_name])
        inp_img_iris_xyr = torch.tensor(self.iris_xyrs[inp_img_name])
        
        tar_img_pupil_xyr = torch.tensor(self.pupil_xyrs[tar_img_name])
        tar_img_iris_xyr = torch.tensor(self.iris_xyrs[tar_img_name])
        
        xform_inp_img = self.transform(inp_img)
        xform_inp_mask = self.transform(inp_mask)
        xform_inp_mask[xform_inp_mask<0] = -1
        xform_inp_mask[xform_inp_mask>=0] = 1
        
        xform_inp_img_polar, xform_inp_mask_polar = self.cartToPol(xform_inp_img, xform_inp_mask, inp_img_pupil_xyr, inp_img_iris_xyr)
        
        xform_tar_img = self.transform(tar_img)
        xform_tar_mask = self.transform(tar_mask)
        xform_tar_mask[xform_tar_mask<0] = -1
        xform_tar_mask[xform_tar_mask>=0] = 1
        
        xform_tar_img_polar, xform_tar_mask_polar = self.cartToPol(xform_tar_img, xform_tar_mask, tar_img_pupil_xyr, tar_img_iris_xyr)
        
        return {'inp_img': xform_inp_img_polar, 'inp_mask' : xform_inp_mask_polar, 'tar_img' : xform_tar_img_polar, 'tar_mask' : xform_tar_mask_polar, 'inp_img_pxyr': inp_img_pupil_xyr, 'inp_img_ixyr' : inp_img_iris_xyr, 'tar_img_pxyr' : tar_img_pupil_xyr, 'tar_img_ixyr' :  tar_img_iris_xyr }
    
    def __len__(self):
        return len(self.pairs)
    
    def reset(self):
        print('Setting Dataset for Min-max pairs...')
        self.pairs = []
        for identifier in tqdm(self.bins.keys()):
            min_bin_num = min(list(self.bins[identifier].keys()))
            max_bin_num = max(list(self.bins[identifier].keys()))
            random.shuffle(self.bins[identifier][min_bin_num])
            random.shuffle(self.bins[identifier][max_bin_num])
            if self.max_pairs_inp is None:
                max_pairs = min(len(self.bins[identifier][min_bin_num]), len(self.bins[identifier][max_bin_num]))
            else:
                max_pairs = self.max_pairs_inp
            for img_ind in range(max_pairs):
                self.pairs.append([self.bins[identifier][min_bin_num][img_ind], self.bins[identifier][max_bin_num][img_ind]])     

class PairFromBinDatasetBoth(Dataset):
    def __init__(self, bins_path, parent_dir, flip_data = True, res_mult=1):
        super().__init__()
        
        self.image_dir = os.path.join(parent_dir, 'WarsawData')
        self.mask_dir = os.path.join(parent_dir, 'WarsawMask')
        self.pupil_xyrs_path = os.path.join(parent_dir, 'WarsawPupilCircles.pkl')
        self.iris_xyrs_path = os.path.join(parent_dir, 'WarsawIrisCircles.pkl')
        self.flip_data = flip_data   
        self.input_size=(int(240*res_mult),int(320*res_mult))
        self.res_mult = res_mult
        
        with open(bins_path, 'rb') as binsFile:
            self.bins = pkl.load(binsFile)
        
        print('Initializing pairs...')
        self.pairs = []
        self.pair_ids = []
        for identifier in tqdm(self.bins.keys()):
            for bin_num_1 in self.bins[identifier].keys():
                for bin_num_2 in self.bins[identifier].keys():
                    if bin_num_1 < bin_num_2:
                        random.shuffle(self.bins[identifier][bin_num_1])
                        random.shuffle(self.bins[identifier][bin_num_2])
                        max_pairs = min(len(self.bins[identifier][bin_num_1]), len(self.bins[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            self.pairs.append([self.bins[identifier][bin_num_1][img_ind], self.bins[identifier][bin_num_2][img_ind]])
                            self.pair_ids.append('_'.join(identifier.split('_')[:-1]))
        
        for identifier in tqdm(self.bins.keys()):
            for bin_num_1 in self.bins[identifier].keys():
                for bin_num_2 in self.bins[identifier].keys():
                    if bin_num_1 > bin_num_2:
                        random.shuffle(self.bins[identifier][bin_num_1])
                        random.shuffle(self.bins[identifier][bin_num_2])
                        max_pairs = min(len(self.bins[identifier][bin_num_1]), len(self.bins[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            self.pairs.append([self.bins[identifier][bin_num_1][img_ind], self.bins[identifier][bin_num_2][img_ind]])
                            self.pair_ids.append('_'.join(identifier.split('_')[:-1]))                
        
        self.transform = transforms.Compose([
            transforms.Resize([self.input_size[0], self.input_size[1]]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ])
        
        with open(self.pupil_xyrs_path, 'rb') as pupilFile:
            self.pupil_xyrs = pkl.load(pupilFile)
        
        with open(self.iris_xyrs_path, 'rb') as irisFile:
            self.iris_xyrs = pkl.load(irisFile)
    
    def load_image(self, file):
        return Image.open(file).convert('L')
        
    def __getitem__(self, index):
    
        pair = self.pairs[index]
        
        inp_img_name = pair[0].strip()
        tar_img_name = pair[1].strip()
        
        indiv_id1 = inp_img_name.split('_')[0]
        indiv_id2 = tar_img_name.split('_')[0]
        
        inp_img_path = os.path.join(os.path.join(self.image_dir, indiv_id1), inp_img_name)
        inp_mask_path = os.path.join(os.path.join(self.mask_dir, indiv_id1), inp_img_name)
        
        tar_img_path = os.path.join(os.path.join(self.image_dir, indiv_id2), tar_img_name)
        tar_mask_path = os.path.join(os.path.join(self.mask_dir, indiv_id2), tar_img_name)
        
        inp_img = self.load_image(inp_img_path)
        inp_mask = self.load_image(inp_mask_path)
        
        tar_img = self.load_image(tar_img_path)
        tar_mask = self.load_image(tar_mask_path)
        
        inp_img_pupil_xyr = torch.tensor(self.pupil_xyrs[inp_img_name]) * self.res_mult
        inp_img_iris_xyr = torch.tensor(self.iris_xyrs[inp_img_name]) * self.res_mult
        
        tar_img_pupil_xyr = torch.tensor(self.pupil_xyrs[tar_img_name]) * self.res_mult
        tar_img_iris_xyr = torch.tensor(self.iris_xyrs[tar_img_name]) * self.res_mult
        
        if self.flip_data and random.random() < 0.5:
            inp_img = inp_img.transpose(Image.FLIP_LEFT_RIGHT) 
            inp_mask = inp_mask.transpose(Image.FLIP_LEFT_RIGHT)
            tar_img = tar_img.transpose(Image.FLIP_LEFT_RIGHT) 
            tar_mask = tar_mask.transpose(Image.FLIP_LEFT_RIGHT)
            inp_img_pupil_xyr[0] = self.input_size[1] - inp_img_pupil_xyr[0]
            tar_img_pupil_xyr[0] = self.input_size[1] - tar_img_pupil_xyr[0]
            inp_img_iris_xyr[0] = self.input_size[1] - inp_img_iris_xyr[0]
            tar_img_iris_xyr[0] = self.input_size[1] - tar_img_iris_xyr[0]
        
        xform_inp_img = self.transform(inp_img)
        xform_inp_mask = self.transform(inp_mask)
        xform_inp_mask[xform_inp_mask<0] = -1
        xform_inp_mask[xform_inp_mask>=0] = 1
        
        xform_tar_img = self.transform(tar_img)
        xform_tar_mask = self.transform(tar_mask)
        xform_tar_mask[xform_tar_mask<0] = -1
        xform_tar_mask[xform_tar_mask>=0] = 1
        
        
        return {'inp_img': xform_inp_img, 'inp_mask' : xform_inp_mask, 'tar_img' : xform_tar_img, 'tar_mask' : xform_tar_mask, 'inp_img_pxyr': inp_img_pupil_xyr, 'inp_img_ixyr' : inp_img_iris_xyr, 'tar_img_pxyr' : tar_img_pupil_xyr, 'tar_img_ixyr' :  tar_img_iris_xyr, 'identifier':self.pair_ids[index]}
    
    def __len__(self):
        return len(self.pairs)
    
    def reset(self):
        print('Resetting Pairs....')
        self.pairs = []
        self.pair_ids = []
        for identifier in tqdm(self.bins.keys()):
            for bin_num_1 in self.bins[identifier].keys():
                for bin_num_2 in self.bins[identifier].keys():
                    if bin_num_1 < bin_num_2:
                        random.shuffle(self.bins[identifier][bin_num_1])
                        random.shuffle(self.bins[identifier][bin_num_2])
                        max_pairs = min(len(self.bins[identifier][bin_num_1]), len(self.bins[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            self.pairs.append([self.bins[identifier][bin_num_1][img_ind], self.bins[identifier][bin_num_2][img_ind]])
                            self.pair_ids.append('_'.join(identifier.split('_')[:-1]))
        
        for identifier in tqdm(self.bins.keys()):
            for bin_num_1 in self.bins[identifier].keys():
                for bin_num_2 in self.bins[identifier].keys():
                    if bin_num_1 > bin_num_2:
                        random.shuffle(self.bins[identifier][bin_num_1])
                        random.shuffle(self.bins[identifier][bin_num_2])
                        max_pairs = min(len(self.bins[identifier][bin_num_1]), len(self.bins[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            self.pairs.append([self.bins[identifier][bin_num_1][img_ind], self.bins[identifier][bin_num_2][img_ind]])
                            self.pair_ids.append('_'.join(identifier.split('_')[:-1]))    
                            
class PairFromBinAllDatasetSB(Dataset):
    def __init__(self, bins_path_wsd, parent_dir_wsd, bins_path_bxgrid, parent_dir_bxgrid, normalize=True, flip_data = True, res_mult=1):
        super().__init__()
        
        self.flip_data = flip_data   
        self.input_size=(int(240*res_mult),int(320*res_mult))
        self.res_mult = res_mult
        
        self.image_dir_wsd = os.path.join(parent_dir_wsd, 'WarsawData2')
        self.mask_dir_wsd = os.path.join(parent_dir_wsd, 'WarsawMask2')            
        
        with open(bins_path_wsd, 'rb') as wsdbinsFile:
            self.bins_wsd = pkl.load(wsdbinsFile)
        
        with open(bins_path_bxgrid, 'rb') as bxgridbinsFile:
            self.bins_bxgrid = pkl.load(bxgridbinsFile)
        
        self.all_ids = list(self.bins_wsd.keys()) + list(self.bins_bxgrid.keys())
        
        print('Initializing pairs for Warsaw Pupil Dynamics...')
        self.pairs = []
        self.pair_ids = []
        
        for identifier in tqdm(self.bins_wsd.keys()):
            for bin_num_1 in self.bins_wsd[identifier].keys():
                for bin_num_2 in self.bins_wsd[identifier].keys():
                    if bin_num_1 < bin_num_2:
                        random.shuffle(self.bins_wsd[identifier][bin_num_1])
                        random.shuffle(self.bins_wsd[identifier][bin_num_2])
                        max_pairs = min(len(self.bins_wsd[identifier][bin_num_1]), len(self.bins_wsd[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            imagename1 = self.bins_wsd[identifier][bin_num_1][img_ind]
                            imagename2 = self.bins_wsd[identifier][bin_num_2][img_ind]
                            indiv_id1 = imagename1.split('_')[0]
                            indiv_id2 = imagename2.split('_')[0]
                            imagepath1 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id1), imagename1)
                            maskpath1 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id1), imagename1)
                            imagepath2 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id2), imagename2)
                            maskpath2 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id2), imagename2)
                            self.pairs.append([imagepath1, maskpath1, imagepath2, maskpath2])  
                            self.pair_ids.append(identifier)
        
        
        self.image_dir_bxgrid = os.path.join(parent_dir_bxgrid, 'iris')
        self.mask_dir_bxgrid = os.path.join(parent_dir_bxgrid, 'mask')            
        
        print('Initializing pairs for BXGRID...')
        for identifier in tqdm(self.bins_bxgrid.keys()):
            for bin_num_1 in self.bins_bxgrid[identifier].keys():
                for bin_num_2 in self.bins_bxgrid[identifier].keys():
                    if bin_num_1 < bin_num_2:
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_1])
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_2])
                        max_pairs = min(len(self.bins_bxgrid[identifier][bin_num_1]), len(self.bins_bxgrid[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            imagename1 = self.bins_bxgrid[identifier][bin_num_1][img_ind]
                            imagename2 = self.bins_bxgrid[identifier][bin_num_2][img_ind]
                            imagepath1 = os.path.join(self.image_dir_bxgrid, imagename1)
                            maskpath1 = os.path.join(self.mask_dir_bxgrid, imagename1)
                            imagepath2 = os.path.join(self.image_dir_bxgrid, imagename2)
                            maskpath2 = os.path.join(self.mask_dir_bxgrid, imagename2)
                            self.pairs.append([imagepath1, maskpath1, imagepath2, maskpath2])  
                            self.pair_ids.append(identifier)    
        
        print('Total pairs: ', len(self.pairs))
        
        if normalize:
            self.transform = transforms.Compose([
                transforms.Resize([self.input_size[0], self.input_size[1]]),
                transforms.ToTensor(),
                transforms.Normalize((0.5, ), (0.5, ))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize([self.input_size[0], self.input_size[1]]),
                transforms.ToTensor()
            ])
        
        
        
        self.pupil_iris_xyrs_path_wsd = os.path.join(parent_dir_wsd, 'pupil_iris_xyrs2.pkl')
        with open(self.pupil_iris_xyrs_path_wsd, 'rb') as pupilirisxyrsFile:
            self.pupil_iris_xyrs_wsd = pkl.load(pupilirisxyrsFile)
        
        self.pupil_iris_xyrs_path_bxgrid = os.path.join(parent_dir_bxgrid, 'pupil_iris_xyrs.pkl')
        with open(self.pupil_iris_xyrs_path_bxgrid, 'rb') as pupilirisxyrsFile:
            self.pupil_iris_xyrs_bxgrid = pkl.load(pupilirisxyrsFile)
        
        self.pupil_xyrs = {}
        self.iris_xyrs = {}
        
        for imagename in self.pupil_iris_xyrs_wsd.keys():
            indiv_id = imagename.split('_')[0]
            imagepath = os.path.join(os.path.join(self.image_dir_wsd, indiv_id), imagename)
            self.pupil_xyrs[imagepath] = self.pupil_iris_xyrs_wsd[imagename]['pxyr']
            self.iris_xyrs[imagepath] = self.pupil_iris_xyrs_wsd[imagename]['ixyr']
        
        for imagename in self.pupil_iris_xyrs_bxgrid.keys():
            imagepath = os.path.join(self.image_dir_bxgrid, imagename)
            self.pupil_xyrs[imagepath] = self.pupil_iris_xyrs_bxgrid[imagename]['pxyr']
            self.iris_xyrs[imagepath] = self.pupil_iris_xyrs_bxgrid[imagename]['ixyr']
        
       
    
    def load_image(self, file):
        return Image.open(file).convert('L')
        
    def __getitem__(self, index):
    
        pair = self.pairs[index]
        
        inp_img_path = pair[0].strip()
        inp_mask_path = pair[1].strip()
        tar_img_path = pair[2].strip()
        tar_mask_path = pair[3].strip()
        
        inp_img = self.load_image(inp_img_path)
        inp_mask = self.load_image(inp_mask_path)
        
        tar_img = self.load_image(tar_img_path)
        tar_mask = self.load_image(tar_mask_path)
        
        inp_img_pupil_xyr = torch.FloatTensor(self.pupil_xyrs[inp_img_path])
        inp_img_iris_xyr = torch.FloatTensor(self.iris_xyrs[inp_img_path])
        
        tar_img_pupil_xyr = torch.FloatTensor(self.pupil_xyrs[tar_img_path])
        tar_img_iris_xyr = torch.FloatTensor(self.iris_xyrs[tar_img_path])
        
        sw, sh = inp_img.size
        sw_mult = (self.input_size[1]/sw)
        sh_mult = (self.input_size[0]/sh)
        
        bw, bh = tar_img.size
        bw_mult = (self.input_size[1]/bw)
        bh_mult = (self.input_size[0]/bh)
        
        inp_img_pupil_xyr[0] = (inp_img_pupil_xyr[0] * sw_mult)
        inp_img_pupil_xyr[1] = (inp_img_pupil_xyr[1] * sh_mult)
        inp_img_pupil_xyr[2] = (inp_img_pupil_xyr[2] * max(sw_mult, sh_mult))
        
        tar_img_pupil_xyr[0] = (tar_img_pupil_xyr[0] * bw_mult)
        tar_img_pupil_xyr[1] = (tar_img_pupil_xyr[1] * bh_mult)
        tar_img_pupil_xyr[2] = (tar_img_pupil_xyr[2] * max(bw_mult, bh_mult))
    
        inp_img_iris_xyr[0] = (inp_img_iris_xyr[0] * sw_mult)
        inp_img_iris_xyr[1] = (inp_img_iris_xyr[1] * sh_mult)
        inp_img_iris_xyr[2] = (inp_img_iris_xyr[2] * max(sw_mult, sh_mult))
        
        tar_img_iris_xyr[0] = (tar_img_iris_xyr[0] * bw_mult)
        tar_img_iris_xyr[1] = (tar_img_iris_xyr[1] * bh_mult)
        tar_img_iris_xyr[2] = (tar_img_iris_xyr[2] * max(bw_mult, bh_mult))
        
        if self.flip_data and random.random() < 0.5:
            inp_img = inp_img.transpose(Image.FLIP_LEFT_RIGHT) 
            inp_mask = inp_mask.transpose(Image.FLIP_LEFT_RIGHT)
            tar_img = tar_img.transpose(Image.FLIP_LEFT_RIGHT) 
            tar_mask = tar_mask.transpose(Image.FLIP_LEFT_RIGHT)
            inp_img_pupil_xyr[0] = self.input_size[1] - inp_img_pupil_xyr[0]
            tar_img_pupil_xyr[0] = self.input_size[1] - tar_img_pupil_xyr[0]
            inp_img_iris_xyr[0] = self.input_size[1] - inp_img_iris_xyr[0]
            tar_img_iris_xyr[0] = self.input_size[1] - tar_img_iris_xyr[0]

        xform_inp_img = self.transform(inp_img)
        xform_inp_mask = self.transform(inp_mask)
        xform_inp_mask[xform_inp_mask<0] = -1
        xform_inp_mask[xform_inp_mask>=0] = 1
        
        xform_tar_img = self.transform(tar_img)
        xform_tar_mask = self.transform(tar_mask)
        xform_tar_mask[xform_tar_mask<0] = -1
        xform_tar_mask[xform_tar_mask>=0] = 1
        
        return {'inp_img': xform_inp_img, 'tar_img' : xform_tar_img, 'inp_mask': xform_inp_mask, 'tar_mask' : xform_tar_mask, 'inp_img_pxyr' : inp_img_pupil_xyr, 'tar_img_pxyr' : tar_img_pupil_xyr, 'inp_img_ixyr': inp_img_iris_xyr, 'tar_img_ixyr' :  tar_img_iris_xyr, 'identifier':self.pair_ids[index] }
    
    def __len__(self):
        return len(self.pairs)
    
    def reset(self):
        print('Shuffling pairs for Warsaw Pupil Dynamics...')
        self.pairs = []
        self.pair_ids = []
        for identifier in tqdm(self.bins_wsd.keys()):
            for bin_num_1 in self.bins_wsd[identifier].keys():
                for bin_num_2 in self.bins_wsd[identifier].keys():
                    if bin_num_1 < bin_num_2:
                        random.shuffle(self.bins_wsd[identifier][bin_num_1])
                        random.shuffle(self.bins_wsd[identifier][bin_num_2])
                        max_pairs = min(len(self.bins_wsd[identifier][bin_num_1]), len(self.bins_wsd[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            imagename1 = self.bins_wsd[identifier][bin_num_1][img_ind]
                            imagename2 = self.bins_wsd[identifier][bin_num_2][img_ind]
                            indiv_id1 = imagename1.split('_')[0]
                            indiv_id2 = imagename2.split('_')[0]
                            imagepath1 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id1), imagename1)
                            maskpath1 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id1), imagename1)
                            imagepath2 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id2), imagename2)
                            maskpath2 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id2), imagename2)
                            self.pairs.append([imagepath1, maskpath1, imagepath2, maskpath2])  
                            self.pair_ids.append(identifier)                        
        
        print('Shuffling pairs for BXGRID...')
        for identifier in tqdm(self.bins_bxgrid.keys()):
            for bin_num_1 in self.bins_bxgrid[identifier].keys():
                for bin_num_2 in self.bins_bxgrid[identifier].keys():
                    if bin_num_1 < bin_num_2:
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_1])
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_2])
                        max_pairs = min(len(self.bins_bxgrid[identifier][bin_num_1]), len(self.bins_bxgrid[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            imagename1 = self.bins_bxgrid[identifier][bin_num_1][img_ind]
                            imagename2 = self.bins_bxgrid[identifier][bin_num_2][img_ind]
                            imagepath1 = os.path.join(self.image_dir_bxgrid, imagename1)
                            maskpath1 = os.path.join(self.mask_dir_bxgrid, imagename1)
                            imagepath2 = os.path.join(self.image_dir_bxgrid, imagename2)
                            maskpath2 = os.path.join(self.mask_dir_bxgrid, imagename2)
                            self.pairs.append([imagepath1, maskpath1, imagepath2, maskpath2])  
                            self.pair_ids.append(identifier)
                            
        print('Total pairs: ', len(self.pairs))
        
class PairFromBinAllDatasetMinMax(Dataset):
    def __init__(self, bins_path_wsd, parent_dir_wsd, bins_path_bxgrid, parent_dir_bxgrid, flip_data = True, res_mult=1):
        super().__init__()
        
        self.flip_data = flip_data   
        self.input_size=(int(240*res_mult),int(320*res_mult))
        self.res_mult = res_mult
        
        self.image_dir_wsd = os.path.join(parent_dir_wsd, 'WarsawData2')
        self.mask_dir_wsd = os.path.join(parent_dir_wsd, 'WarsawMask2')            
        
        with open(bins_path_wsd, 'rb') as wsdbinsFile:
            self.bins_wsd = pkl.load(wsdbinsFile)
        
        with open(bins_path_bxgrid, 'rb') as bxgridbinsFile:
            self.bins_bxgrid = pkl.load(bxgridbinsFile)
        
        self.all_ids = list(self.bins_wsd.keys()) + list(self.bins_bxgrid.keys())
        
        print('Initializing pairs for Warsaw Pupil Dynamics...')
        self.pairs = []
        
        for identifier in tqdm(self.bins_wsd.keys()):
            bin_num_1 = min(list(self.bins_wsd[identifier].keys()))
            bin_num_2 = max(list(self.bins_wsd[identifier].keys()))
            random.shuffle(self.bins_wsd[identifier][bin_num_1])
            random.shuffle(self.bins_wsd[identifier][bin_num_2])
            max_pairs = min(len(self.bins_wsd[identifier][bin_num_1]), len(self.bins_wsd[identifier][bin_num_2]))
            for img_ind in range(max_pairs):
                imagename1 = self.bins_wsd[identifier][bin_num_1][img_ind]
                imagename2 = self.bins_wsd[identifier][bin_num_2][img_ind]
                indiv_id1 = imagename1.split('_')[0]
                indiv_id2 = imagename2.split('_')[0]
                imagepath1 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id1), imagename1)
                maskpath1 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id1), imagename1)
                imagepath2 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id2), imagename2)
                maskpath2 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id2), imagename2)
                self.pairs.append([imagepath1, maskpath1, imagepath2, maskpath2, identifier])  
        
        
        self.image_dir_bxgrid = os.path.join(parent_dir_bxgrid, 'iris')
        self.mask_dir_bxgrid = os.path.join(parent_dir_bxgrid, 'mask')            
        
        print('Initializing pairs for BXGRID...')
        for identifier in tqdm(self.bins_bxgrid.keys()):
            bin_num_1 = min(list(self.bins_bxgrid[identifier].keys()))
            bin_num_2 = max(list(self.bins_bxgrid[identifier].keys()))
            random.shuffle(self.bins_bxgrid[identifier][bin_num_1])
            random.shuffle(self.bins_bxgrid[identifier][bin_num_2])
            max_pairs = min(len(self.bins_bxgrid[identifier][bin_num_1]), len(self.bins_bxgrid[identifier][bin_num_2]))
            for img_ind in range(max_pairs):
                imagename1 = self.bins_bxgrid[identifier][bin_num_1][img_ind]
                imagename2 = self.bins_bxgrid[identifier][bin_num_2][img_ind]
                imagepath1 = os.path.join(self.image_dir_bxgrid, imagename1)
                maskpath1 = os.path.join(self.mask_dir_bxgrid, imagename1)
                imagepath2 = os.path.join(self.image_dir_bxgrid, imagename2)
                maskpath2 = os.path.join(self.mask_dir_bxgrid, imagename2)
                self.pairs.append([imagepath1, maskpath1, imagepath2, maskpath2, identifier])    
        
        print('Total pairs: ', len(self.pairs))
        
        self.transform = transforms.Compose([
            transforms.Resize([self.input_size[0], self.input_size[1]]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ])
        
        self.pupil_iris_xyrs_path_wsd = os.path.join(parent_dir_wsd, 'pupil_iris_xyrs2.pkl')
        with open(self.pupil_iris_xyrs_path_wsd, 'rb') as pupilirisxyrsFile:
            self.pupil_iris_xyrs_wsd = pkl.load(pupilirisxyrsFile)
        
        self.pupil_iris_xyrs_path_bxgrid = os.path.join(parent_dir_bxgrid, 'pupil_iris_xyrs.pkl')
        with open(self.pupil_iris_xyrs_path_bxgrid, 'rb') as pupilirisxyrsFile:
            self.pupil_iris_xyrs_bxgrid = pkl.load(pupilirisxyrsFile)
        
        self.pupil_xyrs = {}
        self.iris_xyrs = {}
        
        for imagename in self.pupil_iris_xyrs_wsd.keys():
            indiv_id = imagename.split('_')[0]
            imagepath = os.path.join(os.path.join(self.image_dir_wsd, indiv_id), imagename)
            self.pupil_xyrs[imagepath] = self.pupil_iris_xyrs_wsd[imagename]['pxyr']
            self.iris_xyrs[imagepath] = self.pupil_iris_xyrs_wsd[imagename]['ixyr']
        
        for imagename in self.pupil_iris_xyrs_bxgrid.keys():
            imagepath = os.path.join(self.image_dir_bxgrid, imagename)
            self.pupil_xyrs[imagepath] = self.pupil_iris_xyrs_bxgrid[imagename]['pxyr']
            self.iris_xyrs[imagepath] = self.pupil_iris_xyrs_bxgrid[imagename]['ixyr']
        
       
    
    def load_image(self, file):
        return Image.open(file).convert('L')
        
    def __getitem__(self, index):
    
        pair = self.pairs[index]
        
        inp_img_path = pair[0].strip()
        inp_mask_path = pair[1].strip()
        tar_img_path = pair[2].strip()
        tar_mask_path = pair[3].strip()
        identifier = pair[4].strip()
        
        inp_img = self.load_image(inp_img_path)
        inp_mask = self.load_image(inp_mask_path)
        
        tar_img = self.load_image(tar_img_path)
        tar_mask = self.load_image(tar_mask_path)
        
        inp_img_pupil_xyr = torch.FloatTensor(self.pupil_xyrs[inp_img_path])
        inp_img_iris_xyr = torch.FloatTensor(self.iris_xyrs[inp_img_path])
        
        tar_img_pupil_xyr = torch.FloatTensor(self.pupil_xyrs[tar_img_path])
        tar_img_iris_xyr = torch.FloatTensor(self.iris_xyrs[tar_img_path])
        
        sw, sh = inp_img.size
        sw_mult = (self.input_size[1]/sw)
        sh_mult = (self.input_size[0]/sh)
        
        bw, bh = tar_img.size
        bw_mult = (self.input_size[1]/bw)
        bh_mult = (self.input_size[0]/bh)
        
        inp_img_pupil_xyr[0] = (inp_img_pupil_xyr[0] * sw_mult)
        inp_img_pupil_xyr[1] = (inp_img_pupil_xyr[1] * sh_mult)
        inp_img_pupil_xyr[2] = (inp_img_pupil_xyr[2] * max(sw_mult, sh_mult))
        
        tar_img_pupil_xyr[0] = (tar_img_pupil_xyr[0] * bw_mult)
        tar_img_pupil_xyr[1] = (tar_img_pupil_xyr[1] * bh_mult)
        tar_img_pupil_xyr[2] = (tar_img_pupil_xyr[2] * max(bw_mult, bh_mult))
    
        inp_img_iris_xyr[0] = (inp_img_iris_xyr[0] * sw_mult)
        inp_img_iris_xyr[1] = (inp_img_iris_xyr[1] * sh_mult)
        inp_img_iris_xyr[2] = (inp_img_iris_xyr[2] * max(sw_mult, sh_mult))
        
        tar_img_iris_xyr[0] = (tar_img_iris_xyr[0] * bw_mult)
        tar_img_iris_xyr[1] = (tar_img_iris_xyr[1] * bh_mult)
        tar_img_iris_xyr[2] = (tar_img_iris_xyr[2] * max(bw_mult, bh_mult))
        
        if self.flip_data and random.random() < 0.5:
            inp_img = inp_img.transpose(Image.FLIP_LEFT_RIGHT) 
            inp_mask = inp_mask.transpose(Image.FLIP_LEFT_RIGHT)
            tar_img = tar_img.transpose(Image.FLIP_LEFT_RIGHT) 
            tar_mask = tar_mask.transpose(Image.FLIP_LEFT_RIGHT)
            inp_img_pupil_xyr[0] = self.input_size[1] - inp_img_pupil_xyr[0]
            tar_img_pupil_xyr[0] = self.input_size[1] - tar_img_pupil_xyr[0]
            inp_img_iris_xyr[0] = self.input_size[1] - inp_img_iris_xyr[0]
            tar_img_iris_xyr[0] = self.input_size[1] - tar_img_iris_xyr[0]

        xform_inp_img = self.transform(inp_img)
        xform_inp_mask = self.transform(inp_mask)
        xform_inp_mask[xform_inp_mask<0] = -1
        xform_inp_mask[xform_inp_mask>=0] = 1
        
        xform_tar_img = self.transform(tar_img)
        xform_tar_mask = self.transform(tar_mask)
        xform_tar_mask[xform_tar_mask<0] = -1
        xform_tar_mask[xform_tar_mask>=0] = 1
        
        return {'inp_img': xform_inp_img, 'tar_img' : xform_tar_img, 'inp_mask': xform_inp_mask, 'tar_mask' : xform_tar_mask, 'inp_img_pxyr' : inp_img_pupil_xyr, 'tar_img_pxyr' : tar_img_pupil_xyr, 'inp_img_ixyr': inp_img_iris_xyr, 'tar_img_ixyr' :  tar_img_iris_xyr, 'identifier':identifier }
    
    def __len__(self):
        return len(self.pairs)
    
    def reset(self):
        print('Shuffling pairs for Warsaw Pupil Dynamics...')
        print('Initializing pairs for Warsaw Pupil Dynamics...')
        self.pairs = []
        self.pair_ids = []
        
        for identifier in tqdm(self.bins_wsd.keys()):
            bin_num_1 = min(list(self.bins_wsd[identifier].keys()))
            bin_num_2 = max(list(self.bins_wsd[identifier].keys()))
            random.shuffle(self.bins_wsd[identifier][bin_num_1])
            random.shuffle(self.bins_wsd[identifier][bin_num_2])
            max_pairs = min(len(self.bins_wsd[identifier][bin_num_1]), len(self.bins_wsd[identifier][bin_num_2]))
            for img_ind in range(max_pairs):
                imagename1 = self.bins_wsd[identifier][bin_num_1][img_ind]
                imagename2 = self.bins_wsd[identifier][bin_num_2][img_ind]
                indiv_id1 = imagename1.split('_')[0]
                indiv_id2 = imagename2.split('_')[0]
                imagepath1 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id1), imagename1)
                maskpath1 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id1), imagename1)
                imagepath2 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id2), imagename2)
                maskpath2 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id2), imagename2)
                self.pairs.append([imagepath1, maskpath1, imagepath2, maskpath2])  
                self.pair_ids.append(identifier)         
        
        print('Initializing pairs for BXGRID...')
        for identifier in tqdm(self.bins_bxgrid.keys()):
            bin_num_1 = min(list(self.bins_bxgrid[identifier].keys()))
            bin_num_2 = max(list(self.bins_bxgrid[identifier].keys()))
            random.shuffle(self.bins_bxgrid[identifier][bin_num_1])
            random.shuffle(self.bins_bxgrid[identifier][bin_num_2])
            max_pairs = min(len(self.bins_bxgrid[identifier][bin_num_1]), len(self.bins_bxgrid[identifier][bin_num_2]))
            for img_ind in range(max_pairs):
                imagename1 = self.bins_bxgrid[identifier][bin_num_1][img_ind]
                imagename2 = self.bins_bxgrid[identifier][bin_num_2][img_ind]
                imagepath1 = os.path.join(self.image_dir_bxgrid, imagename1)
                maskpath1 = os.path.join(self.mask_dir_bxgrid, imagename1)
                imagepath2 = os.path.join(self.image_dir_bxgrid, imagename2)
                maskpath2 = os.path.join(self.mask_dir_bxgrid, imagename2)
                self.pairs.append([imagepath1, maskpath1, imagepath2, maskpath2])  
                self.pair_ids.append(identifier)    
        
        print('Total pairs: ', len(self.pairs))

            
class TripletFromBinAllDatasetBoth(Dataset):
    def __init__(self, bins_path_wsd, parent_dir_wsd, bins_path_bxgrid, parent_dir_bxgrid, normalize=True, brightness_and_contrast = False, flip_data = True, res_mult=1):
        super().__init__()
        
        self.flip_data = flip_data
        self.b_n_c = brightness_and_contrast
        self.input_size=(int(240*res_mult),int(320*res_mult))
        self.res_mult = res_mult
        self.normalize = normalize
        
        self.image_dir_wsd = os.path.join(parent_dir_wsd, 'WarsawData2')
        self.mask_dir_wsd = os.path.join(parent_dir_wsd, 'WarsawMask2')            
        
        with open(bins_path_wsd, 'rb') as wsdbinsFile:
            self.bins_wsd = pkl.load(wsdbinsFile)
        
        self.image_dir_bxgrid = os.path.join(parent_dir_bxgrid, 'iris')
        self.mask_dir_bxgrid = os.path.join(parent_dir_bxgrid, 'mask')       
        
        with open(bins_path_bxgrid, 'rb') as bxgridbinsFile:
            self.bins_bxgrid = pkl.load(bxgridbinsFile)
        
        self.all_ids = list(self.bins_wsd.keys()) + list(self.bins_bxgrid.keys())
        
        print('Initializing triplets for Warsaw Pupil Dynamics...')
        self.triplets = []
        self.genuine_ids = []
        self.imposter_ids = []
        wsd_identifiers = list(self.bins_wsd.keys())
        random.shuffle(wsd_identifiers)
        for identifier in tqdm(wsd_identifiers):
            wsd_identifiers_removed = wsd_identifiers.copy()
            wsd_identifiers_removed.remove(identifier)
            bin_numbers1 = list(self.bins_wsd[identifier].keys())
            bin_numbers2 = list(self.bins_wsd[identifier].keys())
            bin_numbers1.sort()
            bin_numbers2.sort()
            for bin_num_1 in bin_numbers1:
                for bin_num_2 in bin_numbers2:
                    if bin_num_1 != bin_num_2:
                        random.shuffle(self.bins_wsd[identifier][bin_num_1])
                        random.shuffle(self.bins_wsd[identifier][bin_num_2])
                        max_pairs = min(len(self.bins_wsd[identifier][bin_num_1]), len(self.bins_wsd[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            imposter_ind_max = 0
                            while imposter_ind_max == 0:
                                imposter_id = random.choice(wsd_identifiers_removed)
                                imposter_ind_max = len(self.bins_wsd[imposter_id][bin_num_2])
                            imagename1 = self.bins_wsd[identifier][bin_num_1][img_ind]
                            imagename2 = self.bins_wsd[identifier][bin_num_2][img_ind]
                            imagename3 = self.bins_wsd[imposter_id][bin_num_2][random.randrange(0, imposter_ind_max)]
                            indiv_id1 = imagename1.split('_')[0]
                            indiv_id2 = imagename2.split('_')[0]
                            indiv_id3 = imagename3.split('_')[0]
                            imagepath1 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id1), imagename1)
                            maskpath1 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id1), imagename1)
                            imagepath2 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id2), imagename2)
                            maskpath2 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id2), imagename2)
                            imagepath3 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id3), imagename3)
                            maskpath3 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id3), imagename3)
                            self.triplets.append([imagepath1, maskpath1, imagepath2, maskpath2, imagepath3, maskpath3, identifier, imposter_id])
                        random.shuffle(self.bins_wsd[identifier][bin_num_1])
                        random.shuffle(self.bins_wsd[identifier][bin_num_2])
                        for img_ind in range(max_pairs):
                            imposter_ind_max = 0
                            while imposter_ind_max == 0:
                                imposter_id = random.choice(wsd_identifiers_removed)
                                imposter_ind_max = len(self.bins_wsd[imposter_id][bin_num_1])
                            imagename1 = self.bins_wsd[identifier][bin_num_2][img_ind]
                            imagename2 = self.bins_wsd[identifier][bin_num_1][img_ind]
                            imagename3 = self.bins_wsd[imposter_id][bin_num_1][random.randrange(0, imposter_ind_max)]
                            indiv_id1 = imagename1.split('_')[0]
                            indiv_id2 = imagename2.split('_')[0]
                            indiv_id3 = imagename3.split('_')[0]
                            imagepath1 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id1), imagename1)
                            maskpath1 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id1), imagename1)
                            imagepath2 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id2), imagename2)
                            maskpath2 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id2), imagename2)
                            imagepath3 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id3), imagename3)
                            maskpath3 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id3), imagename3)
                            self.triplets.append([imagepath1, maskpath1, imagepath2, maskpath2, imagepath3, maskpath3, identifier, imposter_id])              
        
        print('Initializing triplets for BXGRID...')
        bxgrid_identifiers = list(self.bins_bxgrid.keys())
        random.shuffle(bxgrid_identifiers)
        for identifier in tqdm(bxgrid_identifiers):
            bxgrid_identifiers_removed = bxgrid_identifiers.copy()
            bxgrid_identifiers_removed.remove(identifier)
            bin_numbers1 = list(self.bins_bxgrid[identifier].keys())
            bin_numbers2 = list(self.bins_bxgrid[identifier].keys())
            bin_numbers1.sort()
            bin_numbers2.sort()
            for bin_num_1 in bin_numbers1:
                for bin_num_2 in bin_numbers2:
                    if bin_num_1 != bin_num_2:
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_1])
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_2])
                        max_pairs = min(len(self.bins_bxgrid[identifier][bin_num_1]), len(self.bins_bxgrid[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            imposter_ind_max = 0
                            while imposter_ind_max == 0:
                                imposter_id = random.choice(bxgrid_identifiers_removed)
                                imposter_ind_max = len(self.bins_bxgrid[imposter_id][bin_num_2])
                            imagename1 = self.bins_bxgrid[identifier][bin_num_1][img_ind]
                            imagename2 = self.bins_bxgrid[identifier][bin_num_2][img_ind]
                            imagename3 = self.bins_bxgrid[imposter_id][bin_num_2][random.randrange(0, imposter_ind_max)]
                            imagepath1 = os.path.join(self.image_dir_bxgrid, imagename1)
                            maskpath1 = os.path.join(self.mask_dir_bxgrid, imagename1)
                            imagepath2 = os.path.join(self.image_dir_bxgrid, imagename2)
                            maskpath2 = os.path.join(self.mask_dir_bxgrid, imagename2)
                            imagepath3 = os.path.join(self.image_dir_bxgrid, imagename3)
                            maskpath3 = os.path.join(self.mask_dir_bxgrid, imagename3)
                            self.triplets.append([imagepath1, maskpath1, imagepath2, maskpath2, imagepath3, maskpath3, identifier, imposter_id])
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_1])
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_2])
                        for img_ind in range(max_pairs):
                            imposter_ind_max = 0
                            while imposter_ind_max == 0:
                                imposter_id = random.choice(bxgrid_identifiers_removed)
                                imposter_ind_max = len(self.bins_bxgrid[imposter_id][bin_num_1])
                            imagename1 = self.bins_bxgrid[identifier][bin_num_2][img_ind]
                            imagename2 = self.bins_bxgrid[identifier][bin_num_1][img_ind]
                            imagename3 = self.bins_bxgrid[imposter_id][bin_num_1][random.randrange(0, imposter_ind_max)]
                            imagepath1 = os.path.join(self.image_dir_bxgrid, imagename1)
                            maskpath1 = os.path.join(self.mask_dir_bxgrid, imagename1)
                            imagepath2 = os.path.join(self.image_dir_bxgrid, imagename2)
                            maskpath2 = os.path.join(self.mask_dir_bxgrid, imagename2)
                            imagepath3 = os.path.join(self.image_dir_bxgrid, imagename3)
                            maskpath3 = os.path.join(self.mask_dir_bxgrid, imagename3)
                            self.triplets.append([imagepath1, maskpath1, imagepath2, maskpath2, imagepath3, maskpath3, identifier, imposter_id])
                            
        print('Total triplets: ', len(self.triplets))
        
        if self.normalize:
            self.transform = transforms.Compose([
                transforms.Resize([self.input_size[0], self.input_size[1]]),
                transforms.ToTensor(),
                transforms.Normalize((0.5, ), (0.5, ))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize([self.input_size[0], self.input_size[1]]),
                transforms.ToTensor()
            ])
        
        self.pupil_iris_xyrs_path_wsd = os.path.join(parent_dir_wsd, 'pupil_iris_xyrs2.pkl')
        with open(self.pupil_iris_xyrs_path_wsd, 'rb') as pupilirisxyrsFile:
            self.pupil_iris_xyrs_wsd = pkl.load(pupilirisxyrsFile)
        
        self.pupil_iris_xyrs_path_bxgrid = os.path.join(parent_dir_bxgrid, 'pupil_iris_xyrs.pkl')
        with open(self.pupil_iris_xyrs_path_bxgrid, 'rb') as pupilirisxyrsFile:
            self.pupil_iris_xyrs_bxgrid = pkl.load(pupilirisxyrsFile)
        
        self.pupil_xyrs = {}
        self.iris_xyrs = {}
        
        for imagename in self.pupil_iris_xyrs_wsd.keys():
            indiv_id = imagename.split('_')[0]
            imagepath = os.path.join(os.path.join(self.image_dir_wsd, indiv_id), imagename)
            self.pupil_xyrs[imagepath] = self.pupil_iris_xyrs_wsd[imagename]['pxyr']
            self.iris_xyrs[imagepath] = self.pupil_iris_xyrs_wsd[imagename]['ixyr']
        
        for imagename in self.pupil_iris_xyrs_bxgrid.keys():
            imagepath = os.path.join(self.image_dir_bxgrid, imagename)
            self.pupil_xyrs[imagepath] = self.pupil_iris_xyrs_bxgrid[imagename]['pxyr']
            self.iris_xyrs[imagepath] = self.pupil_iris_xyrs_bxgrid[imagename]['ixyr']
        
       
    
    def load_image(self, file):
        return Image.open(file).convert('L')
        
    def __getitem__(self, index):
    
        triplet = self.triplets[index]
        
        inp_img_path = triplet[0].strip()
        inp_mask_path = triplet[1].strip()
        tar_img_path = triplet[2].strip()
        tar_mask_path = triplet[3].strip()
        imp_img_path = triplet[4].strip()
        imp_mask_path = triplet[5].strip()
        
        inp_img = self.load_image(inp_img_path)
        inp_mask = self.load_image(inp_mask_path)
        
        tar_img = self.load_image(tar_img_path)
        tar_mask = self.load_image(tar_mask_path)
        
        imp_img = self.load_image(imp_img_path)
        imp_mask = self.load_image(imp_mask_path)
        
        inp_img_pupil_xyr = torch.FloatTensor(self.pupil_xyrs[inp_img_path])
        inp_img_iris_xyr = torch.FloatTensor(self.iris_xyrs[inp_img_path])
        
        tar_img_pupil_xyr = torch.FloatTensor(self.pupil_xyrs[tar_img_path])
        tar_img_iris_xyr = torch.FloatTensor(self.iris_xyrs[tar_img_path])
        
        imp_img_pupil_xyr = torch.FloatTensor(self.pupil_xyrs[imp_img_path])
        imp_img_iris_xyr = torch.FloatTensor(self.iris_xyrs[imp_img_path])
        
        sw, sh = inp_img.size
        sw_mult = (self.input_size[1]/sw)
        sh_mult = (self.input_size[0]/sh)
        
        bw, bh = tar_img.size
        bw_mult = (self.input_size[1]/bw)
        bh_mult = (self.input_size[0]/bh)
        
        biw, bih = imp_img.size
        biw_mult = (self.input_size[1]/biw)
        bih_mult = (self.input_size[0]/bih)
        
        inp_img_pupil_xyr[0] = (inp_img_pupil_xyr[0] * sw_mult)
        inp_img_pupil_xyr[1] = (inp_img_pupil_xyr[1] * sh_mult)
        inp_img_pupil_xyr[2] = (inp_img_pupil_xyr[2] * max(sw_mult, sh_mult))
        
        tar_img_pupil_xyr[0] = (tar_img_pupil_xyr[0] * bw_mult)
        tar_img_pupil_xyr[1] = (tar_img_pupil_xyr[1] * bh_mult)
        tar_img_pupil_xyr[2] = (tar_img_pupil_xyr[2] * max(bw_mult, bh_mult))
        
        imp_img_pupil_xyr[0] = (imp_img_pupil_xyr[0] * biw_mult)
        imp_img_pupil_xyr[1] = (imp_img_pupil_xyr[1] * bih_mult)
        imp_img_pupil_xyr[2] = (imp_img_pupil_xyr[2] * max(biw_mult, bih_mult))
    
        inp_img_iris_xyr[0] = (inp_img_iris_xyr[0] * sw_mult)
        inp_img_iris_xyr[1] = (inp_img_iris_xyr[1] * sh_mult)
        inp_img_iris_xyr[2] = (inp_img_iris_xyr[2] * max(sw_mult, sh_mult))
        
        tar_img_iris_xyr[0] = (tar_img_iris_xyr[0] * bw_mult)
        tar_img_iris_xyr[1] = (tar_img_iris_xyr[1] * bh_mult)
        tar_img_iris_xyr[2] = (tar_img_iris_xyr[2] * max(bw_mult, bh_mult))
        
        imp_img_iris_xyr[0] = (imp_img_iris_xyr[0] * biw_mult)
        imp_img_iris_xyr[1] = (imp_img_iris_xyr[1] * bih_mult)
        imp_img_iris_xyr[2] = (imp_img_iris_xyr[2] * max(biw_mult, bih_mult))
        
        if self.flip_data and random.random() < 0.5:
            inp_img = inp_img.transpose(Image.FLIP_LEFT_RIGHT) 
            inp_mask = inp_mask.transpose(Image.FLIP_LEFT_RIGHT)
            tar_img = tar_img.transpose(Image.FLIP_LEFT_RIGHT) 
            tar_mask = tar_mask.transpose(Image.FLIP_LEFT_RIGHT)
            inp_img_pupil_xyr[0] = self.input_size[1] - inp_img_pupil_xyr[0]
            tar_img_pupil_xyr[0] = self.input_size[1] - tar_img_pupil_xyr[0]
            inp_img_iris_xyr[0] = self.input_size[1] - inp_img_iris_xyr[0]
            tar_img_iris_xyr[0] = self.input_size[1] - tar_img_iris_xyr[0]
        
        if self.flip_data and random.random() < 0.5:
            imp_img = imp_img.transpose(Image.FLIP_LEFT_RIGHT)
            imp_mask = tar_mask.transpose(Image.FLIP_LEFT_RIGHT)
            imp_img_pupil_xyr[0] = self.input_size[1] - imp_img_pupil_xyr[0]
            imp_img_iris_xyr[0] = self.input_size[1] - imp_img_iris_xyr[0]
        
        if self.b_n_c and random.random() < 0.5:
            random_brightness = random.randint(-20, 20)
            inp_img = Image.fromarray(np.clip(np.array(inp_img) + random_brightness, 0, 255).astype(np.uint8))
            tar_img = Image.fromarray(np.clip(np.array(tar_img) + random_brightness, 0, 255).astype(np.uint8))
            imp_img = Image.fromarray(np.clip(np.array(imp_img) + random_brightness, 0, 255).astype(np.uint8))
        
        if self.b_n_c and random.random() < 0.5:
            contrast = random.randint(-60, 60)
            factor = (259 * (contrast + 255)) / (255 * (259 - contrast))
            inp_img = Image.fromarray(np.clip(np.around(factor * (np.float32(inp_img) - 127.5) + 127.5), 0, 255).astype(np.uint8))
            tar_img = Image.fromarray(np.clip(np.around(factor * (np.float32(tar_img) - 127.5) + 127.5), 0, 255).astype(np.uint8))
            imp_img = Image.fromarray(np.clip(np.around(factor * (np.float32(inp_img) - 127.5) + 127.5), 0, 255).astype(np.uint8))
        
        xform_inp_img = self.transform(inp_img)
        xform_inp_mask = self.transform(inp_mask)
        if self.normalize:
            xform_inp_mask[xform_inp_mask<0] = -1
            xform_inp_mask[xform_inp_mask>=0] = 1
        else:
            xform_inp_mask[xform_inp_mask<0] = 0
            xform_inp_mask[xform_inp_mask>=0] = 1
        
        xform_tar_img = self.transform(tar_img)
        xform_tar_mask = self.transform(tar_mask)
        if self.normalize:
            xform_tar_mask[xform_tar_mask<0] = -1
            xform_tar_mask[xform_tar_mask>=0] = 1
        else:
            xform_tar_mask[xform_tar_mask<0] = 0
            xform_tar_mask[xform_tar_mask>=0] = 1
        
        xform_imp_img = self.transform(imp_img)
        xform_imp_mask = self.transform(imp_mask)
        if self.normalize:
            xform_imp_mask[xform_imp_mask<0] = -1
            xform_imp_mask[xform_imp_mask>=0] = 1
        else:
            xform_imp_mask[xform_imp_mask<0] = 0
            xform_imp_mask[xform_imp_mask>=0] = 1
        
        return {'inp_img': xform_inp_img, 'tar_img' : xform_tar_img, 'imp_img' : xform_imp_img, 'inp_mask': xform_inp_mask, 'tar_mask' : xform_tar_mask, 'imp_mask' : xform_imp_mask, 'inp_img_pxyr' : inp_img_pupil_xyr, 'tar_img_pxyr' : tar_img_pupil_xyr, 'imp_img_pxyr' : imp_img_pupil_xyr, 'inp_img_ixyr': inp_img_iris_xyr, 'tar_img_ixyr' :  tar_img_iris_xyr, 'imp_img_ixyr' : imp_img_iris_xyr, 'identifier' : triplet[6], 'imposter_id' : triplet[7]}
    
    
    def __len__(self):
        return len(self.triplets)
    
    def reset(self):
        print('Shuffling triplets for WSD...')
        self.triplets = []
        self.genuine_ids = []
        self.imposter_ids = []
        wsd_identifiers = list(self.bins_wsd.keys())
        random.shuffle(wsd_identifiers)
        for identifier in tqdm(wsd_identifiers):
            wsd_identifiers_removed = wsd_identifiers.copy()
            wsd_identifiers_removed.remove(identifier)
            bin_numbers1 = list(self.bins_wsd[identifier].keys())
            bin_numbers2 = list(self.bins_wsd[identifier].keys())
            bin_numbers1.sort()
            bin_numbers2.sort()
            for bin_num_1 in bin_numbers1:
                for bin_num_2 in bin_numbers2:
                    if bin_num_1 != bin_num_2:
                        random.shuffle(self.bins_wsd[identifier][bin_num_1])
                        random.shuffle(self.bins_wsd[identifier][bin_num_2])
                        max_pairs = min(len(self.bins_wsd[identifier][bin_num_1]), len(self.bins_wsd[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            imposter_ind_max = 0
                            while imposter_ind_max == 0:
                                imposter_id = random.choice(wsd_identifiers_removed)
                                imposter_ind_max = len(self.bins_wsd[imposter_id][bin_num_2])
                            imagename1 = self.bins_wsd[identifier][bin_num_1][img_ind]
                            imagename2 = self.bins_wsd[identifier][bin_num_2][img_ind]
                            imagename3 = self.bins_wsd[imposter_id][bin_num_2][random.randrange(0, imposter_ind_max)]
                            indiv_id1 = imagename1.split('_')[0]
                            indiv_id2 = imagename2.split('_')[0]
                            indiv_id3 = imagename3.split('_')[0]
                            imagepath1 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id1), imagename1)
                            maskpath1 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id1), imagename1)
                            imagepath2 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id2), imagename2)
                            maskpath2 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id2), imagename2)
                            imagepath3 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id3), imagename3)
                            maskpath3 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id3), imagename3)
                            self.triplets.append([imagepath1, maskpath1, imagepath2, maskpath2, imagepath3, maskpath3, identifier, imposter_id])
                        random.shuffle(self.bins_wsd[identifier][bin_num_1])
                        random.shuffle(self.bins_wsd[identifier][bin_num_2])
                        for img_ind in range(max_pairs):
                            imposter_ind_max = 0
                            while imposter_ind_max == 0:
                                imposter_id = random.choice(wsd_identifiers_removed)
                                imposter_ind_max = len(self.bins_wsd[imposter_id][bin_num_1])
                            imagename1 = self.bins_wsd[identifier][bin_num_2][img_ind]
                            imagename2 = self.bins_wsd[identifier][bin_num_1][img_ind]
                            imagename3 = self.bins_wsd[imposter_id][bin_num_1][random.randrange(0, imposter_ind_max)]
                            indiv_id1 = imagename1.split('_')[0]
                            indiv_id2 = imagename2.split('_')[0]
                            indiv_id3 = imagename3.split('_')[0]
                            imagepath1 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id1), imagename1)
                            maskpath1 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id1), imagename1)
                            imagepath2 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id2), imagename2)
                            maskpath2 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id2), imagename2)
                            imagepath3 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id3), imagename3)
                            maskpath3 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id3), imagename3)
                            self.triplets.append([imagepath1, maskpath1, imagepath2, maskpath2, imagepath3, maskpath3, identifier, imposter_id])              
        
        print('Shuffling triplets for BXGRID...')
        bxgrid_identifiers = list(self.bins_bxgrid.keys())
        random.shuffle(bxgrid_identifiers)
        for identifier in tqdm(bxgrid_identifiers):
            bxgrid_identifiers_removed = bxgrid_identifiers.copy()
            bxgrid_identifiers_removed.remove(identifier)
            bin_numbers1 = list(self.bins_bxgrid[identifier].keys())
            bin_numbers2 = list(self.bins_bxgrid[identifier].keys())
            bin_numbers1.sort()
            bin_numbers2.sort()
            for bin_num_1 in bin_numbers1:
                for bin_num_2 in bin_numbers2:
                    if bin_num_1 != bin_num_2:
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_1])
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_2])
                        max_pairs = min(len(self.bins_bxgrid[identifier][bin_num_1]), len(self.bins_bxgrid[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            imposter_ind_max = 0
                            while imposter_ind_max == 0:
                                imposter_id = random.choice(bxgrid_identifiers_removed)
                                imposter_ind_max = len(self.bins_bxgrid[imposter_id][bin_num_2])
                            imagename1 = self.bins_bxgrid[identifier][bin_num_1][img_ind]
                            imagename2 = self.bins_bxgrid[identifier][bin_num_2][img_ind]
                            imagename3 = self.bins_bxgrid[imposter_id][bin_num_2][random.randrange(0, imposter_ind_max)]
                            imagepath1 = os.path.join(self.image_dir_bxgrid, imagename1)
                            maskpath1 = os.path.join(self.mask_dir_bxgrid, imagename1)
                            imagepath2 = os.path.join(self.image_dir_bxgrid, imagename2)
                            maskpath2 = os.path.join(self.mask_dir_bxgrid, imagename2)
                            imagepath3 = os.path.join(self.image_dir_bxgrid, imagename3)
                            maskpath3 = os.path.join(self.mask_dir_bxgrid, imagename3)
                            self.triplets.append([imagepath1, maskpath1, imagepath2, maskpath2, imagepath3, maskpath3, identifier, imposter_id])
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_1])
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_2])
                        for img_ind in range(max_pairs):
                            imposter_ind_max = 0
                            while imposter_ind_max == 0:
                                imposter_id = random.choice(bxgrid_identifiers_removed)
                                imposter_ind_max = len(self.bins_bxgrid[imposter_id][bin_num_1])
                            imagename1 = self.bins_bxgrid[identifier][bin_num_2][img_ind]
                            imagename2 = self.bins_bxgrid[identifier][bin_num_1][img_ind]
                            imagename3 = self.bins_bxgrid[imposter_id][bin_num_1][random.randrange(0, imposter_ind_max)]
                            imagepath1 = os.path.join(self.image_dir_bxgrid, imagename1)
                            maskpath1 = os.path.join(self.mask_dir_bxgrid, imagename1)
                            imagepath2 = os.path.join(self.image_dir_bxgrid, imagename2)
                            maskpath2 = os.path.join(self.mask_dir_bxgrid, imagename2)
                            imagepath3 = os.path.join(self.image_dir_bxgrid, imagename3)
                            maskpath3 = os.path.join(self.mask_dir_bxgrid, imagename3)
                            self.triplets.append([imagepath1, maskpath1, imagepath2, maskpath2, imagepath3, maskpath3, identifier, imposter_id])
                            
        print('Total triplets: ', len(self.triplets))
        
class TripletFromBinAllDatasetBothCropped(Dataset):
    def __init__(self, bins_path_wsd, parent_dir_wsd, bins_path_bxgrid, parent_dir_bxgrid, iris_crop=True, normalize=True, brightness_and_contrast = False, flip_data = True, res_mult=1):
        super().__init__()
        self.iris_crop = iris_crop
        self.flip_data = flip_data
        self.b_n_c = brightness_and_contrast
        self.input_size=(int(256*res_mult),int(256*res_mult))
        self.res_mult = res_mult
        self.normalize = normalize
        
        self.image_dir_wsd = os.path.join(parent_dir_wsd, 'WarsawData2')
        self.mask_dir_wsd = os.path.join(parent_dir_wsd, 'WarsawMasksCoarse')            
        
        with open(bins_path_wsd, 'rb') as wsdbinsFile:
            self.bins_wsd = pkl.load(wsdbinsFile)
        
        self.image_dir_bxgrid = os.path.join(parent_dir_bxgrid, 'iris')
        self.mask_dir_bxgrid = os.path.join(parent_dir_bxgrid, 'masks_coarse')       
        
        with open(bins_path_bxgrid, 'rb') as bxgridbinsFile:
            self.bins_bxgrid = pkl.load(bxgridbinsFile)
        
        self.all_ids = list(self.bins_wsd.keys()) + list(self.bins_bxgrid.keys())
        
        print('Initializing triplets for Warsaw Pupil Dynamics...')
        self.triplets = []
        self.genuine_ids = []
        self.imposter_ids = []
        wsd_identifiers = list(self.bins_wsd.keys())
        random.shuffle(wsd_identifiers)
        for identifier in tqdm(wsd_identifiers):
            wsd_identifiers_removed = wsd_identifiers.copy()
            wsd_identifiers_removed.remove(identifier)
            bin_numbers1 = list(self.bins_wsd[identifier].keys())
            bin_numbers2 = list(self.bins_wsd[identifier].keys())
            bin_numbers1.sort()
            bin_numbers2.sort()
            for bin_num_1 in bin_numbers1:
                for bin_num_2 in bin_numbers2:
                    if bin_num_1 != bin_num_2:
                        random.shuffle(self.bins_wsd[identifier][bin_num_1])
                        random.shuffle(self.bins_wsd[identifier][bin_num_2])
                        max_pairs = min(len(self.bins_wsd[identifier][bin_num_1]), len(self.bins_wsd[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            imposter_ind_max = 0
                            while imposter_ind_max == 0:
                                imposter_id = random.choice(wsd_identifiers_removed)
                                imposter_ind_max = len(self.bins_wsd[imposter_id][bin_num_2])
                            imagename1 = self.bins_wsd[identifier][bin_num_1][img_ind]
                            imagename2 = self.bins_wsd[identifier][bin_num_2][img_ind]
                            imagename3 = self.bins_wsd[imposter_id][bin_num_2][random.randrange(0, imposter_ind_max)]
                            indiv_id1 = imagename1.split('_')[0]
                            indiv_id2 = imagename2.split('_')[0]
                            indiv_id3 = imagename3.split('_')[0]
                            imagepath1 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id1), imagename1)
                            maskpath1 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id1), imagename1)
                            imagepath2 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id2), imagename2)
                            maskpath2 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id2), imagename2)
                            imagepath3 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id3), imagename3)
                            maskpath3 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id3), imagename3)
                            self.triplets.append([imagepath1, maskpath1, imagepath2, maskpath2, imagepath3, maskpath3, identifier, imposter_id])
                        random.shuffle(self.bins_wsd[identifier][bin_num_1])
                        random.shuffle(self.bins_wsd[identifier][bin_num_2])
                        for img_ind in range(max_pairs):
                            imposter_ind_max = 0
                            while imposter_ind_max == 0:
                                imposter_id = random.choice(wsd_identifiers_removed)
                                imposter_ind_max = len(self.bins_wsd[imposter_id][bin_num_1])
                            imagename1 = self.bins_wsd[identifier][bin_num_2][img_ind]
                            imagename2 = self.bins_wsd[identifier][bin_num_1][img_ind]
                            imagename3 = self.bins_wsd[imposter_id][bin_num_1][random.randrange(0, imposter_ind_max)]
                            indiv_id1 = imagename1.split('_')[0]
                            indiv_id2 = imagename2.split('_')[0]
                            indiv_id3 = imagename3.split('_')[0]
                            imagepath1 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id1), imagename1)
                            maskpath1 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id1), imagename1)
                            imagepath2 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id2), imagename2)
                            maskpath2 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id2), imagename2)
                            imagepath3 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id3), imagename3)
                            maskpath3 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id3), imagename3)
                            self.triplets.append([imagepath1, maskpath1, imagepath2, maskpath2, imagepath3, maskpath3, identifier, imposter_id])              
        
        print('Initializing triplets for BXGRID...')
        bxgrid_identifiers = list(self.bins_bxgrid.keys())
        random.shuffle(bxgrid_identifiers)
        for identifier in tqdm(bxgrid_identifiers):
            bxgrid_identifiers_removed = bxgrid_identifiers.copy()
            bxgrid_identifiers_removed.remove(identifier)
            bin_numbers1 = list(self.bins_bxgrid[identifier].keys())
            bin_numbers2 = list(self.bins_bxgrid[identifier].keys())
            bin_numbers1.sort()
            bin_numbers2.sort()
            for bin_num_1 in bin_numbers1:
                for bin_num_2 in bin_numbers2:
                    if bin_num_1 != bin_num_2:
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_1])
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_2])
                        max_pairs = min(len(self.bins_bxgrid[identifier][bin_num_1]), len(self.bins_bxgrid[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            imposter_ind_max = 0
                            while imposter_ind_max == 0:
                                imposter_id = random.choice(bxgrid_identifiers_removed)
                                imposter_ind_max = len(self.bins_bxgrid[imposter_id][bin_num_2])
                            imagename1 = self.bins_bxgrid[identifier][bin_num_1][img_ind]
                            imagename2 = self.bins_bxgrid[identifier][bin_num_2][img_ind]
                            imagename3 = self.bins_bxgrid[imposter_id][bin_num_2][random.randrange(0, imposter_ind_max)]
                            imagepath1 = os.path.join(self.image_dir_bxgrid, imagename1)
                            maskpath1 = os.path.join(self.mask_dir_bxgrid, imagename1)
                            imagepath2 = os.path.join(self.image_dir_bxgrid, imagename2)
                            maskpath2 = os.path.join(self.mask_dir_bxgrid, imagename2)
                            imagepath3 = os.path.join(self.image_dir_bxgrid, imagename3)
                            maskpath3 = os.path.join(self.mask_dir_bxgrid, imagename3)
                            self.triplets.append([imagepath1, maskpath1, imagepath2, maskpath2, imagepath3, maskpath3, identifier, imposter_id])
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_1])
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_2])
                        for img_ind in range(max_pairs):
                            imposter_ind_max = 0
                            while imposter_ind_max == 0:
                                imposter_id = random.choice(bxgrid_identifiers_removed)
                                imposter_ind_max = len(self.bins_bxgrid[imposter_id][bin_num_1])
                            imagename1 = self.bins_bxgrid[identifier][bin_num_2][img_ind]
                            imagename2 = self.bins_bxgrid[identifier][bin_num_1][img_ind]
                            imagename3 = self.bins_bxgrid[imposter_id][bin_num_1][random.randrange(0, imposter_ind_max)]
                            imagepath1 = os.path.join(self.image_dir_bxgrid, imagename1)
                            maskpath1 = os.path.join(self.mask_dir_bxgrid, imagename1)
                            imagepath2 = os.path.join(self.image_dir_bxgrid, imagename2)
                            maskpath2 = os.path.join(self.mask_dir_bxgrid, imagename2)
                            imagepath3 = os.path.join(self.image_dir_bxgrid, imagename3)
                            maskpath3 = os.path.join(self.mask_dir_bxgrid, imagename3)
                            self.triplets.append([imagepath1, maskpath1, imagepath2, maskpath2, imagepath3, maskpath3, identifier, imposter_id])
                            
        print('Total triplets: ', len(self.triplets))
        

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5, ))
        ])
        
        self.pupil_iris_xyrs_path_wsd = os.path.join(parent_dir_wsd, 'pupil_iris_xyrs_new.pkl')
        with open(self.pupil_iris_xyrs_path_wsd, 'rb') as pupilirisxyrsFile:
            self.pupil_iris_xyrs_wsd = pkl.load(pupilirisxyrsFile)
        
        self.pupil_iris_xyrs_path_bxgrid = os.path.join(parent_dir_bxgrid, 'pupil_iris_xyrs_new.pkl')
        with open(self.pupil_iris_xyrs_path_bxgrid, 'rb') as pupilirisxyrsFile:
            self.pupil_iris_xyrs_bxgrid = pkl.load(pupilirisxyrsFile)
        
        self.pupil_xyrs = {}
        self.iris_xyrs = {}
        
        for imagename in self.pupil_iris_xyrs_wsd.keys():
            imagepath = os.path.join(self.image_dir_wsd, imagename + '.bmp')
            self.pupil_xyrs[imagepath] = self.pupil_iris_xyrs_wsd[imagename]['pxyr']
            self.iris_xyrs[imagepath] = self.pupil_iris_xyrs_wsd[imagename]['ixyr']
        
        for imagename in self.pupil_iris_xyrs_bxgrid.keys():
            imagepath = os.path.join(self.image_dir_bxgrid, imagename + '.png')
            self.pupil_xyrs[imagepath] = self.pupil_iris_xyrs_bxgrid[imagename]['pxyr']
            self.iris_xyrs[imagepath] = self.pupil_iris_xyrs_bxgrid[imagename]['ixyr']
        
        self.heavy_augment_prob = 0.3
    
    def load_image(self, file):
        return Image.open(file).convert('L')
        
    def __getitem__(self, index):
    
        triplet = self.triplets[index]
        
        inp_img_path = triplet[0].strip()
        inp_mask_path = triplet[1].strip()
        tar_img_path = triplet[2].strip()
        tar_mask_path = triplet[3].strip()
        imp_img_path = triplet[4].strip()
        imp_mask_path = triplet[5].strip()
        
        inp_img = self.load_image(inp_img_path)
        inp_mask = self.load_image(inp_mask_path)
        
        tar_img = self.load_image(tar_img_path)
        tar_mask = self.load_image(tar_mask_path)
        
        imp_img = self.load_image(imp_img_path)
        imp_mask = self.load_image(imp_mask_path)
        
        inp_img_pupil_xyr = torch.FloatTensor(self.pupil_xyrs[inp_img_path])
        inp_img_iris_xyr = torch.FloatTensor(self.iris_xyrs[inp_img_path])
        
        tar_img_pupil_xyr = torch.FloatTensor(self.pupil_xyrs[tar_img_path])
        tar_img_iris_xyr = torch.FloatTensor(self.iris_xyrs[tar_img_path])
        
        imp_img_pupil_xyr = torch.FloatTensor(self.pupil_xyrs[imp_img_path])
        imp_img_iris_xyr = torch.FloatTensor(self.iris_xyrs[imp_img_path])
        
        sw, sh = inp_img.size
        bw, bh = tar_img.size
        iw, ih = imp_img.size
        
        if self.flip_data and random.random() < 0.5:
            inp_img = inp_img.transpose(Image.FLIP_LEFT_RIGHT) 
            inp_mask = inp_mask.transpose(Image.FLIP_LEFT_RIGHT)
            tar_img = tar_img.transpose(Image.FLIP_LEFT_RIGHT) 
            tar_mask = tar_mask.transpose(Image.FLIP_LEFT_RIGHT)
            inp_img_pupil_xyr[0] = sw - inp_img_pupil_xyr[0]
            tar_img_pupil_xyr[0] = bw - tar_img_pupil_xyr[0]
            inp_img_iris_xyr[0] = sw - inp_img_iris_xyr[0]
            tar_img_iris_xyr[0] = bw - tar_img_iris_xyr[0]
        
        if self.flip_data and random.random() < 0.5:
            imp_img = imp_img.transpose(Image.FLIP_LEFT_RIGHT)
            imp_mask = imp_mask.transpose(Image.FLIP_LEFT_RIGHT)
            imp_img_pupil_xyr[0] = iw - imp_img_pupil_xyr[0]
            imp_img_iris_xyr[0] = iw - imp_img_iris_xyr[0]
        
        if self.iris_crop:
            #inp_img.save('./check1.png')
            #print((inp_img_iris_xyr[0].item() - inp_img_iris_xyr[2].item(), inp_img_iris_xyr[1].item() - inp_img_iris_xyr[2].item(), inp_img_iris_xyr[0].item() + inp_img_iris_xyr[2].item(), inp_img_iris_xyr[1].item() + inp_img_iris_xyr[2].item()))
            inp_img = inp_img.crop((inp_img_iris_xyr[0].item() - inp_img_iris_xyr[2].item(), inp_img_iris_xyr[1].item() - inp_img_iris_xyr[2].item(), inp_img_iris_xyr[0].item() + inp_img_iris_xyr[2].item(), inp_img_iris_xyr[1].item() + inp_img_iris_xyr[2].item()))
            inp_mask = inp_mask.crop((inp_img_iris_xyr[0].item() - inp_img_iris_xyr[2].item(), inp_img_iris_xyr[1].item() - inp_img_iris_xyr[2].item(), inp_img_iris_xyr[0].item() + inp_img_iris_xyr[2].item(), inp_img_iris_xyr[1].item() + inp_img_iris_xyr[2].item()))
            sw_crop, sh_crop = inp_img.size
            inp_img_pupil_xyr[0] = inp_img_pupil_xyr[0] - inp_img_iris_xyr[0] + sw_crop/2
            inp_img_pupil_xyr[1] = inp_img_pupil_xyr[1] - inp_img_iris_xyr[1] + sh_crop/2
            inp_img_iris_xyr[0] = sw_crop/2
            inp_img_iris_xyr[1] = sh_crop/2            
            
            tar_img = tar_img.crop((tar_img_iris_xyr[0].item() - tar_img_iris_xyr[2].item(), tar_img_iris_xyr[1].item() - tar_img_iris_xyr[2].item(), tar_img_iris_xyr[0].item() + tar_img_iris_xyr[2].item(), tar_img_iris_xyr[1].item() + tar_img_iris_xyr[2].item()))
            tar_mask = tar_mask.crop((tar_img_iris_xyr[0].item() - tar_img_iris_xyr[2].item(), tar_img_iris_xyr[1].item() - tar_img_iris_xyr[2].item(), tar_img_iris_xyr[0].item() + tar_img_iris_xyr[2].item(), tar_img_iris_xyr[1].item() + tar_img_iris_xyr[2].item()))
            bw_crop, bh_crop = tar_img.size
            tar_img_pupil_xyr[0] = tar_img_pupil_xyr[0] - tar_img_iris_xyr[0] + bw_crop/2
            tar_img_pupil_xyr[1] = tar_img_pupil_xyr[1] - tar_img_iris_xyr[1] + bh_crop/2
            tar_img_iris_xyr[0] = bw_crop/2
            tar_img_iris_xyr[1] = bh_crop/2
            
            imp_img = imp_img.crop((imp_img_iris_xyr[0].item() - imp_img_iris_xyr[2].item(), imp_img_iris_xyr[1].item() - imp_img_iris_xyr[2].item(), imp_img_iris_xyr[0].item() + imp_img_iris_xyr[2].item(), imp_img_iris_xyr[1].item() + imp_img_iris_xyr[2].item()))
            imp_mask = imp_mask.crop((imp_img_iris_xyr[0].item() - imp_img_iris_xyr[2].item(), imp_img_iris_xyr[1].item() - imp_img_iris_xyr[2].item(), imp_img_iris_xyr[0].item() + imp_img_iris_xyr[2].item(), imp_img_iris_xyr[1].item() + imp_img_iris_xyr[2].item()))
            iw_crop, ih_crop = imp_img.size
            imp_img_pupil_xyr[0] = imp_img_pupil_xyr[0] - imp_img_iris_xyr[0] + iw_crop/2
            imp_img_pupil_xyr[1] = imp_img_pupil_xyr[1] - imp_img_iris_xyr[1] + ih_crop/2
            imp_img_iris_xyr[0] = iw_crop/2
            imp_img_iris_xyr[1] = ih_crop/2
            
            '''
            inp_img_vis = np.uint8(inp_img.convert('RGB'))
            try:
                inp_img_vis = cv2.circle(inp_img_vis, (int(inp_img_pupil_xyr[0].item()),int(inp_img_pupil_xyr[1].item())), int(inp_img_pupil_xyr[2].item()), (0, 0, 255), 2)
            except:
                print("Pupil circle could not be visualized, values are: ", (int(inp_img_pupil_xyr[0].item()),int(inp_img_pupil_xyr[1].item()), int(inp_img_pupil_xyr[2].item())), ' ', inp_img_vis.shape)
                pass
            try:
                inp_img_vis = cv2.circle(inp_img_vis, (int(inp_img_iris_xyr[0].item()),int(inp_img_iris_xyr[1].item())), int(inp_img_iris_xyr[2].item()), (255, 0, 0), 2)
            except:
                print("Iris circle could not be visualized, values are: ", (int(inp_img_iris_xyr[0].item()),int(inp_img_iris_xyr[1].item()), int(inp_img_iris_xyr[2].item())), ' ', inp_img_vis.shape)
            
            Image.fromarray(inp_img_vis).save('./check2.png')
            '''
            
            sh_mult = self.input_size[0]/sh_crop
            sw_mult = self.input_size[1]/sw_crop
            inp_img = inp_img.resize((self.input_size[1], self.input_size[0]))
            inp_mask = inp_mask.resize((self.input_size[1], self.input_size[0]))
            inp_img_pupil_xyr[0] = self.input_size[1]/2 - sw_mult * (sw_crop/2 - inp_img_pupil_xyr[0])
            inp_img_pupil_xyr[1] = self.input_size[0]/2 - sh_mult * (sh_crop/2 - inp_img_pupil_xyr[1])
            inp_img_pupil_xyr[2] = max(sw_mult, sh_mult) * inp_img_pupil_xyr[2]
            inp_img_iris_xyr[0] = self.input_size[1]/2
            inp_img_iris_xyr[1] = self.input_size[0]/2
            inp_img_iris_xyr[2] = max(sw_mult, sh_mult) * inp_img_iris_xyr[2]
            
            bh_mult = self.input_size[0]/bh_crop
            bw_mult = self.input_size[1]/bw_crop
            tar_img = tar_img.resize((self.input_size[1], self.input_size[0]))
            tar_mask = tar_mask.resize((self.input_size[1], self.input_size[0]))
            tar_img_pupil_xyr[0] = self.input_size[1]/2 - bw_mult * (bw_crop/2 - tar_img_pupil_xyr[0])
            tar_img_pupil_xyr[1] = self.input_size[0]/2 - bh_mult * (bh_crop/2 - tar_img_pupil_xyr[1])
            tar_img_pupil_xyr[2] = max(bw_mult, bh_mult) * tar_img_pupil_xyr[2]
            tar_img_iris_xyr[0] = self.input_size[1]/2
            tar_img_iris_xyr[1] = self.input_size[0]/2
            tar_img_iris_xyr[2] = max(bw_mult, bh_mult) * tar_img_iris_xyr[2]
            
            ih_mult = self.input_size[0]/ih_crop
            iw_mult = self.input_size[1]/iw_crop
            imp_img = imp_img.resize((self.input_size[0], self.input_size[1]))
            imp_mask = imp_mask.resize((self.input_size[0], self.input_size[1]))
            imp_img_pupil_xyr[0] = self.input_size[1]/2 - iw_mult * (iw_crop/2 - imp_img_pupil_xyr[0])
            imp_img_pupil_xyr[1] = self.input_size[0]/2 - ih_mult * (ih_crop/2 - imp_img_pupil_xyr[1])
            imp_img_pupil_xyr[2] = max(iw_mult, ih_mult) * imp_img_pupil_xyr[2]
            imp_img_iris_xyr[0] = self.input_size[1]/2
            imp_img_iris_xyr[1] = self.input_size[0]/2
            imp_img_iris_xyr[2] = max(iw_mult, ih_mult) * imp_img_iris_xyr[2]
            
            #print('Saving ..')
            '''
            inp_img_vis = np.uint8(inp_img.convert('RGB'))
            try:
                inp_img_vis = cv2.circle(inp_img_vis, (int(inp_img_pupil_xyr[0].item()),int(inp_img_pupil_xyr[1].item())), int(inp_img_pupil_xyr[2].item()), (0, 0, 255), 2)
            except:
                print("Pupil circle could not be visualized, values are: ", (int(inp_img_pupil_xyr[0].item()),int(inp_img_pupil_xyr[1].item()), int(inp_img_pupil_xyr[2].item())), ' ', inp_img_vis.shape)
                pass
            try:
                inp_img_vis = cv2.circle(inp_img_vis, (int(inp_img_iris_xyr[0].item()),int(inp_img_iris_xyr[1].item())), int(inp_img_iris_xyr[2].item()), (255, 0, 0), 2)
            except:
                print("Iris circle could not be visualized, values are: ", (int(inp_img_iris_xyr[0].item()),int(inp_img_iris_xyr[1].item()), int(inp_img_iris_xyr[2].item())), ' ', inp_img_vis.shape)
            
            Image.fromarray(inp_img_vis).save('./check3.png')
            '''
        if random.random() < self.heavy_augment_prob:
            if random.random() < 0.5:
                random_choice = np.random.choice([1,2,3])
                if random_choice == 1:   
                    # sharpening
                    random_degree = np.random.choice([1,2,3,4,5])
                    if random_degree == 1:
                        inp_img = inp_img.filter(ImageFilter.DETAIL)
                    elif random_degree == 2:
                        inp_img = inp_img.filter(ImageFilter.SHARPEN)
                    elif random_degree == 3:
                        inp_img = inp_img.filter(ImageFilter.EDGE_ENHANCE)
                    elif random_degree == 4:
                        inp_img = inp_img.filter(ImageFilter.EDGE_ENHANCE_MORE)                            
                    else:
                        aug = iaa.Sharpen(alpha=(0.0, 0.3), lightness=(0.6, 1.0))
                        inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                        inp_img = Image.fromarray(inp_img_np[0])
                elif random_choice == 2:
                    #blurring
                    random_degree = np.random.choice([1,2,3,4,5])
                    if random_degree == 1:
                        inp_img = inp_img.filter(ImageFilter.GaussianBlur())
                    elif random_degree == 2:
                        inp_img = inp_img.filter(ImageFilter.BLUR)
                    elif random_degree == 3:
                        inp_img = inp_img.filter(ImageFilter.SMOOTH)
                    elif random_degree == 4:
                        inp_img = inp_img.filter(ImageFilter.SMOOTH_MORE)
                    else:
                        aug = iaa.MotionBlur(k=3)
                        inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                        inp_img = Image.fromarray(inp_img_np[0])
                else:
                    if random.random() < 0.5:
                        aug = iaa.AdditiveGaussianNoise(scale=3)
                        inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                        inp_img = Image.fromarray(inp_img_np[0])
                    else:
                        aug = iaa.AdditiveLaplaceNoise(scale=3)
                        inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                        inp_img = Image.fromarray(inp_img_np[0])
            if random.random() < 0.5:
                if random.random() < 0.5:
                    # Basic compression and expansion
                    divider = random.uniform(1.0, 1.5)
                    cw, ch = inp_img.size
                    new_cw = int(cw/divider)
                    new_ch = int(ch/divider)
                    
                    first_choice = np.random.choice([1,2,3,4,5,6])
                    if first_choice == 1:
                        inp_img = inp_img.resize((new_cw, new_ch), Image.NEAREST)
                    elif first_choice == 2:
                        inp_img = inp_img.resize((new_cw, new_ch), Image.BILINEAR)
                    elif first_choice == 3:
                        inp_img = inp_img.resize((new_cw, new_ch), Image.BICUBIC)
                    elif first_choice == 4:
                        inp_img = inp_img.resize((new_cw, new_ch), Image.LANCZOS)
                    elif first_choice == 5:
                        inp_img = inp_img.resize((new_cw, new_ch), Image.HAMMING)
                    else:
                        inp_img = inp_img.resize((new_cw, new_ch), Image.BOX)
                    
                    second_choice = np.random.choice([1,2,3,4,5,6])
                    if second_choice == 1:
                        inp_img = inp_img.resize((cw, ch), Image.NEAREST)
                    elif second_choice == 2:
                        inp_img = inp_img.resize((cw, ch), Image.BILINEAR)
                    elif second_choice == 3:
                        inp_img = inp_img.resize((cw, ch), Image.BICUBIC)
                    elif second_choice == 4:
                        inp_img = inp_img.resize((cw, ch), Image.LANCZOS)
                    elif second_choice == 5:
                        inp_img = inp_img.resize((cw, ch), Image.HAMMING)
                    else:
                        inp_img = inp_img.resize((cw, ch), Image.BOX)
                else:
                    #JPEG compression
                    aug = iaa.JpegCompression(compression=(10, 30))
                    inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                    inp_img = Image.fromarray(inp_img_np[0])
                         
            if random.random() < 0.5: # random contrast change
                random_degree = np.random.choice([1,2,3,4,5,6,7,8])                
                if random_degree == 1:
                    aug = iaa.GammaContrast((0.5, 2.0))
                    inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                    inp_img = Image.fromarray(inp_img_np[0])
                elif random_degree == 2:
                    aug = iaa.LinearContrast((0.4, 1.6))
                    inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                    inp_img = Image.fromarray(inp_img_np[0])
                elif random_degree == 3:
                    aug = iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6))
                    inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                    inp_img = Image.fromarray(inp_img_np[0])
                elif random_degree == 4:
                    aug = iaa.LogContrast(gain=(0.6, 1.4))
                    inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                    inp_img = Image.fromarray(inp_img_np[0])            
                else:    
                    aug = iaa.pillike.EnhanceBrightness()
                    inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                    inp_img = Image.fromarray(inp_img_np[0])
        '''         
        if random.random() < self.heavy_augment_prob:
            if random.random() < 0.5:
                random_choice = np.random.choice([1,2])
                if random_choice == 1:   
                    # sharpening
                    random_degree = np.random.choice([1,2,3,4,5])
                    if random_degree == 1:
                        tar_img = tar_img.filter(ImageFilter.DETAIL)
                    elif random_degree == 2:
                        tar_img = tar_img.filter(ImageFilter.SHARPEN)
                    elif random_degree == 3:
                        tar_img = tar_img.filter(ImageFilter.EDGE_ENHANCE)
                    elif random_degree == 4:
                        tar_img = tar_img.filter(ImageFilter.EDGE_ENHANCE_MORE)                            
                    else:
                        aug = iaa.Sharpen(alpha=(0.0, 0.3), lightness=(0.6, 1.0))
                        tar_img_np = aug(images = np.expand_dims(np.array(tar_img), axis=0))
                        tar_img = Image.fromarray(tar_img_np[0])
                else:
                    #blurring
                    random_degree = np.random.choice([1,2,3,4,5])
                    if random_degree == 1:
                        tar_img = tar_img.filter(ImageFilter.GaussianBlur())
                    elif random_degree == 2:
                        tar_img = tar_img.filter(ImageFilter.BLUR)
                    elif random_degree == 3:
                        tar_img = tar_img.filter(ImageFilter.SMOOTH)
                    elif random_degree == 4:
                        tar_img = tar_img.filter(ImageFilter.SMOOTH_MORE)
                    else:
                        aug = iaa.MotionBlur(k=3)
                        tar_img_np = aug(images = np.expand_dims(np.array(tar_img), axis=0))
                        tar_img = Image.fromarray(tar_img_np[0])
            if random.random() < 0.5:
                if random.random() < 0.5:
                    # Basic compression and expansion
                    divider = random.uniform(1.0, 1.5)
                    cw, ch = tar_img.size
                    new_cw = int(cw/divider)
                    new_ch = int(ch/divider)
                    
                    first_choice = np.random.choice([1,2,3,4,5,6])
                    if first_choice == 1:
                        tar_img = tar_img.resize((new_cw, new_ch), Image.NEAREST)
                    elif first_choice == 2:
                        tar_img = tar_img.resize((new_cw, new_ch), Image.BILINEAR)
                    elif first_choice == 3:
                        tar_img = tar_img.resize((new_cw, new_ch), Image.BICUBIC)
                    elif first_choice == 4:
                        tar_img = tar_img.resize((new_cw, new_ch), Image.LANCZOS)
                    elif first_choice == 5:
                        tar_img = tar_img.resize((new_cw, new_ch), Image.HAMMING)
                    else:
                        tar_img = tar_img.resize((new_cw, new_ch), Image.BOX)
                    
                    second_choice = np.random.choice([1,2,3,4,5,6])
                    if second_choice == 1:
                        tar_img = tar_img.resize((cw, ch), Image.NEAREST)
                    elif second_choice == 2:
                        tar_img = tar_img.resize((cw, ch), Image.BILINEAR)
                    elif second_choice == 3:
                        tar_img = tar_img.resize((cw, ch), Image.BICUBIC)
                    elif second_choice == 4:
                        tar_img = tar_img.resize((cw, ch), Image.LANCZOS)
                    elif second_choice == 5:
                        tar_img = tar_img.resize((cw, ch), Image.HAMMING)
                    else:
                        tar_img = tar_img.resize((cw, ch), Image.BOX)
                else:
                    #JPEG compression
                    aug = iaa.JpegCompression(compression=(10, 30))
                    tar_img_np = aug(images = np.expand_dims(np.array(tar_img), axis=0))
                    tar_img = Image.fromarray(tar_img_np[0])
                         
            if random.random() < 0.5: # random contrast change
                random_degree = np.random.choice([1,2,3,4,5,6,7,8])                
                if random_degree == 1:
                    aug = iaa.GammaContrast((0.5, 2.0))
                    tar_img_np = aug(images = np.expand_dims(np.array(tar_img), axis=0))
                    tar_img = Image.fromarray(tar_img_np[0])
                elif random_degree == 2:
                    aug = iaa.LinearContrast((0.4, 1.6))
                    tar_img_np = aug(images = np.expand_dims(np.array(tar_img), axis=0))
                    tar_img = Image.fromarray(tar_img_np[0])
                elif random_degree == 3:
                    aug = iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6))
                    tar_img_np = aug(images = np.expand_dims(np.array(tar_img), axis=0))
                    tar_img = Image.fromarray(tar_img_np[0])
                elif random_degree == 4:
                    aug = iaa.LogContrast(gain=(0.6, 1.4))
                    tar_img_np = aug(images = np.expand_dims(np.array(tar_img), axis=0))
                    tar_img = Image.fromarray(tar_img_np[0])            
                else:    
                    aug = iaa.pillike.EnhanceBrightness()
                    tar_img_np = aug(images = np.expand_dims(np.array(tar_img), axis=0))
                    tar_img = Image.fromarray(tar_img_np[0])
        
        if random.random() < self.heavy_augment_prob:
            if random.random() < 0.5:
                random_choice = np.random.choice([1,2,3])
                if random_choice == 1:   
                    # sharpening
                    random_degree = np.random.choice([1,2,3,4,5])
                    if random_degree == 1:
                        imp_img = imp_img.filter(ImageFilter.DETAIL)
                    elif random_degree == 2:
                        imp_img = imp_img.filter(ImageFilter.SHARPEN)
                    elif random_degree == 3:
                        imp_img = imp_img.filter(ImageFilter.EDGE_ENHANCE)
                    elif random_degree == 4:
                        imp_img = imp_img.filter(ImageFilter.EDGE_ENHANCE_MORE)                            
                    else:
                        aug = iaa.Sharpen(alpha=(0.0, 0.3), lightness=(0.6, 1.0))
                        imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                        imp_img = Image.fromarray(imp_img_np[0])
                elif random_choice == 2:
                    #blurring
                    random_degree = np.random.choice([1,2,3,4,5])
                    if random_degree == 1:
                        imp_img = imp_img.filter(ImageFilter.GaussianBlur())
                    elif random_degree == 2:
                        imp_img = imp_img.filter(ImageFilter.BLUR)
                    elif random_degree == 3:
                        imp_img = imp_img.filter(ImageFilter.SMOOTH)
                    elif random_degree == 4:
                        imp_img = imp_img.filter(ImageFilter.SMOOTH_MORE)
                    else:
                        aug = iaa.MotionBlur(k=3)
                        imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                        imp_img = Image.fromarray(imp_img_np[0])
                else:
                    if random.random() < 0.5:
                        aug = iaa.AdditiveGaussianNoise(scale=3)
                        imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                        imp_img = Image.fromarray(imp_img_np[0])
                    else:
                        aug = iaa.AdditiveLaplaceNoise(scale=3)
                        imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                        imp_img = Image.fromarray(imp_img_np[0])
            if random.random() < 0.5:
                if random.random() < 0.5:
                    # Basic compression and expansion
                    divider = random.uniform(1.0, 1.5)
                    cw, ch = imp_img.size
                    new_cw = int(cw/divider)
                    new_ch = int(ch/divider)
                    
                    first_choice = np.random.choice([1,2,3,4,5,6])
                    if first_choice == 1:
                        imp_img = imp_img.resize((new_cw, new_ch), Image.NEAREST)
                    elif first_choice == 2:
                        imp_img = imp_img.resize((new_cw, new_ch), Image.BILINEAR)
                    elif first_choice == 3:
                        imp_img = imp_img.resize((new_cw, new_ch), Image.BICUBIC)
                    elif first_choice == 4:
                        imp_img = imp_img.resize((new_cw, new_ch), Image.LANCZOS)
                    elif first_choice == 5:
                        imp_img = imp_img.resize((new_cw, new_ch), Image.HAMMING)
                    else:
                        imp_img = imp_img.resize((new_cw, new_ch), Image.BOX)
                    
                    second_choice = np.random.choice([1,2,3,4,5,6])
                    if second_choice == 1:
                        imp_img = imp_img.resize((cw, ch), Image.NEAREST)
                    elif second_choice == 2:
                        imp_img = imp_img.resize((cw, ch), Image.BILINEAR)
                    elif second_choice == 3:
                        imp_img = imp_img.resize((cw, ch), Image.BICUBIC)
                    elif second_choice == 4:
                        imp_img = imp_img.resize((cw, ch), Image.LANCZOS)
                    elif second_choice == 5:
                        imp_img = imp_img.resize((cw, ch), Image.HAMMING)
                    else:
                        imp_img = imp_img.resize((cw, ch), Image.BOX)
                else:
                    #JPEG compression
                    aug = iaa.JpegCompression(compression=(10, 30))
                    imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                    imp_img = Image.fromarray(imp_img_np[0])
                         
            if random.random() < 0.5: # random contrast change
                random_degree = np.random.choice([1,2,3,4,5,6,7,8])                
                if random_degree == 1:
                    aug = iaa.GammaContrast((0.5, 2.0))
                    imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                    imp_img = Image.fromarray(imp_img_np[0])
                elif random_degree == 2:
                    aug = iaa.LinearContrast((0.4, 1.6))
                    imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                    imp_img = Image.fromarray(imp_img_np[0])
                elif random_degree == 3:
                    aug = iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6))
                    imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                    imp_img = Image.fromarray(imp_img_np[0])
                elif random_degree == 4:
                    aug = iaa.LogContrast(gain=(0.6, 1.4))
                    imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                    imp_img = Image.fromarray(imp_img_np[0])            
                else:    
                    aug = iaa.pillike.EnhanceBrightness()
                    imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                    imp_img = Image.fromarray(imp_img_np[0])         
        '''
        if self.normalize:
            inp_img_t = torch.from_numpy(np.float32(inp_img)).float()
            inp_img_mean = torch.mean(inp_img_t)
            inp_img_std = torch.std(inp_img_t)
            xform_inp_img = (inp_img_t - inp_img_mean)/inp_img_std
            xform_inp_img = xform_inp_img.unsqueeze(0)
            
            xform_inp_mask = self.transform(inp_mask)
            xform_inp_mask[xform_inp_mask<0] = -1
            xform_inp_mask[xform_inp_mask>=0] = 1
            
            tar_img_t = torch.from_numpy(np.float32(tar_img)).float()
            tar_img_mean = torch.mean(tar_img_t)
            tar_img_std = torch.std(tar_img_t)
            xform_tar_img = (tar_img_t - tar_img_mean)/tar_img_std
            xform_tar_img = xform_tar_img.unsqueeze(0)
            
            xform_tar_mask = self.transform(tar_mask)
            xform_tar_mask[xform_tar_mask<0] = -1.0
            xform_tar_mask[xform_tar_mask>=0] = 1.0
            
            imp_img_t = torch.from_numpy(np.float32(imp_img)).float()
            imp_img_mean = torch.mean(imp_img_t)
            imp_img_std = torch.std(imp_img_t)
            xform_imp_img = (imp_img_t - imp_img_mean)/imp_img_std
            xform_imp_img = xform_imp_img.unsqueeze(0)
            
            xform_imp_mask = self.transform(imp_mask)
            xform_imp_mask[xform_imp_mask<0] = -1.0
            xform_imp_mask[xform_imp_mask>=0] = 1.0
            return {'inp_img': xform_inp_img, 'inp_img_mean': inp_img_mean, 'inp_img_std':inp_img_std, 'tar_img' : xform_tar_img, 'tar_img_mean':tar_img_mean, 'tar_img_std':tar_img_std, 'imp_img' : xform_imp_img, 'imp_img_mean': imp_img_mean, 'imp_img_std':imp_img_std, 'inp_mask': xform_inp_mask, 'tar_mask' : xform_tar_mask, 'imp_mask' : xform_imp_mask, 'inp_img_pxyr' : inp_img_pupil_xyr, 'tar_img_pxyr' : tar_img_pupil_xyr, 'imp_img_pxyr' : imp_img_pupil_xyr, 'inp_img_ixyr': inp_img_iris_xyr, 'tar_img_ixyr' :  tar_img_iris_xyr, 'imp_img_ixyr' : imp_img_iris_xyr, 'identifier' : triplet[6], 'imposter_id' : triplet[7]}

        else:
            xform_inp_img = self.transform(inp_img)
            xform_inp_mask = self.transform(inp_mask)
            xform_inp_mask[xform_inp_mask<0] = -1
            xform_inp_mask[xform_inp_mask>=0] = 1
            
            xform_tar_img = self.transform(tar_img)
            xform_tar_mask = self.transform(tar_mask)
            xform_tar_mask[xform_tar_mask<0] = -1.0
            xform_tar_mask[xform_tar_mask>=0] = 1.0
            
            xform_imp_img = self.transform(imp_img)
            xform_imp_mask = self.transform(imp_mask)
            xform_imp_mask[xform_imp_mask<0] = -1.0
            xform_imp_mask[xform_imp_mask>=0] = 1.0
            return {'inp_img': xform_inp_img, 'tar_img' : xform_tar_img, 'imp_img' : xform_imp_img, 'inp_mask': xform_inp_mask, 'tar_mask' : xform_tar_mask, 'imp_mask' : xform_imp_mask, 'inp_img_pxyr' : inp_img_pupil_xyr, 'tar_img_pxyr' : tar_img_pupil_xyr, 'imp_img_pxyr' : imp_img_pupil_xyr, 'inp_img_ixyr': inp_img_iris_xyr, 'tar_img_ixyr' :  tar_img_iris_xyr, 'imp_img_ixyr' : imp_img_iris_xyr, 'identifier' : triplet[6], 'imposter_id' : triplet[7]}

            
        #print(xform_inp_img.shape, xform_tar_img.shape,  xform_imp_img.shape)
        
        
    
    def __len__(self):
        return len(self.triplets)
    
    def reset(self):
        print('Shuffling triplets for WSD...')
        self.triplets = []
        self.genuine_ids = []
        self.imposter_ids = []
        wsd_identifiers = list(self.bins_wsd.keys())
        random.shuffle(wsd_identifiers)
        for identifier in tqdm(wsd_identifiers):
            wsd_identifiers_removed = wsd_identifiers.copy()
            wsd_identifiers_removed.remove(identifier)
            bin_numbers1 = list(self.bins_wsd[identifier].keys())
            bin_numbers2 = list(self.bins_wsd[identifier].keys())
            bin_numbers1.sort()
            bin_numbers2.sort()
            for bin_num_1 in bin_numbers1:
                for bin_num_2 in bin_numbers2:
                    if bin_num_1 != bin_num_2:
                        random.shuffle(self.bins_wsd[identifier][bin_num_1])
                        random.shuffle(self.bins_wsd[identifier][bin_num_2])
                        max_pairs = min(len(self.bins_wsd[identifier][bin_num_1]), len(self.bins_wsd[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            imposter_ind_max = 0
                            while imposter_ind_max == 0:
                                imposter_id = random.choice(wsd_identifiers_removed)
                                imposter_ind_max = len(self.bins_wsd[imposter_id][bin_num_2])
                            imagename1 = self.bins_wsd[identifier][bin_num_1][img_ind]
                            imagename2 = self.bins_wsd[identifier][bin_num_2][img_ind]
                            imagename3 = self.bins_wsd[imposter_id][bin_num_2][random.randrange(0, imposter_ind_max)]
                            indiv_id1 = imagename1.split('_')[0]
                            indiv_id2 = imagename2.split('_')[0]
                            indiv_id3 = imagename3.split('_')[0]
                            imagepath1 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id1), imagename1)
                            maskpath1 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id1), imagename1)
                            imagepath2 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id2), imagename2)
                            maskpath2 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id2), imagename2)
                            imagepath3 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id3), imagename3)
                            maskpath3 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id3), imagename3)
                            self.triplets.append([imagepath1, maskpath1, imagepath2, maskpath2, imagepath3, maskpath3, identifier, imposter_id])
                        random.shuffle(self.bins_wsd[identifier][bin_num_1])
                        random.shuffle(self.bins_wsd[identifier][bin_num_2])
                        for img_ind in range(max_pairs):
                            imposter_ind_max = 0
                            while imposter_ind_max == 0:
                                imposter_id = random.choice(wsd_identifiers_removed)
                                imposter_ind_max = len(self.bins_wsd[imposter_id][bin_num_1])
                            imagename1 = self.bins_wsd[identifier][bin_num_2][img_ind]
                            imagename2 = self.bins_wsd[identifier][bin_num_1][img_ind]
                            imagename3 = self.bins_wsd[imposter_id][bin_num_1][random.randrange(0, imposter_ind_max)]
                            indiv_id1 = imagename1.split('_')[0]
                            indiv_id2 = imagename2.split('_')[0]
                            indiv_id3 = imagename3.split('_')[0]
                            imagepath1 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id1), imagename1)
                            maskpath1 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id1), imagename1)
                            imagepath2 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id2), imagename2)
                            maskpath2 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id2), imagename2)
                            imagepath3 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id3), imagename3)
                            maskpath3 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id3), imagename3)
                            self.triplets.append([imagepath1, maskpath1, imagepath2, maskpath2, imagepath3, maskpath3, identifier, imposter_id])              
        
        print('Shuffling triplets for BXGRID...')
        bxgrid_identifiers = list(self.bins_bxgrid.keys())
        random.shuffle(bxgrid_identifiers)
        for identifier in tqdm(bxgrid_identifiers):
            bxgrid_identifiers_removed = bxgrid_identifiers.copy()
            bxgrid_identifiers_removed.remove(identifier)
            bin_numbers1 = list(self.bins_bxgrid[identifier].keys())
            bin_numbers2 = list(self.bins_bxgrid[identifier].keys())
            bin_numbers1.sort()
            bin_numbers2.sort()
            for bin_num_1 in bin_numbers1:
                for bin_num_2 in bin_numbers2:
                    if bin_num_1 != bin_num_2:
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_1])
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_2])
                        max_pairs = min(len(self.bins_bxgrid[identifier][bin_num_1]), len(self.bins_bxgrid[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            imposter_ind_max = 0
                            while imposter_ind_max == 0:
                                imposter_id = random.choice(bxgrid_identifiers_removed)
                                imposter_ind_max = len(self.bins_bxgrid[imposter_id][bin_num_2])
                            imagename1 = self.bins_bxgrid[identifier][bin_num_1][img_ind]
                            imagename2 = self.bins_bxgrid[identifier][bin_num_2][img_ind]
                            imagename3 = self.bins_bxgrid[imposter_id][bin_num_2][random.randrange(0, imposter_ind_max)]
                            imagepath1 = os.path.join(self.image_dir_bxgrid, imagename1)
                            maskpath1 = os.path.join(self.mask_dir_bxgrid, imagename1)
                            imagepath2 = os.path.join(self.image_dir_bxgrid, imagename2)
                            maskpath2 = os.path.join(self.mask_dir_bxgrid, imagename2)
                            imagepath3 = os.path.join(self.image_dir_bxgrid, imagename3)
                            maskpath3 = os.path.join(self.mask_dir_bxgrid, imagename3)
                            self.triplets.append([imagepath1, maskpath1, imagepath2, maskpath2, imagepath3, maskpath3, identifier, imposter_id])
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_1])
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_2])
                        for img_ind in range(max_pairs):
                            imposter_ind_max = 0
                            while imposter_ind_max == 0:
                                imposter_id = random.choice(bxgrid_identifiers_removed)
                                imposter_ind_max = len(self.bins_bxgrid[imposter_id][bin_num_1])
                            imagename1 = self.bins_bxgrid[identifier][bin_num_2][img_ind]
                            imagename2 = self.bins_bxgrid[identifier][bin_num_1][img_ind]
                            imagename3 = self.bins_bxgrid[imposter_id][bin_num_1][random.randrange(0, imposter_ind_max)]
                            imagepath1 = os.path.join(self.image_dir_bxgrid, imagename1)
                            maskpath1 = os.path.join(self.mask_dir_bxgrid, imagename1)
                            imagepath2 = os.path.join(self.image_dir_bxgrid, imagename2)
                            maskpath2 = os.path.join(self.mask_dir_bxgrid, imagename2)
                            imagepath3 = os.path.join(self.image_dir_bxgrid, imagename3)
                            maskpath3 = os.path.join(self.mask_dir_bxgrid, imagename3)
                            self.triplets.append([imagepath1, maskpath1, imagepath2, maskpath2, imagepath3, maskpath3, identifier, imposter_id])
                            
        print('Total triplets: ', len(self.triplets))

class TripletFromBinAllDatasetBothPolarized(Dataset):
    def __init__(self, bins_path_wsd, parent_dir_wsd, bins_path_bxgrid, parent_dir_bxgrid, iris_crop=True, normalize=True, brightness_and_contrast = False, flip_data = True, res_mult=1):
        super().__init__()
        self.iris_crop = iris_crop
        self.flip_data = flip_data
        self.b_n_c = brightness_and_contrast
        self.input_size=(int(256*res_mult),int(256*res_mult))
        self.res_mult = res_mult
        self.normalize = normalize
        
        self.image_dir_wsd = os.path.join(parent_dir_wsd, 'WarsawDataPolar')
        self.mask_dir_wsd = os.path.join(parent_dir_wsd, 'WarsawMaskPolar')            
        
        with open(bins_path_wsd, 'rb') as wsdbinsFile:
            self.bins_wsd = pkl.load(wsdbinsFile)
        
        self.image_dir_bxgrid = os.path.join(parent_dir_bxgrid, 'images_polar')
        self.mask_dir_bxgrid = os.path.join(parent_dir_bxgrid, 'masks_polar')       
        
        with open(bins_path_bxgrid, 'rb') as bxgridbinsFile:
            self.bins_bxgrid = pkl.load(bxgridbinsFile)
        
        self.all_ids = list(self.bins_wsd.keys()) + list(self.bins_bxgrid.keys())
        
        print('Initializing triplets for Warsaw Pupil Dynamics...')
        self.triplets = []
        self.genuine_ids = []
        self.imposter_ids = []
        wsd_identifiers = list(self.bins_wsd.keys())
        random.shuffle(wsd_identifiers)
        for identifier in tqdm(wsd_identifiers):
            wsd_identifiers_removed = wsd_identifiers.copy()
            wsd_identifiers_removed.remove(identifier)
            bin_numbers1 = list(self.bins_wsd[identifier].keys())
            bin_numbers2 = list(self.bins_wsd[identifier].keys())
            bin_numbers1.sort()
            bin_numbers2.sort()
            for bin_num_1 in bin_numbers1:
                for bin_num_2 in bin_numbers2:
                    if bin_num_1 != bin_num_2:
                        random.shuffle(self.bins_wsd[identifier][bin_num_1])
                        random.shuffle(self.bins_wsd[identifier][bin_num_2])
                        max_pairs = min(len(self.bins_wsd[identifier][bin_num_1]), len(self.bins_wsd[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            imposter_ind_max = 0
                            while imposter_ind_max == 0:
                                imposter_id = random.choice(wsd_identifiers_removed)
                                imposter_ind_max = len(self.bins_wsd[imposter_id][bin_num_2])
                            imagename1 = self.bins_wsd[identifier][bin_num_1][img_ind]
                            imagename2 = self.bins_wsd[identifier][bin_num_2][img_ind]
                            imagename3 = self.bins_wsd[imposter_id][bin_num_2][random.randrange(0, imposter_ind_max)]
                            indiv_id1 = imagename1.split('_')[0]
                            indiv_id2 = imagename2.split('_')[0]
                            indiv_id3 = imagename3.split('_')[0]
                            imagepath1 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id1), imagename1)
                            maskpath1 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id1), imagename1)
                            imagepath2 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id2), imagename2)
                            maskpath2 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id2), imagename2)
                            imagepath3 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id3), imagename3)
                            maskpath3 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id3), imagename3)
                            self.triplets.append([imagepath1, maskpath1, imagepath2, maskpath2, imagepath3, maskpath3, identifier, imposter_id])
                        random.shuffle(self.bins_wsd[identifier][bin_num_1])
                        random.shuffle(self.bins_wsd[identifier][bin_num_2])
                        for img_ind in range(max_pairs):
                            imposter_ind_max = 0
                            while imposter_ind_max == 0:
                                imposter_id = random.choice(wsd_identifiers_removed)
                                imposter_ind_max = len(self.bins_wsd[imposter_id][bin_num_1])
                            imagename1 = self.bins_wsd[identifier][bin_num_2][img_ind]
                            imagename2 = self.bins_wsd[identifier][bin_num_1][img_ind]
                            imagename3 = self.bins_wsd[imposter_id][bin_num_1][random.randrange(0, imposter_ind_max)]
                            indiv_id1 = imagename1.split('_')[0]
                            indiv_id2 = imagename2.split('_')[0]
                            indiv_id3 = imagename3.split('_')[0]
                            imagepath1 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id1), imagename1)
                            maskpath1 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id1), imagename1)
                            imagepath2 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id2), imagename2)
                            maskpath2 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id2), imagename2)
                            imagepath3 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id3), imagename3)
                            maskpath3 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id3), imagename3)
                            self.triplets.append([imagepath1, maskpath1, imagepath2, maskpath2, imagepath3, maskpath3, identifier, imposter_id])              
        
        print('Initializing triplets for BXGRID...')
        bxgrid_identifiers = list(self.bins_bxgrid.keys())
        random.shuffle(bxgrid_identifiers)
        for identifier in tqdm(bxgrid_identifiers):
            bxgrid_identifiers_removed = bxgrid_identifiers.copy()
            bxgrid_identifiers_removed.remove(identifier)
            bin_numbers1 = list(self.bins_bxgrid[identifier].keys())
            bin_numbers2 = list(self.bins_bxgrid[identifier].keys())
            bin_numbers1.sort()
            bin_numbers2.sort()
            for bin_num_1 in bin_numbers1:
                for bin_num_2 in bin_numbers2:
                    if bin_num_1 != bin_num_2:
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_1])
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_2])
                        max_pairs = min(len(self.bins_bxgrid[identifier][bin_num_1]), len(self.bins_bxgrid[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            imposter_ind_max = 0
                            while imposter_ind_max == 0:
                                imposter_id = random.choice(bxgrid_identifiers_removed)
                                imposter_ind_max = len(self.bins_bxgrid[imposter_id][bin_num_2])
                            imagename1 = self.bins_bxgrid[identifier][bin_num_1][img_ind]
                            imagename2 = self.bins_bxgrid[identifier][bin_num_2][img_ind]
                            imagename3 = self.bins_bxgrid[imposter_id][bin_num_2][random.randrange(0, imposter_ind_max)]
                            imagepath1 = os.path.join(self.image_dir_bxgrid, imagename1)
                            maskpath1 = os.path.join(self.mask_dir_bxgrid, imagename1)
                            imagepath2 = os.path.join(self.image_dir_bxgrid, imagename2)
                            maskpath2 = os.path.join(self.mask_dir_bxgrid, imagename2)
                            imagepath3 = os.path.join(self.image_dir_bxgrid, imagename3)
                            maskpath3 = os.path.join(self.mask_dir_bxgrid, imagename3)
                            self.triplets.append([imagepath1, maskpath1, imagepath2, maskpath2, imagepath3, maskpath3, identifier, imposter_id])
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_1])
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_2])
                        for img_ind in range(max_pairs):
                            imposter_ind_max = 0
                            while imposter_ind_max == 0:
                                imposter_id = random.choice(bxgrid_identifiers_removed)
                                imposter_ind_max = len(self.bins_bxgrid[imposter_id][bin_num_1])
                            imagename1 = self.bins_bxgrid[identifier][bin_num_2][img_ind]
                            imagename2 = self.bins_bxgrid[identifier][bin_num_1][img_ind]
                            imagename3 = self.bins_bxgrid[imposter_id][bin_num_1][random.randrange(0, imposter_ind_max)]
                            imagepath1 = os.path.join(self.image_dir_bxgrid, imagename1)
                            maskpath1 = os.path.join(self.mask_dir_bxgrid, imagename1)
                            imagepath2 = os.path.join(self.image_dir_bxgrid, imagename2)
                            maskpath2 = os.path.join(self.mask_dir_bxgrid, imagename2)
                            imagepath3 = os.path.join(self.image_dir_bxgrid, imagename3)
                            maskpath3 = os.path.join(self.mask_dir_bxgrid, imagename3)
                            self.triplets.append([imagepath1, maskpath1, imagepath2, maskpath2, imagepath3, maskpath3, identifier, imposter_id])
                            
        print('Total triplets: ', len(self.triplets))
        

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5, ))
        ])
        
        self.pupil_iris_xyrs_path_wsd = os.path.join(parent_dir_wsd, 'pupil_iris_xyrs_new.pkl')
        with open(self.pupil_iris_xyrs_path_wsd, 'rb') as pupilirisxyrsFile:
            self.pupil_iris_xyrs_wsd = pkl.load(pupilirisxyrsFile)
        
        self.pupil_iris_xyrs_path_bxgrid = os.path.join(parent_dir_bxgrid, 'pupil_iris_xyrs_new.pkl')
        with open(self.pupil_iris_xyrs_path_bxgrid, 'rb') as pupilirisxyrsFile:
            self.pupil_iris_xyrs_bxgrid = pkl.load(pupilirisxyrsFile)
        
        self.pupil_xyrs = {}
        self.iris_xyrs = {}
        
        for imagename in self.pupil_iris_xyrs_wsd.keys():
            indiv_id = imagename.split('_')[0]
            imagepath = os.path.join(os.path.join(self.image_dir_wsd, indiv_id), imagename)
            self.pupil_xyrs[imagepath] = self.pupil_iris_xyrs_wsd[imagename]['pxyr']
            self.iris_xyrs[imagepath] = self.pupil_iris_xyrs_wsd[imagename]['ixyr']
        
        for imagename in self.pupil_iris_xyrs_bxgrid.keys():
            imagepath = os.path.join(self.image_dir_bxgrid, imagename)
            self.pupil_xyrs[imagepath] = self.pupil_iris_xyrs_bxgrid[imagename]['pxyr']
            self.iris_xyrs[imagepath] = self.pupil_iris_xyrs_bxgrid[imagename]['ixyr']
        
        self.heavy_augment_prob = 0.3
    
    def load_image(self, file):
        return Image.open(file).convert('L')
        
    def __getitem__(self, index):
    
        triplet = self.triplets[index]
        
        inp_img_path = triplet[0].strip()
        inp_mask_path = triplet[1].strip()
        tar_img_path = triplet[2].strip()
        tar_mask_path = triplet[3].strip()
        imp_img_path = triplet[4].strip()
        imp_mask_path = triplet[5].strip()
        
        inp_img = self.load_image(inp_img_path)
        inp_mask = self.load_image(inp_mask_path)
        
        tar_img = self.load_image(tar_img_path)
        tar_mask = self.load_image(tar_mask_path)
        
        imp_img = self.load_image(imp_img_path)
        imp_mask = self.load_image(imp_mask_path)
        
        inp_img_pupil_xyr = torch.FloatTensor(self.pupil_xyrs[inp_img_path])
        inp_img_iris_xyr = torch.FloatTensor(self.iris_xyrs[inp_img_path])
        
        tar_img_pupil_xyr = torch.FloatTensor(self.pupil_xyrs[tar_img_path])
        tar_img_iris_xyr = torch.FloatTensor(self.iris_xyrs[tar_img_path])
        
        imp_img_pupil_xyr = torch.FloatTensor(self.pupil_xyrs[imp_img_path])
        imp_img_iris_xyr = torch.FloatTensor(self.iris_xyrs[imp_img_path])
        
        '''
        sw, sh = inp_img.size
        sw_mult = (sw/320)
        sh_mult = (self.input_size[0]/sh)
        
        bw, bh = tar_img.size
        bw_mult = (self.input_size[1]/bw)
        bh_mult = (self.input_size[0]/bh)
        
        biw, bih = imp_img.size
        biw_mult = (self.input_size[1]/biw)
        bih_mult = (self.input_size[0]/bih)
        
        inp_img_pupil_xyr[0] = (inp_img_pupil_xyr[0] * sw_mult)
        inp_img_pupil_xyr[1] = (inp_img_pupil_xyr[1] * sh_mult)
        inp_img_pupil_xyr[2] = (inp_img_pupil_xyr[2] * max(sw_mult, sh_mult))
        
        tar_img_pupil_xyr[0] = (tar_img_pupil_xyr[0] * bw_mult)
        tar_img_pupil_xyr[1] = (tar_img_pupil_xyr[1] * bh_mult)
        tar_img_pupil_xyr[2] = (tar_img_pupil_xyr[2] * max(bw_mult, bh_mult))
        
        imp_img_pupil_xyr[0] = (imp_img_pupil_xyr[0] * biw_mult)
        imp_img_pupil_xyr[1] = (imp_img_pupil_xyr[1] * bih_mult)
        imp_img_pupil_xyr[2] = (imp_img_pupil_xyr[2] * max(biw_mult, bih_mult))
    
        inp_img_iris_xyr[0] = (inp_img_iris_xyr[0] * sw_mult)
        inp_img_iris_xyr[1] = (inp_img_iris_xyr[1] * sh_mult)
        inp_img_iris_xyr[2] = (inp_img_iris_xyr[2] * max(sw_mult, sh_mult))
        
        tar_img_iris_xyr[0] = (tar_img_iris_xyr[0] * bw_mult)
        tar_img_iris_xyr[1] = (tar_img_iris_xyr[1] * bh_mult)
        tar_img_iris_xyr[2] = (tar_img_iris_xyr[2] * max(bw_mult, bh_mult))
        
        imp_img_iris_xyr[0] = (imp_img_iris_xyr[0] * biw_mult)
        imp_img_iris_xyr[1] = (imp_img_iris_xyr[1] * bih_mult)
        imp_img_iris_xyr[2] = (imp_img_iris_xyr[2] * max(biw_mult, bih_mult))
        '''
        sw, sh = inp_img.size
        bw, bh = tar_img.size
        iw, ih = imp_img.size
        
        if self.iris_crop:
            #inp_img.save('./check1.png')
            #print((inp_img_iris_xyr[0].item() - inp_img_iris_xyr[2].item(), inp_img_iris_xyr[1].item() - inp_img_iris_xyr[2].item(), inp_img_iris_xyr[0].item() + inp_img_iris_xyr[2].item(), inp_img_iris_xyr[1].item() + inp_img_iris_xyr[2].item()))
            inp_img = inp_img.crop((inp_img_iris_xyr[0].item() - inp_img_iris_xyr[2].item(), inp_img_iris_xyr[1].item() - inp_img_iris_xyr[2].item(), inp_img_iris_xyr[0].item() + inp_img_iris_xyr[2].item(), inp_img_iris_xyr[1].item() + inp_img_iris_xyr[2].item()))
            inp_mask = inp_mask.crop((inp_img_iris_xyr[0].item() - inp_img_iris_xyr[2].item(), inp_img_iris_xyr[1].item() - inp_img_iris_xyr[2].item(), inp_img_iris_xyr[0].item() + inp_img_iris_xyr[2].item(), inp_img_iris_xyr[1].item() + inp_img_iris_xyr[2].item()))
            sw_crop, sh_crop = inp_img.size
            inp_img_pupil_xyr[0] = inp_img_pupil_xyr[0] - inp_img_iris_xyr[0] + sw_crop/2
            inp_img_pupil_xyr[1] = inp_img_pupil_xyr[1] - inp_img_iris_xyr[1] + sh_crop/2
            inp_img_iris_xyr[0] = sw_crop/2
            inp_img_iris_xyr[1] = sh_crop/2            
            
            tar_img = tar_img.crop((tar_img_iris_xyr[0].item() - tar_img_iris_xyr[2].item(), tar_img_iris_xyr[1].item() - tar_img_iris_xyr[2].item(), tar_img_iris_xyr[0].item() + tar_img_iris_xyr[2].item(), tar_img_iris_xyr[1].item() + tar_img_iris_xyr[2].item()))
            tar_mask = tar_mask.crop((tar_img_iris_xyr[0].item() - tar_img_iris_xyr[2].item(), tar_img_iris_xyr[1].item() - tar_img_iris_xyr[2].item(), tar_img_iris_xyr[0].item() + tar_img_iris_xyr[2].item(), tar_img_iris_xyr[1].item() + tar_img_iris_xyr[2].item()))
            bw_crop, bh_crop = tar_img.size
            tar_img_pupil_xyr[0] = tar_img_pupil_xyr[0] - tar_img_iris_xyr[0] + bw_crop/2
            tar_img_pupil_xyr[1] = tar_img_pupil_xyr[1] - tar_img_iris_xyr[1] + bh_crop/2
            tar_img_iris_xyr[0] = bw_crop/2
            tar_img_iris_xyr[1] = bh_crop/2
            
            imp_img = imp_img.crop((imp_img_iris_xyr[0].item() - imp_img_iris_xyr[2].item(), imp_img_iris_xyr[1].item() - imp_img_iris_xyr[2].item(), imp_img_iris_xyr[0].item() + imp_img_iris_xyr[2].item(), imp_img_iris_xyr[1].item() + imp_img_iris_xyr[2].item()))
            imp_mask = imp_mask.crop((imp_img_iris_xyr[0].item() - imp_img_iris_xyr[2].item(), imp_img_iris_xyr[1].item() - imp_img_iris_xyr[2].item(), imp_img_iris_xyr[0].item() + imp_img_iris_xyr[2].item(), imp_img_iris_xyr[1].item() + imp_img_iris_xyr[2].item()))
            iw_crop, ih_crop = imp_img.size
            imp_img_pupil_xyr[0] = imp_img_pupil_xyr[0] - imp_img_iris_xyr[0] + iw_crop/2
            imp_img_pupil_xyr[1] = imp_img_pupil_xyr[1] - imp_img_iris_xyr[1] + ih_crop/2
            imp_img_iris_xyr[0] = iw_crop/2
            imp_img_iris_xyr[1] = ih_crop/2
            
            '''
            inp_img_vis = np.uint8(inp_img.convert('RGB'))
            try:
                inp_img_vis = cv2.circle(inp_img_vis, (int(inp_img_pupil_xyr[0].item()),int(inp_img_pupil_xyr[1].item())), int(inp_img_pupil_xyr[2].item()), (0, 0, 255), 2)
            except:
                print("Pupil circle could not be visualized, values are: ", (int(inp_img_pupil_xyr[0].item()),int(inp_img_pupil_xyr[1].item()), int(inp_img_pupil_xyr[2].item())), ' ', inp_img_vis.shape)
                pass
            try:
                inp_img_vis = cv2.circle(inp_img_vis, (int(inp_img_iris_xyr[0].item()),int(inp_img_iris_xyr[1].item())), int(inp_img_iris_xyr[2].item()), (255, 0, 0), 2)
            except:
                print("Iris circle could not be visualized, values are: ", (int(inp_img_iris_xyr[0].item()),int(inp_img_iris_xyr[1].item()), int(inp_img_iris_xyr[2].item())), ' ', inp_img_vis.shape)
            
            Image.fromarray(inp_img_vis).save('./check2.png')
            '''
            
            sh_mult = self.input_size[0]/sh_crop
            sw_mult = self.input_size[1]/sw_crop
            inp_img = inp_img.resize((self.input_size[1], self.input_size[0]))
            inp_mask = inp_mask.resize((self.input_size[1], self.input_size[0]))
            inp_img_pupil_xyr[0] = self.input_size[1]/2 - sw_mult * (sw_crop/2 - inp_img_pupil_xyr[0])
            inp_img_pupil_xyr[1] = self.input_size[0]/2 - sh_mult * (sh_crop/2 - inp_img_pupil_xyr[1])
            inp_img_pupil_xyr[2] = max(sw_mult, sh_mult) * inp_img_pupil_xyr[2]
            inp_img_iris_xyr[0] = self.input_size[1]/2
            inp_img_iris_xyr[1] = self.input_size[0]/2
            inp_img_iris_xyr[2] = max(sw_mult, sh_mult) * inp_img_iris_xyr[2]
            
            bh_mult = self.input_size[0]/bh_crop
            bw_mult = self.input_size[1]/bw_crop
            tar_img = tar_img.resize((self.input_size[1], self.input_size[0]))
            tar_mask = tar_mask.resize((self.input_size[1], self.input_size[0]))
            tar_img_pupil_xyr[0] = self.input_size[1]/2 - bw_mult * (bw_crop/2 - tar_img_pupil_xyr[0])
            tar_img_pupil_xyr[1] = self.input_size[0]/2 - bh_mult * (bh_crop/2 - tar_img_pupil_xyr[1])
            tar_img_pupil_xyr[2] = max(bw_mult, bh_mult) * tar_img_pupil_xyr[2]
            tar_img_iris_xyr[0] = self.input_size[1]/2
            tar_img_iris_xyr[1] = self.input_size[0]/2
            tar_img_iris_xyr[2] = max(bw_mult, bh_mult) * tar_img_iris_xyr[2]
            
            ih_mult = self.input_size[0]/ih_crop
            iw_mult = self.input_size[1]/iw_crop
            imp_img = imp_img.resize((self.input_size[0], self.input_size[1]))
            imp_mask = imp_mask.resize((self.input_size[0], self.input_size[1]))
            imp_img_pupil_xyr[0] = self.input_size[1]/2 - iw_mult * (iw_crop/2 - imp_img_pupil_xyr[0])
            imp_img_pupil_xyr[1] = self.input_size[0]/2 - ih_mult * (ih_crop/2 - imp_img_pupil_xyr[1])
            imp_img_pupil_xyr[2] = max(iw_mult, ih_mult) * imp_img_pupil_xyr[2]
            imp_img_iris_xyr[0] = self.input_size[1]/2
            imp_img_iris_xyr[1] = self.input_size[0]/2
            imp_img_iris_xyr[2] = max(iw_mult, ih_mult) * imp_img_iris_xyr[2]
            
            #print('Saving ..')
            '''
            inp_img_vis = np.uint8(inp_img.convert('RGB'))
            try:
                inp_img_vis = cv2.circle(inp_img_vis, (int(inp_img_pupil_xyr[0].item()),int(inp_img_pupil_xyr[1].item())), int(inp_img_pupil_xyr[2].item()), (0, 0, 255), 2)
            except:
                print("Pupil circle could not be visualized, values are: ", (int(inp_img_pupil_xyr[0].item()),int(inp_img_pupil_xyr[1].item()), int(inp_img_pupil_xyr[2].item())), ' ', inp_img_vis.shape)
                pass
            try:
                inp_img_vis = cv2.circle(inp_img_vis, (int(inp_img_iris_xyr[0].item()),int(inp_img_iris_xyr[1].item())), int(inp_img_iris_xyr[2].item()), (255, 0, 0), 2)
            except:
                print("Iris circle could not be visualized, values are: ", (int(inp_img_iris_xyr[0].item()),int(inp_img_iris_xyr[1].item()), int(inp_img_iris_xyr[2].item())), ' ', inp_img_vis.shape)
            
            Image.fromarray(inp_img_vis).save('./check3.png')
            '''
        if random.random() < self.heavy_augment_prob:
            if random.random() < 0.5:
                random_choice = np.random.choice([1,2,3])
                if random_choice == 1:   
                    # sharpening
                    random_degree = np.random.choice([1,2,3,4,5])
                    if random_degree == 1:
                        inp_img = inp_img.filter(ImageFilter.DETAIL)
                    elif random_degree == 2:
                        inp_img = inp_img.filter(ImageFilter.SHARPEN)
                    elif random_degree == 3:
                        inp_img = inp_img.filter(ImageFilter.EDGE_ENHANCE)
                    elif random_degree == 4:
                        inp_img = inp_img.filter(ImageFilter.EDGE_ENHANCE_MORE)                            
                    else:
                        aug = iaa.Sharpen(alpha=(0.0, 0.3), lightness=(0.6, 1.0))
                        inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                        inp_img = Image.fromarray(inp_img_np[0])
                elif random_choice == 2:
                    #blurring
                    random_degree = np.random.choice([1,2,3,4,5])
                    if random_degree == 1:
                        inp_img = inp_img.filter(ImageFilter.GaussianBlur())
                    elif random_degree == 2:
                        inp_img = inp_img.filter(ImageFilter.BLUR)
                    elif random_degree == 3:
                        inp_img = inp_img.filter(ImageFilter.SMOOTH)
                    elif random_degree == 4:
                        inp_img = inp_img.filter(ImageFilter.SMOOTH_MORE)
                    else:
                        aug = iaa.MotionBlur(k=3)
                        inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                        inp_img = Image.fromarray(inp_img_np[0])
                else:
                    if random.random() < 0.5:
                        aug = iaa.AdditiveGaussianNoise(scale=3)
                        inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                        inp_img = Image.fromarray(inp_img_np[0])
                    else:
                        aug = iaa.AdditiveLaplaceNoise(scale=3)
                        inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                        inp_img = Image.fromarray(inp_img_np[0])
            if random.random() < 0.5:
                if random.random() < 0.5:
                    # Basic compression and expansion
                    divider = random.uniform(1.0, 1.5)
                    cw, ch = inp_img.size
                    new_cw = int(cw/divider)
                    new_ch = int(ch/divider)
                    
                    first_choice = np.random.choice([1,2,3,4,5,6])
                    if first_choice == 1:
                        inp_img = inp_img.resize((new_cw, new_ch), Image.NEAREST)
                    elif first_choice == 2:
                        inp_img = inp_img.resize((new_cw, new_ch), Image.BILINEAR)
                    elif first_choice == 3:
                        inp_img = inp_img.resize((new_cw, new_ch), Image.BICUBIC)
                    elif first_choice == 4:
                        inp_img = inp_img.resize((new_cw, new_ch), Image.LANCZOS)
                    elif first_choice == 5:
                        inp_img = inp_img.resize((new_cw, new_ch), Image.HAMMING)
                    else:
                        inp_img = inp_img.resize((new_cw, new_ch), Image.BOX)
                    
                    second_choice = np.random.choice([1,2,3,4,5,6])
                    if second_choice == 1:
                        inp_img = inp_img.resize((cw, ch), Image.NEAREST)
                    elif second_choice == 2:
                        inp_img = inp_img.resize((cw, ch), Image.BILINEAR)
                    elif second_choice == 3:
                        inp_img = inp_img.resize((cw, ch), Image.BICUBIC)
                    elif second_choice == 4:
                        inp_img = inp_img.resize((cw, ch), Image.LANCZOS)
                    elif second_choice == 5:
                        inp_img = inp_img.resize((cw, ch), Image.HAMMING)
                    else:
                        inp_img = inp_img.resize((cw, ch), Image.BOX)
                else:
                    #JPEG compression
                    aug = iaa.JpegCompression(compression=(10, 30))
                    inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                    inp_img = Image.fromarray(inp_img_np[0])
                         
            if random.random() < 0.5: # random contrast change
                random_degree = np.random.choice([1,2,3,4,5,6,7,8])                
                if random_degree == 1:
                    aug = iaa.GammaContrast((0.5, 2.0))
                    inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                    inp_img = Image.fromarray(inp_img_np[0])
                elif random_degree == 2:
                    aug = iaa.LinearContrast((0.4, 1.6))
                    inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                    inp_img = Image.fromarray(inp_img_np[0])
                elif random_degree == 3:
                    aug = iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6))
                    inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                    inp_img = Image.fromarray(inp_img_np[0])
                elif random_degree == 4:
                    aug = iaa.LogContrast(gain=(0.6, 1.4))
                    inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                    inp_img = Image.fromarray(inp_img_np[0])            
                else:    
                    aug = iaa.pillike.EnhanceBrightness()
                    inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                    inp_img = Image.fromarray(inp_img_np[0])
        '''         
        if random.random() < self.heavy_augment_prob:
            if random.random() < 0.5:
                random_choice = np.random.choice([1,2])
                if random_choice == 1:   
                    # sharpening
                    random_degree = np.random.choice([1,2,3,4,5])
                    if random_degree == 1:
                        tar_img = tar_img.filter(ImageFilter.DETAIL)
                    elif random_degree == 2:
                        tar_img = tar_img.filter(ImageFilter.SHARPEN)
                    elif random_degree == 3:
                        tar_img = tar_img.filter(ImageFilter.EDGE_ENHANCE)
                    elif random_degree == 4:
                        tar_img = tar_img.filter(ImageFilter.EDGE_ENHANCE_MORE)                            
                    else:
                        aug = iaa.Sharpen(alpha=(0.0, 0.3), lightness=(0.6, 1.0))
                        tar_img_np = aug(images = np.expand_dims(np.array(tar_img), axis=0))
                        tar_img = Image.fromarray(tar_img_np[0])
                else:
                    #blurring
                    random_degree = np.random.choice([1,2,3,4,5])
                    if random_degree == 1:
                        tar_img = tar_img.filter(ImageFilter.GaussianBlur())
                    elif random_degree == 2:
                        tar_img = tar_img.filter(ImageFilter.BLUR)
                    elif random_degree == 3:
                        tar_img = tar_img.filter(ImageFilter.SMOOTH)
                    elif random_degree == 4:
                        tar_img = tar_img.filter(ImageFilter.SMOOTH_MORE)
                    else:
                        aug = iaa.MotionBlur(k=3)
                        tar_img_np = aug(images = np.expand_dims(np.array(tar_img), axis=0))
                        tar_img = Image.fromarray(tar_img_np[0])
            if random.random() < 0.5:
                if random.random() < 0.5:
                    # Basic compression and expansion
                    divider = random.uniform(1.0, 1.5)
                    cw, ch = tar_img.size
                    new_cw = int(cw/divider)
                    new_ch = int(ch/divider)
                    
                    first_choice = np.random.choice([1,2,3,4,5,6])
                    if first_choice == 1:
                        tar_img = tar_img.resize((new_cw, new_ch), Image.NEAREST)
                    elif first_choice == 2:
                        tar_img = tar_img.resize((new_cw, new_ch), Image.BILINEAR)
                    elif first_choice == 3:
                        tar_img = tar_img.resize((new_cw, new_ch), Image.BICUBIC)
                    elif first_choice == 4:
                        tar_img = tar_img.resize((new_cw, new_ch), Image.LANCZOS)
                    elif first_choice == 5:
                        tar_img = tar_img.resize((new_cw, new_ch), Image.HAMMING)
                    else:
                        tar_img = tar_img.resize((new_cw, new_ch), Image.BOX)
                    
                    second_choice = np.random.choice([1,2,3,4,5,6])
                    if second_choice == 1:
                        tar_img = tar_img.resize((cw, ch), Image.NEAREST)
                    elif second_choice == 2:
                        tar_img = tar_img.resize((cw, ch), Image.BILINEAR)
                    elif second_choice == 3:
                        tar_img = tar_img.resize((cw, ch), Image.BICUBIC)
                    elif second_choice == 4:
                        tar_img = tar_img.resize((cw, ch), Image.LANCZOS)
                    elif second_choice == 5:
                        tar_img = tar_img.resize((cw, ch), Image.HAMMING)
                    else:
                        tar_img = tar_img.resize((cw, ch), Image.BOX)
                else:
                    #JPEG compression
                    aug = iaa.JpegCompression(compression=(10, 30))
                    tar_img_np = aug(images = np.expand_dims(np.array(tar_img), axis=0))
                    tar_img = Image.fromarray(tar_img_np[0])
                         
            if random.random() < 0.5: # random contrast change
                random_degree = np.random.choice([1,2,3,4,5,6,7,8])                
                if random_degree == 1:
                    aug = iaa.GammaContrast((0.5, 2.0))
                    tar_img_np = aug(images = np.expand_dims(np.array(tar_img), axis=0))
                    tar_img = Image.fromarray(tar_img_np[0])
                elif random_degree == 2:
                    aug = iaa.LinearContrast((0.4, 1.6))
                    tar_img_np = aug(images = np.expand_dims(np.array(tar_img), axis=0))
                    tar_img = Image.fromarray(tar_img_np[0])
                elif random_degree == 3:
                    aug = iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6))
                    tar_img_np = aug(images = np.expand_dims(np.array(tar_img), axis=0))
                    tar_img = Image.fromarray(tar_img_np[0])
                elif random_degree == 4:
                    aug = iaa.LogContrast(gain=(0.6, 1.4))
                    tar_img_np = aug(images = np.expand_dims(np.array(tar_img), axis=0))
                    tar_img = Image.fromarray(tar_img_np[0])            
                else:    
                    aug = iaa.pillike.EnhanceBrightness()
                    tar_img_np = aug(images = np.expand_dims(np.array(tar_img), axis=0))
                    tar_img = Image.fromarray(tar_img_np[0])
        
        if random.random() < self.heavy_augment_prob:
            if random.random() < 0.5:
                random_choice = np.random.choice([1,2,3])
                if random_choice == 1:   
                    # sharpening
                    random_degree = np.random.choice([1,2,3,4,5])
                    if random_degree == 1:
                        imp_img = imp_img.filter(ImageFilter.DETAIL)
                    elif random_degree == 2:
                        imp_img = imp_img.filter(ImageFilter.SHARPEN)
                    elif random_degree == 3:
                        imp_img = imp_img.filter(ImageFilter.EDGE_ENHANCE)
                    elif random_degree == 4:
                        imp_img = imp_img.filter(ImageFilter.EDGE_ENHANCE_MORE)                            
                    else:
                        aug = iaa.Sharpen(alpha=(0.0, 0.3), lightness=(0.6, 1.0))
                        imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                        imp_img = Image.fromarray(imp_img_np[0])
                elif random_choice == 2:
                    #blurring
                    random_degree = np.random.choice([1,2,3,4,5])
                    if random_degree == 1:
                        imp_img = imp_img.filter(ImageFilter.GaussianBlur())
                    elif random_degree == 2:
                        imp_img = imp_img.filter(ImageFilter.BLUR)
                    elif random_degree == 3:
                        imp_img = imp_img.filter(ImageFilter.SMOOTH)
                    elif random_degree == 4:
                        imp_img = imp_img.filter(ImageFilter.SMOOTH_MORE)
                    else:
                        aug = iaa.MotionBlur(k=3)
                        imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                        imp_img = Image.fromarray(imp_img_np[0])
                else:
                    if random.random() < 0.5:
                        aug = iaa.AdditiveGaussianNoise(scale=3)
                        imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                        imp_img = Image.fromarray(imp_img_np[0])
                    else:
                        aug = iaa.AdditiveLaplaceNoise(scale=3)
                        imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                        imp_img = Image.fromarray(imp_img_np[0])
            if random.random() < 0.5:
                if random.random() < 0.5:
                    # Basic compression and expansion
                    divider = random.uniform(1.0, 1.5)
                    cw, ch = imp_img.size
                    new_cw = int(cw/divider)
                    new_ch = int(ch/divider)
                    
                    first_choice = np.random.choice([1,2,3,4,5,6])
                    if first_choice == 1:
                        imp_img = imp_img.resize((new_cw, new_ch), Image.NEAREST)
                    elif first_choice == 2:
                        imp_img = imp_img.resize((new_cw, new_ch), Image.BILINEAR)
                    elif first_choice == 3:
                        imp_img = imp_img.resize((new_cw, new_ch), Image.BICUBIC)
                    elif first_choice == 4:
                        imp_img = imp_img.resize((new_cw, new_ch), Image.LANCZOS)
                    elif first_choice == 5:
                        imp_img = imp_img.resize((new_cw, new_ch), Image.HAMMING)
                    else:
                        imp_img = imp_img.resize((new_cw, new_ch), Image.BOX)
                    
                    second_choice = np.random.choice([1,2,3,4,5,6])
                    if second_choice == 1:
                        imp_img = imp_img.resize((cw, ch), Image.NEAREST)
                    elif second_choice == 2:
                        imp_img = imp_img.resize((cw, ch), Image.BILINEAR)
                    elif second_choice == 3:
                        imp_img = imp_img.resize((cw, ch), Image.BICUBIC)
                    elif second_choice == 4:
                        imp_img = imp_img.resize((cw, ch), Image.LANCZOS)
                    elif second_choice == 5:
                        imp_img = imp_img.resize((cw, ch), Image.HAMMING)
                    else:
                        imp_img = imp_img.resize((cw, ch), Image.BOX)
                else:
                    #JPEG compression
                    aug = iaa.JpegCompression(compression=(10, 30))
                    imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                    imp_img = Image.fromarray(imp_img_np[0])
                         
            if random.random() < 0.5: # random contrast change
                random_degree = np.random.choice([1,2,3,4,5,6,7,8])                
                if random_degree == 1:
                    aug = iaa.GammaContrast((0.5, 2.0))
                    imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                    imp_img = Image.fromarray(imp_img_np[0])
                elif random_degree == 2:
                    aug = iaa.LinearContrast((0.4, 1.6))
                    imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                    imp_img = Image.fromarray(imp_img_np[0])
                elif random_degree == 3:
                    aug = iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6))
                    imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                    imp_img = Image.fromarray(imp_img_np[0])
                elif random_degree == 4:
                    aug = iaa.LogContrast(gain=(0.6, 1.4))
                    imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                    imp_img = Image.fromarray(imp_img_np[0])            
                else:    
                    aug = iaa.pillike.EnhanceBrightness()
                    imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                    imp_img = Image.fromarray(imp_img_np[0])         
        '''
        if self.normalize:
            inp_img_t = torch.from_numpy(np.float32(inp_img)).float()
            inp_img_mean = torch.mean(inp_img_t)
            inp_img_std = torch.std(inp_img_t)
            xform_inp_img = (inp_img_t - inp_img_mean)/inp_img_std
            xform_inp_img = xform_inp_img.unsqueeze(0)
            
            xform_inp_mask = self.transform(inp_mask)
            xform_inp_mask[xform_inp_mask<0] = -1
            xform_inp_mask[xform_inp_mask>=0] = 1
            
            tar_img_t = torch.from_numpy(np.float32(tar_img)).float()
            tar_img_mean = torch.mean(tar_img_t)
            tar_img_std = torch.std(tar_img_t)
            xform_tar_img = (tar_img_t - tar_img_mean)/tar_img_std
            xform_tar_img = xform_tar_img.unsqueeze(0)
            
            xform_tar_mask = self.transform(tar_mask)
            xform_tar_mask[xform_tar_mask<0] = -1.0
            xform_tar_mask[xform_tar_mask>=0] = 1.0
            
            imp_img_t = torch.from_numpy(np.float32(imp_img)).float()
            imp_img_mean = torch.mean(imp_img_t)
            imp_img_std = torch.std(imp_img_t)
            xform_imp_img = (imp_img_t - imp_img_mean)/imp_img_std
            xform_imp_img = xform_imp_img.unsqueeze(0)
            
            xform_imp_mask = self.transform(imp_mask)
            xform_imp_mask[xform_imp_mask<0] = -1.0
            xform_imp_mask[xform_imp_mask>=0] = 1.0
            return {'inp_img': xform_inp_img, 'inp_img_mean': inp_img_mean, 'inp_img_std':inp_img_std, 'tar_img' : xform_tar_img, 'tar_img_mean':tar_img_mean, 'tar_img_std':tar_img_std, 'imp_img' : xform_imp_img, 'imp_img_mean': imp_img_mean, 'imp_img_std':imp_img_std, 'inp_mask': xform_inp_mask, 'tar_mask' : xform_tar_mask, 'imp_mask' : xform_imp_mask, 'inp_img_pxyr' : inp_img_pupil_xyr, 'tar_img_pxyr' : tar_img_pupil_xyr, 'imp_img_pxyr' : imp_img_pupil_xyr, 'inp_img_ixyr': inp_img_iris_xyr, 'tar_img_ixyr' :  tar_img_iris_xyr, 'imp_img_ixyr' : imp_img_iris_xyr, 'identifier' : triplet[6], 'imposter_id' : triplet[7]}

        else:
            xform_inp_img = self.transform(inp_img)
            xform_inp_mask = self.transform(inp_mask)
            xform_inp_mask[xform_inp_mask<0] = -1
            xform_inp_mask[xform_inp_mask>=0] = 1
            
            xform_tar_img = self.transform(tar_img)
            xform_tar_mask = self.transform(tar_mask)
            xform_tar_mask[xform_tar_mask<0] = -1.0
            xform_tar_mask[xform_tar_mask>=0] = 1.0
            
            xform_imp_img = self.transform(imp_img)
            xform_imp_mask = self.transform(imp_mask)
            xform_imp_mask[xform_imp_mask<0] = -1.0
            xform_imp_mask[xform_imp_mask>=0] = 1.0
            return {'inp_img': xform_inp_img, 'tar_img' : xform_tar_img, 'imp_img' : xform_imp_img, 'inp_mask': xform_inp_mask, 'tar_mask' : xform_tar_mask, 'imp_mask' : xform_imp_mask, 'inp_img_pxyr' : inp_img_pupil_xyr, 'tar_img_pxyr' : tar_img_pupil_xyr, 'imp_img_pxyr' : imp_img_pupil_xyr, 'inp_img_ixyr': inp_img_iris_xyr, 'tar_img_ixyr' :  tar_img_iris_xyr, 'imp_img_ixyr' : imp_img_iris_xyr, 'identifier' : triplet[6], 'imposter_id' : triplet[7]}

            
        #print(xform_inp_img.shape, xform_tar_img.shape,  xform_imp_img.shape)
        
        
    
    def __len__(self):
        return len(self.triplets)
    
    def reset(self):
        print('Shuffling triplets for WSD...')
        self.triplets = []
        self.genuine_ids = []
        self.imposter_ids = []
        wsd_identifiers = list(self.bins_wsd.keys())
        random.shuffle(wsd_identifiers)
        for identifier in tqdm(wsd_identifiers):
            wsd_identifiers_removed = wsd_identifiers.copy()
            wsd_identifiers_removed.remove(identifier)
            bin_numbers1 = list(self.bins_wsd[identifier].keys())
            bin_numbers2 = list(self.bins_wsd[identifier].keys())
            bin_numbers1.sort()
            bin_numbers2.sort()
            for bin_num_1 in bin_numbers1:
                for bin_num_2 in bin_numbers2:
                    if bin_num_1 != bin_num_2:
                        random.shuffle(self.bins_wsd[identifier][bin_num_1])
                        random.shuffle(self.bins_wsd[identifier][bin_num_2])
                        max_pairs = min(len(self.bins_wsd[identifier][bin_num_1]), len(self.bins_wsd[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            imposter_ind_max = 0
                            while imposter_ind_max == 0:
                                imposter_id = random.choice(wsd_identifiers_removed)
                                imposter_ind_max = len(self.bins_wsd[imposter_id][bin_num_2])
                            imagename1 = self.bins_wsd[identifier][bin_num_1][img_ind]
                            imagename2 = self.bins_wsd[identifier][bin_num_2][img_ind]
                            imagename3 = self.bins_wsd[imposter_id][bin_num_2][random.randrange(0, imposter_ind_max)]
                            indiv_id1 = imagename1.split('_')[0]
                            indiv_id2 = imagename2.split('_')[0]
                            indiv_id3 = imagename3.split('_')[0]
                            imagepath1 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id1), imagename1)
                            maskpath1 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id1), imagename1)
                            imagepath2 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id2), imagename2)
                            maskpath2 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id2), imagename2)
                            imagepath3 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id3), imagename3)
                            maskpath3 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id3), imagename3)
                            self.triplets.append([imagepath1, maskpath1, imagepath2, maskpath2, imagepath3, maskpath3, identifier, imposter_id])
                        random.shuffle(self.bins_wsd[identifier][bin_num_1])
                        random.shuffle(self.bins_wsd[identifier][bin_num_2])
                        for img_ind in range(max_pairs):
                            imposter_ind_max = 0
                            while imposter_ind_max == 0:
                                imposter_id = random.choice(wsd_identifiers_removed)
                                imposter_ind_max = len(self.bins_wsd[imposter_id][bin_num_1])
                            imagename1 = self.bins_wsd[identifier][bin_num_2][img_ind]
                            imagename2 = self.bins_wsd[identifier][bin_num_1][img_ind]
                            imagename3 = self.bins_wsd[imposter_id][bin_num_1][random.randrange(0, imposter_ind_max)]
                            indiv_id1 = imagename1.split('_')[0]
                            indiv_id2 = imagename2.split('_')[0]
                            indiv_id3 = imagename3.split('_')[0]
                            imagepath1 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id1), imagename1)
                            maskpath1 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id1), imagename1)
                            imagepath2 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id2), imagename2)
                            maskpath2 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id2), imagename2)
                            imagepath3 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id3), imagename3)
                            maskpath3 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id3), imagename3)
                            self.triplets.append([imagepath1, maskpath1, imagepath2, maskpath2, imagepath3, maskpath3, identifier, imposter_id])              
        
        print('Shuffling triplets for BXGRID...')
        bxgrid_identifiers = list(self.bins_bxgrid.keys())
        random.shuffle(bxgrid_identifiers)
        for identifier in tqdm(bxgrid_identifiers):
            bxgrid_identifiers_removed = bxgrid_identifiers.copy()
            bxgrid_identifiers_removed.remove(identifier)
            bin_numbers1 = list(self.bins_bxgrid[identifier].keys())
            bin_numbers2 = list(self.bins_bxgrid[identifier].keys())
            bin_numbers1.sort()
            bin_numbers2.sort()
            for bin_num_1 in bin_numbers1:
                for bin_num_2 in bin_numbers2:
                    if bin_num_1 != bin_num_2:
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_1])
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_2])
                        max_pairs = min(len(self.bins_bxgrid[identifier][bin_num_1]), len(self.bins_bxgrid[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            imposter_ind_max = 0
                            while imposter_ind_max == 0:
                                imposter_id = random.choice(bxgrid_identifiers_removed)
                                imposter_ind_max = len(self.bins_bxgrid[imposter_id][bin_num_2])
                            imagename1 = self.bins_bxgrid[identifier][bin_num_1][img_ind]
                            imagename2 = self.bins_bxgrid[identifier][bin_num_2][img_ind]
                            imagename3 = self.bins_bxgrid[imposter_id][bin_num_2][random.randrange(0, imposter_ind_max)]
                            imagepath1 = os.path.join(self.image_dir_bxgrid, imagename1)
                            maskpath1 = os.path.join(self.mask_dir_bxgrid, imagename1)
                            imagepath2 = os.path.join(self.image_dir_bxgrid, imagename2)
                            maskpath2 = os.path.join(self.mask_dir_bxgrid, imagename2)
                            imagepath3 = os.path.join(self.image_dir_bxgrid, imagename3)
                            maskpath3 = os.path.join(self.mask_dir_bxgrid, imagename3)
                            self.triplets.append([imagepath1, maskpath1, imagepath2, maskpath2, imagepath3, maskpath3, identifier, imposter_id])
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_1])
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_2])
                        for img_ind in range(max_pairs):
                            imposter_ind_max = 0
                            while imposter_ind_max == 0:
                                imposter_id = random.choice(bxgrid_identifiers_removed)
                                imposter_ind_max = len(self.bins_bxgrid[imposter_id][bin_num_1])
                            imagename1 = self.bins_bxgrid[identifier][bin_num_2][img_ind]
                            imagename2 = self.bins_bxgrid[identifier][bin_num_1][img_ind]
                            imagename3 = self.bins_bxgrid[imposter_id][bin_num_1][random.randrange(0, imposter_ind_max)]
                            imagepath1 = os.path.join(self.image_dir_bxgrid, imagename1)
                            maskpath1 = os.path.join(self.mask_dir_bxgrid, imagename1)
                            imagepath2 = os.path.join(self.image_dir_bxgrid, imagename2)
                            maskpath2 = os.path.join(self.mask_dir_bxgrid, imagename2)
                            imagepath3 = os.path.join(self.image_dir_bxgrid, imagename3)
                            maskpath3 = os.path.join(self.mask_dir_bxgrid, imagename3)
                            self.triplets.append([imagepath1, maskpath1, imagepath2, maskpath2, imagepath3, maskpath3, identifier, imposter_id])
                            
        print('Total triplets: ', len(self.triplets))

class TripletFromBinAllDatasetSingleCropped(Dataset):
    def __init__(self, bins_path_wsd, parent_dir_wsd, bins_path_bxgrid, parent_dir_bxgrid, direction='bs', iris_crop=True, normalize=True, brightness_and_contrast = False, flip_data = True, res_mult=1):
        super().__init__()
        self.iris_crop = iris_crop
        self.flip_data = flip_data
        self.direction = direction
        self.b_n_c = brightness_and_contrast
        self.input_size=(int(256*res_mult),int(256*res_mult))
        self.res_mult = res_mult
        self.normalize = normalize
        
        self.image_dir_wsd = os.path.join(parent_dir_wsd, 'WarsawData2')
        self.mask_dir_wsd = os.path.join(parent_dir_wsd, 'WarsawMask2')            
        
        with open(bins_path_wsd, 'rb') as wsdbinsFile:
            self.bins_wsd = pkl.load(wsdbinsFile)
        
        self.image_dir_bxgrid = os.path.join(parent_dir_bxgrid, 'iris')
        self.mask_dir_bxgrid = os.path.join(parent_dir_bxgrid, 'mask')       
        
        with open(bins_path_bxgrid, 'rb') as bxgridbinsFile:
            self.bins_bxgrid = pkl.load(bxgridbinsFile)
        
        self.all_ids = list(self.bins_wsd.keys()) + list(self.bins_bxgrid.keys())
        
        print('Initializing triplets for Warsaw Pupil Dynamics...')
        self.triplets = []
        self.genuine_ids = []
        self.imposter_ids = []
        wsd_identifiers = list(self.bins_wsd.keys())
        random.shuffle(wsd_identifiers)
        for identifier in tqdm(wsd_identifiers):
            wsd_identifiers_removed = wsd_identifiers.copy()
            wsd_identifiers_removed.remove(identifier)
            bin_numbers1 = list(self.bins_wsd[identifier].keys())
            bin_numbers2 = list(self.bins_wsd[identifier].keys())
            bin_numbers1.sort()
            bin_numbers2.sort()
            for bin_num_1 in bin_numbers1:
                for bin_num_2 in bin_numbers2:
                    check = bin_num_1 > bin_num_2 if self.direction == 'bs' else bin_num_1 < bin_num_2
                    if check:
                        random.shuffle(self.bins_wsd[identifier][bin_num_1])
                        random.shuffle(self.bins_wsd[identifier][bin_num_2])
                        max_pairs = min(len(self.bins_wsd[identifier][bin_num_1]), len(self.bins_wsd[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            imposter_ind_max = 0
                            while imposter_ind_max == 0:
                                imposter_id = random.choice(wsd_identifiers_removed)
                                imposter_ind_max = len(self.bins_wsd[imposter_id][bin_num_2])
                            imagename1 = self.bins_wsd[identifier][bin_num_1][img_ind]
                            imagename2 = self.bins_wsd[identifier][bin_num_2][img_ind]
                            imagename3 = self.bins_wsd[imposter_id][bin_num_2][random.randrange(0, imposter_ind_max)]
                            indiv_id1 = imagename1.split('_')[0]
                            indiv_id2 = imagename2.split('_')[0]
                            indiv_id3 = imagename3.split('_')[0]
                            imagepath1 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id1), imagename1)
                            maskpath1 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id1), imagename1)
                            imagepath2 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id2), imagename2)
                            maskpath2 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id2), imagename2)
                            imagepath3 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id3), imagename3)
                            maskpath3 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id3), imagename3)
                            self.triplets.append([imagepath1, maskpath1, imagepath2, maskpath2, imagepath3, maskpath3, identifier, imposter_id])
                        random.shuffle(self.bins_wsd[identifier][bin_num_1])
                        random.shuffle(self.bins_wsd[identifier][bin_num_2])
                        for img_ind in range(max_pairs):
                            imposter_ind_max = 0
                            while imposter_ind_max == 0:
                                imposter_id = random.choice(wsd_identifiers_removed)
                                imposter_ind_max = len(self.bins_wsd[imposter_id][bin_num_1])
                            imagename1 = self.bins_wsd[identifier][bin_num_2][img_ind]
                            imagename2 = self.bins_wsd[identifier][bin_num_1][img_ind]
                            imagename3 = self.bins_wsd[imposter_id][bin_num_1][random.randrange(0, imposter_ind_max)]
                            indiv_id1 = imagename1.split('_')[0]
                            indiv_id2 = imagename2.split('_')[0]
                            indiv_id3 = imagename3.split('_')[0]
                            imagepath1 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id1), imagename1)
                            maskpath1 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id1), imagename1)
                            imagepath2 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id2), imagename2)
                            maskpath2 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id2), imagename2)
                            imagepath3 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id3), imagename3)
                            maskpath3 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id3), imagename3)
                            self.triplets.append([imagepath1, maskpath1, imagepath2, maskpath2, imagepath3, maskpath3, identifier, imposter_id])              
        
        print('Initializing triplets for BXGRID...')
        bxgrid_identifiers = list(self.bins_bxgrid.keys())
        random.shuffle(bxgrid_identifiers)
        for identifier in tqdm(bxgrid_identifiers):
            bxgrid_identifiers_removed = bxgrid_identifiers.copy()
            bxgrid_identifiers_removed.remove(identifier)
            bin_numbers1 = list(self.bins_bxgrid[identifier].keys())
            bin_numbers2 = list(self.bins_bxgrid[identifier].keys())
            bin_numbers1.sort()
            bin_numbers2.sort()
            for bin_num_1 in bin_numbers1:
                for bin_num_2 in bin_numbers2:
                    check = bin_num_1 > bin_num_2 if self.direction == 'bs' else bin_num_1 < bin_num_2
                    if check:
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_1])
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_2])
                        max_pairs = min(len(self.bins_bxgrid[identifier][bin_num_1]), len(self.bins_bxgrid[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            imposter_ind_max = 0
                            while imposter_ind_max == 0:
                                imposter_id = random.choice(bxgrid_identifiers_removed)
                                imposter_ind_max = len(self.bins_bxgrid[imposter_id][bin_num_2])
                            imagename1 = self.bins_bxgrid[identifier][bin_num_1][img_ind]
                            imagename2 = self.bins_bxgrid[identifier][bin_num_2][img_ind]
                            imagename3 = self.bins_bxgrid[imposter_id][bin_num_2][random.randrange(0, imposter_ind_max)]
                            imagepath1 = os.path.join(self.image_dir_bxgrid, imagename1)
                            maskpath1 = os.path.join(self.mask_dir_bxgrid, imagename1)
                            imagepath2 = os.path.join(self.image_dir_bxgrid, imagename2)
                            maskpath2 = os.path.join(self.mask_dir_bxgrid, imagename2)
                            imagepath3 = os.path.join(self.image_dir_bxgrid, imagename3)
                            maskpath3 = os.path.join(self.mask_dir_bxgrid, imagename3)
                            self.triplets.append([imagepath1, maskpath1, imagepath2, maskpath2, imagepath3, maskpath3, identifier, imposter_id])
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_1])
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_2])
                        for img_ind in range(max_pairs):
                            imposter_ind_max = 0
                            while imposter_ind_max == 0:
                                imposter_id = random.choice(bxgrid_identifiers_removed)
                                imposter_ind_max = len(self.bins_bxgrid[imposter_id][bin_num_1])
                            imagename1 = self.bins_bxgrid[identifier][bin_num_2][img_ind]
                            imagename2 = self.bins_bxgrid[identifier][bin_num_1][img_ind]
                            imagename3 = self.bins_bxgrid[imposter_id][bin_num_1][random.randrange(0, imposter_ind_max)]
                            imagepath1 = os.path.join(self.image_dir_bxgrid, imagename1)
                            maskpath1 = os.path.join(self.mask_dir_bxgrid, imagename1)
                            imagepath2 = os.path.join(self.image_dir_bxgrid, imagename2)
                            maskpath2 = os.path.join(self.mask_dir_bxgrid, imagename2)
                            imagepath3 = os.path.join(self.image_dir_bxgrid, imagename3)
                            maskpath3 = os.path.join(self.mask_dir_bxgrid, imagename3)
                            self.triplets.append([imagepath1, maskpath1, imagepath2, maskpath2, imagepath3, maskpath3, identifier, imposter_id])
                            
        print('Total triplets: ', len(self.triplets))
        

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5, ))
        ])
        
        self.pupil_iris_xyrs_path_wsd = os.path.join(parent_dir_wsd, 'pupil_iris_xyrs2.pkl')
        with open(self.pupil_iris_xyrs_path_wsd, 'rb') as pupilirisxyrsFile:
            self.pupil_iris_xyrs_wsd = pkl.load(pupilirisxyrsFile)
        
        self.pupil_iris_xyrs_path_bxgrid = os.path.join(parent_dir_bxgrid, 'pupil_iris_xyrs.pkl')
        with open(self.pupil_iris_xyrs_path_bxgrid, 'rb') as pupilirisxyrsFile:
            self.pupil_iris_xyrs_bxgrid = pkl.load(pupilirisxyrsFile)
        
        self.pupil_xyrs = {}
        self.iris_xyrs = {}
        
        for imagename in self.pupil_iris_xyrs_wsd.keys():
            indiv_id = imagename.split('_')[0]
            imagepath = os.path.join(os.path.join(self.image_dir_wsd, indiv_id), imagename)
            self.pupil_xyrs[imagepath] = self.pupil_iris_xyrs_wsd[imagename]['pxyr']
            self.iris_xyrs[imagepath] = self.pupil_iris_xyrs_wsd[imagename]['ixyr']
        
        for imagename in self.pupil_iris_xyrs_bxgrid.keys():
            imagepath = os.path.join(self.image_dir_bxgrid, imagename)
            self.pupil_xyrs[imagepath] = self.pupil_iris_xyrs_bxgrid[imagename]['pxyr']
            self.iris_xyrs[imagepath] = self.pupil_iris_xyrs_bxgrid[imagename]['ixyr']
        
        self.heavy_augment_prob = 0.3
    
    def load_image(self, file):
        return Image.open(file).convert('L')
        
    def __getitem__(self, index):
    
        triplet = self.triplets[index]
        
        inp_img_path = triplet[0].strip()
        inp_mask_path = triplet[1].strip()
        tar_img_path = triplet[2].strip()
        tar_mask_path = triplet[3].strip()
        imp_img_path = triplet[4].strip()
        imp_mask_path = triplet[5].strip()
        
        inp_img = self.load_image(inp_img_path)
        inp_mask = self.load_image(inp_mask_path)
        
        tar_img = self.load_image(tar_img_path)
        tar_mask = self.load_image(tar_mask_path)
        
        imp_img = self.load_image(imp_img_path)
        imp_mask = self.load_image(imp_mask_path)
        
        inp_img_pupil_xyr = torch.FloatTensor(self.pupil_xyrs[inp_img_path])
        inp_img_iris_xyr = torch.FloatTensor(self.iris_xyrs[inp_img_path])
        
        tar_img_pupil_xyr = torch.FloatTensor(self.pupil_xyrs[tar_img_path])
        tar_img_iris_xyr = torch.FloatTensor(self.iris_xyrs[tar_img_path])
        
        imp_img_pupil_xyr = torch.FloatTensor(self.pupil_xyrs[imp_img_path])
        imp_img_iris_xyr = torch.FloatTensor(self.iris_xyrs[imp_img_path])
        
        '''
        sw, sh = inp_img.size
        sw_mult = (sw/320)
        sh_mult = (self.input_size[0]/sh)
        
        bw, bh = tar_img.size
        bw_mult = (self.input_size[1]/bw)
        bh_mult = (self.input_size[0]/bh)
        
        biw, bih = imp_img.size
        biw_mult = (self.input_size[1]/biw)
        bih_mult = (self.input_size[0]/bih)
        
        inp_img_pupil_xyr[0] = (inp_img_pupil_xyr[0] * sw_mult)
        inp_img_pupil_xyr[1] = (inp_img_pupil_xyr[1] * sh_mult)
        inp_img_pupil_xyr[2] = (inp_img_pupil_xyr[2] * max(sw_mult, sh_mult))
        
        tar_img_pupil_xyr[0] = (tar_img_pupil_xyr[0] * bw_mult)
        tar_img_pupil_xyr[1] = (tar_img_pupil_xyr[1] * bh_mult)
        tar_img_pupil_xyr[2] = (tar_img_pupil_xyr[2] * max(bw_mult, bh_mult))
        
        imp_img_pupil_xyr[0] = (imp_img_pupil_xyr[0] * biw_mult)
        imp_img_pupil_xyr[1] = (imp_img_pupil_xyr[1] * bih_mult)
        imp_img_pupil_xyr[2] = (imp_img_pupil_xyr[2] * max(biw_mult, bih_mult))
    
        inp_img_iris_xyr[0] = (inp_img_iris_xyr[0] * sw_mult)
        inp_img_iris_xyr[1] = (inp_img_iris_xyr[1] * sh_mult)
        inp_img_iris_xyr[2] = (inp_img_iris_xyr[2] * max(sw_mult, sh_mult))
        
        tar_img_iris_xyr[0] = (tar_img_iris_xyr[0] * bw_mult)
        tar_img_iris_xyr[1] = (tar_img_iris_xyr[1] * bh_mult)
        tar_img_iris_xyr[2] = (tar_img_iris_xyr[2] * max(bw_mult, bh_mult))
        
        imp_img_iris_xyr[0] = (imp_img_iris_xyr[0] * biw_mult)
        imp_img_iris_xyr[1] = (imp_img_iris_xyr[1] * bih_mult)
        imp_img_iris_xyr[2] = (imp_img_iris_xyr[2] * max(biw_mult, bih_mult))
        '''
        sw, sh = inp_img.size
        bw, bh = tar_img.size
        iw, ih = imp_img.size
        
        if self.flip_data and random.random() < 0.5:
            inp_img = inp_img.transpose(Image.FLIP_LEFT_RIGHT) 
            inp_mask = inp_mask.transpose(Image.FLIP_LEFT_RIGHT)
            tar_img = tar_img.transpose(Image.FLIP_LEFT_RIGHT) 
            tar_mask = tar_mask.transpose(Image.FLIP_LEFT_RIGHT)
            inp_img_pupil_xyr[0] = sw - inp_img_pupil_xyr[0]
            tar_img_pupil_xyr[0] = bw - tar_img_pupil_xyr[0]
            inp_img_iris_xyr[0] = sw - inp_img_iris_xyr[0]
            tar_img_iris_xyr[0] = bw - tar_img_iris_xyr[0]
        
        if self.flip_data and random.random() < 0.5:
            imp_img = imp_img.transpose(Image.FLIP_LEFT_RIGHT)
            imp_mask = imp_mask.transpose(Image.FLIP_LEFT_RIGHT)
            imp_img_pupil_xyr[0] = iw - imp_img_pupil_xyr[0]
            imp_img_iris_xyr[0] = iw - imp_img_iris_xyr[0]
        
        if self.iris_crop:
            #inp_img.save('./check1.png')
            #print((inp_img_iris_xyr[0].item() - inp_img_iris_xyr[2].item(), inp_img_iris_xyr[1].item() - inp_img_iris_xyr[2].item(), inp_img_iris_xyr[0].item() + inp_img_iris_xyr[2].item(), inp_img_iris_xyr[1].item() + inp_img_iris_xyr[2].item()))
            inp_img = inp_img.crop((inp_img_iris_xyr[0].item() - inp_img_iris_xyr[2].item(), inp_img_iris_xyr[1].item() - inp_img_iris_xyr[2].item(), inp_img_iris_xyr[0].item() + inp_img_iris_xyr[2].item(), inp_img_iris_xyr[1].item() + inp_img_iris_xyr[2].item()))
            inp_mask = inp_mask.crop((inp_img_iris_xyr[0].item() - inp_img_iris_xyr[2].item(), inp_img_iris_xyr[1].item() - inp_img_iris_xyr[2].item(), inp_img_iris_xyr[0].item() + inp_img_iris_xyr[2].item(), inp_img_iris_xyr[1].item() + inp_img_iris_xyr[2].item()))
            sw_crop, sh_crop = inp_img.size
            inp_img_pupil_xyr[0] = inp_img_pupil_xyr[0] - inp_img_iris_xyr[0] + sw_crop/2
            inp_img_pupil_xyr[1] = inp_img_pupil_xyr[1] - inp_img_iris_xyr[1] + sh_crop/2
            inp_img_iris_xyr[0] = sw_crop/2
            inp_img_iris_xyr[1] = sh_crop/2            
            
            tar_img = tar_img.crop((tar_img_iris_xyr[0].item() - tar_img_iris_xyr[2].item(), tar_img_iris_xyr[1].item() - tar_img_iris_xyr[2].item(), tar_img_iris_xyr[0].item() + tar_img_iris_xyr[2].item(), tar_img_iris_xyr[1].item() + tar_img_iris_xyr[2].item()))
            tar_mask = tar_mask.crop((tar_img_iris_xyr[0].item() - tar_img_iris_xyr[2].item(), tar_img_iris_xyr[1].item() - tar_img_iris_xyr[2].item(), tar_img_iris_xyr[0].item() + tar_img_iris_xyr[2].item(), tar_img_iris_xyr[1].item() + tar_img_iris_xyr[2].item()))
            bw_crop, bh_crop = tar_img.size
            tar_img_pupil_xyr[0] = tar_img_pupil_xyr[0] - tar_img_iris_xyr[0] + bw_crop/2
            tar_img_pupil_xyr[1] = tar_img_pupil_xyr[1] - tar_img_iris_xyr[1] + bh_crop/2
            tar_img_iris_xyr[0] = bw_crop/2
            tar_img_iris_xyr[1] = bh_crop/2
            
            imp_img = imp_img.crop((imp_img_iris_xyr[0].item() - imp_img_iris_xyr[2].item(), imp_img_iris_xyr[1].item() - imp_img_iris_xyr[2].item(), imp_img_iris_xyr[0].item() + imp_img_iris_xyr[2].item(), imp_img_iris_xyr[1].item() + imp_img_iris_xyr[2].item()))
            imp_mask = imp_mask.crop((imp_img_iris_xyr[0].item() - imp_img_iris_xyr[2].item(), imp_img_iris_xyr[1].item() - imp_img_iris_xyr[2].item(), imp_img_iris_xyr[0].item() + imp_img_iris_xyr[2].item(), imp_img_iris_xyr[1].item() + imp_img_iris_xyr[2].item()))
            iw_crop, ih_crop = imp_img.size
            imp_img_pupil_xyr[0] = imp_img_pupil_xyr[0] - imp_img_iris_xyr[0] + iw_crop/2
            imp_img_pupil_xyr[1] = imp_img_pupil_xyr[1] - imp_img_iris_xyr[1] + ih_crop/2
            imp_img_iris_xyr[0] = iw_crop/2
            imp_img_iris_xyr[1] = ih_crop/2
            
            '''
            inp_img_vis = np.uint8(inp_img.convert('RGB'))
            try:
                inp_img_vis = cv2.circle(inp_img_vis, (int(inp_img_pupil_xyr[0].item()),int(inp_img_pupil_xyr[1].item())), int(inp_img_pupil_xyr[2].item()), (0, 0, 255), 2)
            except:
                print("Pupil circle could not be visualized, values are: ", (int(inp_img_pupil_xyr[0].item()),int(inp_img_pupil_xyr[1].item()), int(inp_img_pupil_xyr[2].item())), ' ', inp_img_vis.shape)
                pass
            try:
                inp_img_vis = cv2.circle(inp_img_vis, (int(inp_img_iris_xyr[0].item()),int(inp_img_iris_xyr[1].item())), int(inp_img_iris_xyr[2].item()), (255, 0, 0), 2)
            except:
                print("Iris circle could not be visualized, values are: ", (int(inp_img_iris_xyr[0].item()),int(inp_img_iris_xyr[1].item()), int(inp_img_iris_xyr[2].item())), ' ', inp_img_vis.shape)
            
            Image.fromarray(inp_img_vis).save('./check2.png')
            '''
            
            sh_mult = self.input_size[0]/sh_crop
            sw_mult = self.input_size[1]/sw_crop
            inp_img = inp_img.resize((self.input_size[1], self.input_size[0]))
            inp_mask = inp_mask.resize((self.input_size[1], self.input_size[0]))
            inp_img_pupil_xyr[0] = self.input_size[1]/2 - sw_mult * (sw_crop/2 - inp_img_pupil_xyr[0])
            inp_img_pupil_xyr[1] = self.input_size[0]/2 - sh_mult * (sh_crop/2 - inp_img_pupil_xyr[1])
            inp_img_pupil_xyr[2] = max(sw_mult, sh_mult) * inp_img_pupil_xyr[2]
            inp_img_iris_xyr[0] = self.input_size[1]/2
            inp_img_iris_xyr[1] = self.input_size[0]/2
            inp_img_iris_xyr[2] = max(sw_mult, sh_mult) * inp_img_iris_xyr[2]
            
            bh_mult = self.input_size[0]/bh_crop
            bw_mult = self.input_size[1]/bw_crop
            tar_img = tar_img.resize((self.input_size[1], self.input_size[0]))
            tar_mask = tar_mask.resize((self.input_size[1], self.input_size[0]))
            tar_img_pupil_xyr[0] = self.input_size[1]/2 - bw_mult * (bw_crop/2 - tar_img_pupil_xyr[0])
            tar_img_pupil_xyr[1] = self.input_size[0]/2 - bh_mult * (bh_crop/2 - tar_img_pupil_xyr[1])
            tar_img_pupil_xyr[2] = max(bw_mult, bh_mult) * tar_img_pupil_xyr[2]
            tar_img_iris_xyr[0] = self.input_size[1]/2
            tar_img_iris_xyr[1] = self.input_size[0]/2
            tar_img_iris_xyr[2] = max(bw_mult, bh_mult) * tar_img_iris_xyr[2]
            
            ih_mult = self.input_size[0]/ih_crop
            iw_mult = self.input_size[1]/iw_crop
            imp_img = imp_img.resize((self.input_size[0], self.input_size[1]))
            imp_mask = imp_mask.resize((self.input_size[0], self.input_size[1]))
            imp_img_pupil_xyr[0] = self.input_size[1]/2 - iw_mult * (iw_crop/2 - imp_img_pupil_xyr[0])
            imp_img_pupil_xyr[1] = self.input_size[0]/2 - ih_mult * (ih_crop/2 - imp_img_pupil_xyr[1])
            imp_img_pupil_xyr[2] = max(iw_mult, ih_mult) * imp_img_pupil_xyr[2]
            imp_img_iris_xyr[0] = self.input_size[1]/2
            imp_img_iris_xyr[1] = self.input_size[0]/2
            imp_img_iris_xyr[2] = max(iw_mult, ih_mult) * imp_img_iris_xyr[2]
            
            #print('Saving ..')
            '''
            inp_img_vis = np.uint8(inp_img.convert('RGB'))
            try:
                inp_img_vis = cv2.circle(inp_img_vis, (int(inp_img_pupil_xyr[0].item()),int(inp_img_pupil_xyr[1].item())), int(inp_img_pupil_xyr[2].item()), (0, 0, 255), 2)
            except:
                print("Pupil circle could not be visualized, values are: ", (int(inp_img_pupil_xyr[0].item()),int(inp_img_pupil_xyr[1].item()), int(inp_img_pupil_xyr[2].item())), ' ', inp_img_vis.shape)
                pass
            try:
                inp_img_vis = cv2.circle(inp_img_vis, (int(inp_img_iris_xyr[0].item()),int(inp_img_iris_xyr[1].item())), int(inp_img_iris_xyr[2].item()), (255, 0, 0), 2)
            except:
                print("Iris circle could not be visualized, values are: ", (int(inp_img_iris_xyr[0].item()),int(inp_img_iris_xyr[1].item()), int(inp_img_iris_xyr[2].item())), ' ', inp_img_vis.shape)
            
            Image.fromarray(inp_img_vis).save('./check3.png')
            '''
        if random.random() < self.heavy_augment_prob:
            if random.random() < 0.5:
                random_choice = np.random.choice([1,2,3])
                if random_choice == 1:   
                    # sharpening
                    random_degree = np.random.choice([1,2,3,4,5])
                    if random_degree == 1:
                        inp_img = inp_img.filter(ImageFilter.DETAIL)
                    elif random_degree == 2:
                        inp_img = inp_img.filter(ImageFilter.SHARPEN)
                    elif random_degree == 3:
                        inp_img = inp_img.filter(ImageFilter.EDGE_ENHANCE)
                    elif random_degree == 4:
                        inp_img = inp_img.filter(ImageFilter.EDGE_ENHANCE_MORE)                            
                    else:
                        aug = iaa.Sharpen(alpha=(0.0, 0.3), lightness=(0.6, 1.0))
                        inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                        inp_img = Image.fromarray(inp_img_np[0])
                elif random_choice == 2:
                    #blurring
                    random_degree = np.random.choice([1,2,3,4,5])
                    if random_degree == 1:
                        inp_img = inp_img.filter(ImageFilter.GaussianBlur())
                    elif random_degree == 2:
                        inp_img = inp_img.filter(ImageFilter.BLUR)
                    elif random_degree == 3:
                        inp_img = inp_img.filter(ImageFilter.SMOOTH)
                    elif random_degree == 4:
                        inp_img = inp_img.filter(ImageFilter.SMOOTH_MORE)
                    else:
                        aug = iaa.MotionBlur(k=3)
                        inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                        inp_img = Image.fromarray(inp_img_np[0])
                else:
                    if random.random() < 0.5:
                        aug = iaa.AdditiveGaussianNoise(scale=3)
                        inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                        inp_img = Image.fromarray(inp_img_np[0])
                    else:
                        aug = iaa.AdditiveLaplaceNoise(scale=3)
                        inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                        inp_img = Image.fromarray(inp_img_np[0])
            if random.random() < 0.5:
                if random.random() < 0.5:
                    # Basic compression and expansion
                    divider = random.uniform(1.0, 1.5)
                    cw, ch = inp_img.size
                    new_cw = int(cw/divider)
                    new_ch = int(ch/divider)
                    
                    first_choice = np.random.choice([1,2,3,4,5,6])
                    if first_choice == 1:
                        inp_img = inp_img.resize((new_cw, new_ch), Image.NEAREST)
                    elif first_choice == 2:
                        inp_img = inp_img.resize((new_cw, new_ch), Image.BILINEAR)
                    elif first_choice == 3:
                        inp_img = inp_img.resize((new_cw, new_ch), Image.BICUBIC)
                    elif first_choice == 4:
                        inp_img = inp_img.resize((new_cw, new_ch), Image.LANCZOS)
                    elif first_choice == 5:
                        inp_img = inp_img.resize((new_cw, new_ch), Image.HAMMING)
                    else:
                        inp_img = inp_img.resize((new_cw, new_ch), Image.BOX)
                    
                    second_choice = np.random.choice([1,2,3,4,5,6])
                    if second_choice == 1:
                        inp_img = inp_img.resize((cw, ch), Image.NEAREST)
                    elif second_choice == 2:
                        inp_img = inp_img.resize((cw, ch), Image.BILINEAR)
                    elif second_choice == 3:
                        inp_img = inp_img.resize((cw, ch), Image.BICUBIC)
                    elif second_choice == 4:
                        inp_img = inp_img.resize((cw, ch), Image.LANCZOS)
                    elif second_choice == 5:
                        inp_img = inp_img.resize((cw, ch), Image.HAMMING)
                    else:
                        inp_img = inp_img.resize((cw, ch), Image.BOX)
                else:
                    #JPEG compression
                    aug = iaa.JpegCompression(compression=(10, 30))
                    inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                    inp_img = Image.fromarray(inp_img_np[0])
                         
            if random.random() < 0.5: # random contrast change
                random_degree = np.random.choice([1,2,3,4,5,6,7,8])                
                if random_degree == 1:
                    aug = iaa.GammaContrast((0.5, 2.0))
                    inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                    inp_img = Image.fromarray(inp_img_np[0])
                elif random_degree == 2:
                    aug = iaa.LinearContrast((0.4, 1.6))
                    inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                    inp_img = Image.fromarray(inp_img_np[0])
                elif random_degree == 3:
                    aug = iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6))
                    inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                    inp_img = Image.fromarray(inp_img_np[0])
                elif random_degree == 4:
                    aug = iaa.LogContrast(gain=(0.6, 1.4))
                    inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                    inp_img = Image.fromarray(inp_img_np[0])            
                else:    
                    aug = iaa.pillike.EnhanceBrightness()
                    inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                    inp_img = Image.fromarray(inp_img_np[0])
        '''         
        if random.random() < self.heavy_augment_prob:
            if random.random() < 0.5:
                random_choice = np.random.choice([1,2])
                if random_choice == 1:   
                    # sharpening
                    random_degree = np.random.choice([1,2,3,4,5])
                    if random_degree == 1:
                        tar_img = tar_img.filter(ImageFilter.DETAIL)
                    elif random_degree == 2:
                        tar_img = tar_img.filter(ImageFilter.SHARPEN)
                    elif random_degree == 3:
                        tar_img = tar_img.filter(ImageFilter.EDGE_ENHANCE)
                    elif random_degree == 4:
                        tar_img = tar_img.filter(ImageFilter.EDGE_ENHANCE_MORE)                            
                    else:
                        aug = iaa.Sharpen(alpha=(0.0, 0.3), lightness=(0.6, 1.0))
                        tar_img_np = aug(images = np.expand_dims(np.array(tar_img), axis=0))
                        tar_img = Image.fromarray(tar_img_np[0])
                else:
                    #blurring
                    random_degree = np.random.choice([1,2,3,4,5])
                    if random_degree == 1:
                        tar_img = tar_img.filter(ImageFilter.GaussianBlur())
                    elif random_degree == 2:
                        tar_img = tar_img.filter(ImageFilter.BLUR)
                    elif random_degree == 3:
                        tar_img = tar_img.filter(ImageFilter.SMOOTH)
                    elif random_degree == 4:
                        tar_img = tar_img.filter(ImageFilter.SMOOTH_MORE)
                    else:
                        aug = iaa.MotionBlur(k=3)
                        tar_img_np = aug(images = np.expand_dims(np.array(tar_img), axis=0))
                        tar_img = Image.fromarray(tar_img_np[0])
            if random.random() < 0.5:
                if random.random() < 0.5:
                    # Basic compression and expansion
                    divider = random.uniform(1.0, 1.5)
                    cw, ch = tar_img.size
                    new_cw = int(cw/divider)
                    new_ch = int(ch/divider)
                    
                    first_choice = np.random.choice([1,2,3,4,5,6])
                    if first_choice == 1:
                        tar_img = tar_img.resize((new_cw, new_ch), Image.NEAREST)
                    elif first_choice == 2:
                        tar_img = tar_img.resize((new_cw, new_ch), Image.BILINEAR)
                    elif first_choice == 3:
                        tar_img = tar_img.resize((new_cw, new_ch), Image.BICUBIC)
                    elif first_choice == 4:
                        tar_img = tar_img.resize((new_cw, new_ch), Image.LANCZOS)
                    elif first_choice == 5:
                        tar_img = tar_img.resize((new_cw, new_ch), Image.HAMMING)
                    else:
                        tar_img = tar_img.resize((new_cw, new_ch), Image.BOX)
                    
                    second_choice = np.random.choice([1,2,3,4,5,6])
                    if second_choice == 1:
                        tar_img = tar_img.resize((cw, ch), Image.NEAREST)
                    elif second_choice == 2:
                        tar_img = tar_img.resize((cw, ch), Image.BILINEAR)
                    elif second_choice == 3:
                        tar_img = tar_img.resize((cw, ch), Image.BICUBIC)
                    elif second_choice == 4:
                        tar_img = tar_img.resize((cw, ch), Image.LANCZOS)
                    elif second_choice == 5:
                        tar_img = tar_img.resize((cw, ch), Image.HAMMING)
                    else:
                        tar_img = tar_img.resize((cw, ch), Image.BOX)
                else:
                    #JPEG compression
                    aug = iaa.JpegCompression(compression=(10, 30))
                    tar_img_np = aug(images = np.expand_dims(np.array(tar_img), axis=0))
                    tar_img = Image.fromarray(tar_img_np[0])
                         
            if random.random() < 0.5: # random contrast change
                random_degree = np.random.choice([1,2,3,4,5,6,7,8])                
                if random_degree == 1:
                    aug = iaa.GammaContrast((0.5, 2.0))
                    tar_img_np = aug(images = np.expand_dims(np.array(tar_img), axis=0))
                    tar_img = Image.fromarray(tar_img_np[0])
                elif random_degree == 2:
                    aug = iaa.LinearContrast((0.4, 1.6))
                    tar_img_np = aug(images = np.expand_dims(np.array(tar_img), axis=0))
                    tar_img = Image.fromarray(tar_img_np[0])
                elif random_degree == 3:
                    aug = iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6))
                    tar_img_np = aug(images = np.expand_dims(np.array(tar_img), axis=0))
                    tar_img = Image.fromarray(tar_img_np[0])
                elif random_degree == 4:
                    aug = iaa.LogContrast(gain=(0.6, 1.4))
                    tar_img_np = aug(images = np.expand_dims(np.array(tar_img), axis=0))
                    tar_img = Image.fromarray(tar_img_np[0])            
                else:    
                    aug = iaa.pillike.EnhanceBrightness()
                    tar_img_np = aug(images = np.expand_dims(np.array(tar_img), axis=0))
                    tar_img = Image.fromarray(tar_img_np[0])
        
        if random.random() < self.heavy_augment_prob:
            if random.random() < 0.5:
                random_choice = np.random.choice([1,2,3])
                if random_choice == 1:   
                    # sharpening
                    random_degree = np.random.choice([1,2,3,4,5])
                    if random_degree == 1:
                        imp_img = imp_img.filter(ImageFilter.DETAIL)
                    elif random_degree == 2:
                        imp_img = imp_img.filter(ImageFilter.SHARPEN)
                    elif random_degree == 3:
                        imp_img = imp_img.filter(ImageFilter.EDGE_ENHANCE)
                    elif random_degree == 4:
                        imp_img = imp_img.filter(ImageFilter.EDGE_ENHANCE_MORE)                            
                    else:
                        aug = iaa.Sharpen(alpha=(0.0, 0.3), lightness=(0.6, 1.0))
                        imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                        imp_img = Image.fromarray(imp_img_np[0])
                elif random_choice == 2:
                    #blurring
                    random_degree = np.random.choice([1,2,3,4,5])
                    if random_degree == 1:
                        imp_img = imp_img.filter(ImageFilter.GaussianBlur())
                    elif random_degree == 2:
                        imp_img = imp_img.filter(ImageFilter.BLUR)
                    elif random_degree == 3:
                        imp_img = imp_img.filter(ImageFilter.SMOOTH)
                    elif random_degree == 4:
                        imp_img = imp_img.filter(ImageFilter.SMOOTH_MORE)
                    else:
                        aug = iaa.MotionBlur(k=3)
                        imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                        imp_img = Image.fromarray(imp_img_np[0])
                else:
                    if random.random() < 0.5:
                        aug = iaa.AdditiveGaussianNoise(scale=3)
                        imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                        imp_img = Image.fromarray(imp_img_np[0])
                    else:
                        aug = iaa.AdditiveLaplaceNoise(scale=3)
                        imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                        imp_img = Image.fromarray(imp_img_np[0])
            if random.random() < 0.5:
                if random.random() < 0.5:
                    # Basic compression and expansion
                    divider = random.uniform(1.0, 1.5)
                    cw, ch = imp_img.size
                    new_cw = int(cw/divider)
                    new_ch = int(ch/divider)
                    
                    first_choice = np.random.choice([1,2,3,4,5,6])
                    if first_choice == 1:
                        imp_img = imp_img.resize((new_cw, new_ch), Image.NEAREST)
                    elif first_choice == 2:
                        imp_img = imp_img.resize((new_cw, new_ch), Image.BILINEAR)
                    elif first_choice == 3:
                        imp_img = imp_img.resize((new_cw, new_ch), Image.BICUBIC)
                    elif first_choice == 4:
                        imp_img = imp_img.resize((new_cw, new_ch), Image.LANCZOS)
                    elif first_choice == 5:
                        imp_img = imp_img.resize((new_cw, new_ch), Image.HAMMING)
                    else:
                        imp_img = imp_img.resize((new_cw, new_ch), Image.BOX)
                    
                    second_choice = np.random.choice([1,2,3,4,5,6])
                    if second_choice == 1:
                        imp_img = imp_img.resize((cw, ch), Image.NEAREST)
                    elif second_choice == 2:
                        imp_img = imp_img.resize((cw, ch), Image.BILINEAR)
                    elif second_choice == 3:
                        imp_img = imp_img.resize((cw, ch), Image.BICUBIC)
                    elif second_choice == 4:
                        imp_img = imp_img.resize((cw, ch), Image.LANCZOS)
                    elif second_choice == 5:
                        imp_img = imp_img.resize((cw, ch), Image.HAMMING)
                    else:
                        imp_img = imp_img.resize((cw, ch), Image.BOX)
                else:
                    #JPEG compression
                    aug = iaa.JpegCompression(compression=(10, 30))
                    imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                    imp_img = Image.fromarray(imp_img_np[0])
                         
            if random.random() < 0.5: # random contrast change
                random_degree = np.random.choice([1,2,3,4,5,6,7,8])                
                if random_degree == 1:
                    aug = iaa.GammaContrast((0.5, 2.0))
                    imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                    imp_img = Image.fromarray(imp_img_np[0])
                elif random_degree == 2:
                    aug = iaa.LinearContrast((0.4, 1.6))
                    imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                    imp_img = Image.fromarray(imp_img_np[0])
                elif random_degree == 3:
                    aug = iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6))
                    imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                    imp_img = Image.fromarray(imp_img_np[0])
                elif random_degree == 4:
                    aug = iaa.LogContrast(gain=(0.6, 1.4))
                    imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                    imp_img = Image.fromarray(imp_img_np[0])            
                else:    
                    aug = iaa.pillike.EnhanceBrightness()
                    imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                    imp_img = Image.fromarray(imp_img_np[0])         
        '''
        if self.normalize:
            inp_img_t = torch.from_numpy(np.float32(inp_img)).float()
            inp_img_mean = torch.mean(inp_img_t)
            inp_img_std = torch.std(inp_img_t)
            xform_inp_img = (inp_img_t - inp_img_mean)/inp_img_std
            xform_inp_img = xform_inp_img.unsqueeze(0)
            
            xform_inp_mask = self.transform(inp_mask)
            xform_inp_mask[xform_inp_mask<0] = -1
            xform_inp_mask[xform_inp_mask>=0] = 1
            
            tar_img_t = torch.from_numpy(np.float32(tar_img)).float()
            tar_img_mean = torch.mean(tar_img_t)
            tar_img_std = torch.std(tar_img_t)
            xform_tar_img = (tar_img_t - tar_img_mean)/tar_img_std
            xform_tar_img = xform_tar_img.unsqueeze(0)
            
            xform_tar_mask = self.transform(tar_mask)
            xform_tar_mask[xform_tar_mask<0] = -1.0
            xform_tar_mask[xform_tar_mask>=0] = 1.0
            
            imp_img_t = torch.from_numpy(np.float32(imp_img)).float()
            imp_img_mean = torch.mean(imp_img_t)
            imp_img_std = torch.std(imp_img_t)
            xform_imp_img = (imp_img_t - imp_img_mean)/imp_img_std
            xform_imp_img = xform_imp_img.unsqueeze(0)
            
            xform_imp_mask = self.transform(imp_mask)
            xform_imp_mask[xform_imp_mask<0] = -1.0
            xform_imp_mask[xform_imp_mask>=0] = 1.0
            return {'inp_img': xform_inp_img, 'inp_img_mean': inp_img_mean, 'inp_img_std':inp_img_std, 'tar_img' : xform_tar_img, 'tar_img_mean':tar_img_mean, 'tar_img_std':tar_img_std, 'imp_img' : xform_imp_img, 'imp_img_mean': imp_img_mean, 'imp_img_std':imp_img_std, 'inp_mask': xform_inp_mask, 'tar_mask' : xform_tar_mask, 'imp_mask' : xform_imp_mask, 'inp_img_pxyr' : inp_img_pupil_xyr, 'tar_img_pxyr' : tar_img_pupil_xyr, 'imp_img_pxyr' : imp_img_pupil_xyr, 'inp_img_ixyr': inp_img_iris_xyr, 'tar_img_ixyr' :  tar_img_iris_xyr, 'imp_img_ixyr' : imp_img_iris_xyr, 'identifier' : triplet[6], 'imposter_id' : triplet[7]}

        else:
            xform_inp_img = self.transform(inp_img)
            xform_inp_mask = self.transform(inp_mask)
            xform_inp_mask[xform_inp_mask<0] = -1
            xform_inp_mask[xform_inp_mask>=0] = 1
            
            xform_tar_img = self.transform(tar_img)
            xform_tar_mask = self.transform(tar_mask)
            xform_tar_mask[xform_tar_mask<0] = -1.0
            xform_tar_mask[xform_tar_mask>=0] = 1.0
            
            xform_imp_img = self.transform(imp_img)
            xform_imp_mask = self.transform(imp_mask)
            xform_imp_mask[xform_imp_mask<0] = -1.0
            xform_imp_mask[xform_imp_mask>=0] = 1.0
            return {'inp_img': xform_inp_img, 'tar_img' : xform_tar_img, 'imp_img' : xform_imp_img, 'inp_mask': xform_inp_mask, 'tar_mask' : xform_tar_mask, 'imp_mask' : xform_imp_mask, 'inp_img_pxyr' : inp_img_pupil_xyr, 'tar_img_pxyr' : tar_img_pupil_xyr, 'imp_img_pxyr' : imp_img_pupil_xyr, 'inp_img_ixyr': inp_img_iris_xyr, 'tar_img_ixyr' :  tar_img_iris_xyr, 'imp_img_ixyr' : imp_img_iris_xyr, 'identifier' : triplet[6], 'imposter_id' : triplet[7]}

            
        #print(xform_inp_img.shape, xform_tar_img.shape,  xform_imp_img.shape)
        
        
    
    def __len__(self):
        return len(self.triplets)
    
    def reset(self):
        print('Shuffling triplets for WSD...')
        self.triplets = []
        self.genuine_ids = []
        self.imposter_ids = []
        wsd_identifiers = list(self.bins_wsd.keys())
        random.shuffle(wsd_identifiers)
        for identifier in tqdm(wsd_identifiers):
            wsd_identifiers_removed = wsd_identifiers.copy()
            wsd_identifiers_removed.remove(identifier)
            bin_numbers1 = list(self.bins_wsd[identifier].keys())
            bin_numbers2 = list(self.bins_wsd[identifier].keys())
            bin_numbers1.sort()
            bin_numbers2.sort()
            for bin_num_1 in bin_numbers1:
                for bin_num_2 in bin_numbers2:
                    check = bin_num_1 > bin_num_2 if self.direction == 'bs' else bin_num_1 < bin_num_2
                    if check:
                        random.shuffle(self.bins_wsd[identifier][bin_num_1])
                        random.shuffle(self.bins_wsd[identifier][bin_num_2])
                        max_pairs = min(len(self.bins_wsd[identifier][bin_num_1]), len(self.bins_wsd[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            imposter_ind_max = 0
                            while imposter_ind_max == 0:
                                imposter_id = random.choice(wsd_identifiers_removed)
                                imposter_ind_max = len(self.bins_wsd[imposter_id][bin_num_2])
                            imagename1 = self.bins_wsd[identifier][bin_num_1][img_ind]
                            imagename2 = self.bins_wsd[identifier][bin_num_2][img_ind]
                            imagename3 = self.bins_wsd[imposter_id][bin_num_2][random.randrange(0, imposter_ind_max)]
                            indiv_id1 = imagename1.split('_')[0]
                            indiv_id2 = imagename2.split('_')[0]
                            indiv_id3 = imagename3.split('_')[0]
                            imagepath1 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id1), imagename1)
                            maskpath1 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id1), imagename1)
                            imagepath2 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id2), imagename2)
                            maskpath2 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id2), imagename2)
                            imagepath3 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id3), imagename3)
                            maskpath3 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id3), imagename3)
                            self.triplets.append([imagepath1, maskpath1, imagepath2, maskpath2, imagepath3, maskpath3, identifier, imposter_id])
                        random.shuffle(self.bins_wsd[identifier][bin_num_1])
                        random.shuffle(self.bins_wsd[identifier][bin_num_2])
                        for img_ind in range(max_pairs):
                            imposter_ind_max = 0
                            while imposter_ind_max == 0:
                                imposter_id = random.choice(wsd_identifiers_removed)
                                imposter_ind_max = len(self.bins_wsd[imposter_id][bin_num_1])
                            imagename1 = self.bins_wsd[identifier][bin_num_2][img_ind]
                            imagename2 = self.bins_wsd[identifier][bin_num_1][img_ind]
                            imagename3 = self.bins_wsd[imposter_id][bin_num_1][random.randrange(0, imposter_ind_max)]
                            indiv_id1 = imagename1.split('_')[0]
                            indiv_id2 = imagename2.split('_')[0]
                            indiv_id3 = imagename3.split('_')[0]
                            imagepath1 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id1), imagename1)
                            maskpath1 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id1), imagename1)
                            imagepath2 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id2), imagename2)
                            maskpath2 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id2), imagename2)
                            imagepath3 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id3), imagename3)
                            maskpath3 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id3), imagename3)
                            self.triplets.append([imagepath1, maskpath1, imagepath2, maskpath2, imagepath3, maskpath3, identifier, imposter_id])              
        
        print('Shuffling triplets for BXGRID...')
        bxgrid_identifiers = list(self.bins_bxgrid.keys())
        random.shuffle(bxgrid_identifiers)
        for identifier in tqdm(bxgrid_identifiers):
            bxgrid_identifiers_removed = bxgrid_identifiers.copy()
            bxgrid_identifiers_removed.remove(identifier)
            bin_numbers1 = list(self.bins_bxgrid[identifier].keys())
            bin_numbers2 = list(self.bins_bxgrid[identifier].keys())
            bin_numbers1.sort()
            bin_numbers2.sort()
            for bin_num_1 in bin_numbers1:
                for bin_num_2 in bin_numbers2:
                    check = bin_num_1 > bin_num_2 if self.direction == 'bs' else bin_num_1 < bin_num_2
                    if check:
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_1])
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_2])
                        max_pairs = min(len(self.bins_bxgrid[identifier][bin_num_1]), len(self.bins_bxgrid[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            imposter_ind_max = 0
                            while imposter_ind_max == 0:
                                imposter_id = random.choice(bxgrid_identifiers_removed)
                                imposter_ind_max = len(self.bins_bxgrid[imposter_id][bin_num_2])
                            imagename1 = self.bins_bxgrid[identifier][bin_num_1][img_ind]
                            imagename2 = self.bins_bxgrid[identifier][bin_num_2][img_ind]
                            imagename3 = self.bins_bxgrid[imposter_id][bin_num_2][random.randrange(0, imposter_ind_max)]
                            imagepath1 = os.path.join(self.image_dir_bxgrid, imagename1)
                            maskpath1 = os.path.join(self.mask_dir_bxgrid, imagename1)
                            imagepath2 = os.path.join(self.image_dir_bxgrid, imagename2)
                            maskpath2 = os.path.join(self.mask_dir_bxgrid, imagename2)
                            imagepath3 = os.path.join(self.image_dir_bxgrid, imagename3)
                            maskpath3 = os.path.join(self.mask_dir_bxgrid, imagename3)
                            self.triplets.append([imagepath1, maskpath1, imagepath2, maskpath2, imagepath3, maskpath3, identifier, imposter_id])
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_1])
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_2])
                        for img_ind in range(max_pairs):
                            imposter_ind_max = 0
                            while imposter_ind_max == 0:
                                imposter_id = random.choice(bxgrid_identifiers_removed)
                                imposter_ind_max = len(self.bins_bxgrid[imposter_id][bin_num_1])
                            imagename1 = self.bins_bxgrid[identifier][bin_num_2][img_ind]
                            imagename2 = self.bins_bxgrid[identifier][bin_num_1][img_ind]
                            imagename3 = self.bins_bxgrid[imposter_id][bin_num_1][random.randrange(0, imposter_ind_max)]
                            imagepath1 = os.path.join(self.image_dir_bxgrid, imagename1)
                            maskpath1 = os.path.join(self.mask_dir_bxgrid, imagename1)
                            imagepath2 = os.path.join(self.image_dir_bxgrid, imagename2)
                            maskpath2 = os.path.join(self.mask_dir_bxgrid, imagename2)
                            imagepath3 = os.path.join(self.image_dir_bxgrid, imagename3)
                            maskpath3 = os.path.join(self.mask_dir_bxgrid, imagename3)
                            self.triplets.append([imagepath1, maskpath1, imagepath2, maskpath2, imagepath3, maskpath3, identifier, imposter_id])
                            
        print('Total triplets: ', len(self.triplets))        

class QuadrupletFromBinAllDatasetSingleCropped(Dataset):
    def __init__(self, bins_path_wsd, parent_dir_wsd, bins_path_bxgrid, parent_dir_bxgrid, direction='bs', iris_crop=True, normalize=True, brightness_and_contrast = False, flip_data = True, res_mult=1):
        super().__init__()
        self.iris_crop = iris_crop
        self.flip_data = flip_data
        self.direction = direction
        self.b_n_c = brightness_and_contrast
        self.input_size=(int(256*res_mult),int(256*res_mult))
        self.res_mult = res_mult
        self.normalize = normalize
        
        self.image_dir_wsd = os.path.join(parent_dir_wsd, 'WarsawData2')
        self.mask_dir_wsd = os.path.join(parent_dir_wsd, 'WarsawMask2')            
        
        with open(bins_path_wsd, 'rb') as wsdbinsFile:
            self.bins_wsd = pkl.load(wsdbinsFile)
        
        self.image_dir_bxgrid = os.path.join(parent_dir_bxgrid, 'iris')
        self.mask_dir_bxgrid = os.path.join(parent_dir_bxgrid, 'mask')       
        
        with open(bins_path_bxgrid, 'rb') as bxgridbinsFile:
            self.bins_bxgrid = pkl.load(bxgridbinsFile)
        
        self.all_ids = list(self.bins_wsd.keys()) + list(self.bins_bxgrid.keys())
        
        print('Initializing quadruplets for Warsaw Pupil Dynamics...')
        self.quadruplets = []
        self.genuine_ids = []
        self.imposter_ids = []
        wsd_identifiers = list(self.bins_wsd.keys())
        random.shuffle(wsd_identifiers)
        for identifier in tqdm(wsd_identifiers):
            wsd_identifiers_removed = wsd_identifiers.copy()
            wsd_identifiers_removed.remove(identifier)
            bin_numbers1 = list(self.bins_wsd[identifier].keys())
            bin_numbers2 = list(self.bins_wsd[identifier].keys())
            bin_numbers1.sort()
            bin_numbers2.sort()
            for bin_num_1 in bin_numbers1:
                for bin_num_2 in bin_numbers2:
                    check = bin_num_1 > bin_num_2 if self.direction == 'bs' else bin_num_1 < bin_num_2
                    if check:
                        random.shuffle(self.bins_wsd[identifier][bin_num_1])
                        random.shuffle(self.bins_wsd[identifier][bin_num_2])
                        max_pairs = min(len(self.bins_wsd[identifier][bin_num_1]), len(self.bins_wsd[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            imposter_ind_max = 0
                            while imposter_ind_max == 0:
                                imposter_id = random.choice(wsd_identifiers_removed)
                                imposter_ind_max = len(self.bins_wsd[imposter_id][bin_num_2])
                            imagename1 = self.bins_wsd[identifier][bin_num_1][img_ind]
                            imagename2 = self.bins_wsd[identifier][bin_num_2][img_ind]
                            imagename3 = self.bins_wsd[imposter_id][bin_num_2][random.randrange(0, imposter_ind_max)]
                            imagename4 = self.bins_wsd[identifier][bin_num_2][random.randrange(0, len(self.bins_wsd[identifier][bin_num_2]))]
                            indiv_id1 = imagename1.split('_')[0]
                            indiv_id2 = imagename2.split('_')[0]
                            indiv_id3 = imagename3.split('_')[0]
                            indiv_id4 = imagename4.split('_')[0]
                            imagepath1 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id1), imagename1)
                            maskpath1 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id1), imagename1)
                            imagepath2 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id2), imagename2)
                            maskpath2 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id2), imagename2)
                            imagepath3 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id3), imagename3)
                            maskpath3 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id3), imagename3)
                            imagepath4 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id4), imagename4)
                            maskpath4 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id4), imagename4)
                            self.quadruplets.append([imagepath1, maskpath1, imagepath2, maskpath2, imagepath3, maskpath3, imagepath4, maskpath4, identifier, imposter_id])             
        
        print('Initializing quadruplets for BXGRID...')
        bxgrid_identifiers = list(self.bins_bxgrid.keys())
        random.shuffle(bxgrid_identifiers)
        for identifier in tqdm(bxgrid_identifiers):
            bxgrid_identifiers_removed = bxgrid_identifiers.copy()
            bxgrid_identifiers_removed.remove(identifier)
            bin_numbers1 = list(self.bins_bxgrid[identifier].keys())
            bin_numbers2 = list(self.bins_bxgrid[identifier].keys())
            bin_numbers1.sort()
            bin_numbers2.sort()
            for bin_num_1 in bin_numbers1:
                for bin_num_2 in bin_numbers2:
                    check = bin_num_1 > bin_num_2 if self.direction == 'bs' else bin_num_1 < bin_num_2
                    if check:
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_1])
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_2])
                        max_pairs = min(len(self.bins_bxgrid[identifier][bin_num_1]), len(self.bins_bxgrid[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            imposter_ind_max = 0
                            while imposter_ind_max == 0:
                                imposter_id = random.choice(bxgrid_identifiers_removed)
                                imposter_ind_max = len(self.bins_bxgrid[imposter_id][bin_num_2])
                            imagename1 = self.bins_bxgrid[identifier][bin_num_1][img_ind]
                            imagename2 = self.bins_bxgrid[identifier][bin_num_2][img_ind]
                            imagename3 = self.bins_bxgrid[imposter_id][bin_num_2][random.randrange(0, imposter_ind_max)]
                            imagename4 = self.bins_bxgrid[identifier][bin_num_2][random.randrange(0, len(self.bins_bxgrid[identifier][bin_num_2]))]
                            imagepath1 = os.path.join(self.image_dir_bxgrid, imagename1)
                            maskpath1 = os.path.join(self.mask_dir_bxgrid, imagename1)
                            imagepath2 = os.path.join(self.image_dir_bxgrid, imagename2)
                            maskpath2 = os.path.join(self.mask_dir_bxgrid, imagename2)
                            imagepath3 = os.path.join(self.image_dir_bxgrid, imagename3)
                            maskpath3 = os.path.join(self.mask_dir_bxgrid, imagename3)
                            imagepath4 = os.path.join(self.image_dir_bxgrid, imagename4)
                            maskpath4 = os.path.join(self.mask_dir_bxgrid, imagename4)
                            self.quadruplets.append([imagepath1, maskpath1, imagepath2, maskpath2, imagepath3, maskpath3, imagepath4, maskpath4, identifier, imposter_id])
                            
        print('Total quadruplets: ', len(self.quadruplets))
        

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5, ))
        ])
        
        self.pupil_iris_xyrs_path_wsd = os.path.join(parent_dir_wsd, 'pupil_iris_xyrs_new.pkl')
        with open(self.pupil_iris_xyrs_path_wsd, 'rb') as pupilirisxyrsFile:
            self.pupil_iris_xyrs_wsd = pkl.load(pupilirisxyrsFile)
        
        self.pupil_iris_xyrs_path_bxgrid = os.path.join(parent_dir_bxgrid, 'pupil_iris_xyrs_new.pkl')
        with open(self.pupil_iris_xyrs_path_bxgrid, 'rb') as pupilirisxyrsFile:
            self.pupil_iris_xyrs_bxgrid = pkl.load(pupilirisxyrsFile)
        
        self.pupil_xyrs = {}
        self.iris_xyrs = {}
        
        for imagename in self.pupil_iris_xyrs_wsd.keys():
            indiv_id = imagename.split('_')[0]
            imagepath = os.path.join(os.path.join(self.image_dir_wsd, indiv_id), imagename)
            self.pupil_xyrs[imagepath] = self.pupil_iris_xyrs_wsd[imagename]['pxyr']
            self.iris_xyrs[imagepath] = self.pupil_iris_xyrs_wsd[imagename]['ixyr']
        
        for imagename in self.pupil_iris_xyrs_bxgrid.keys():
            imagepath = os.path.join(self.image_dir_bxgrid, imagename)
            self.pupil_xyrs[imagepath] = self.pupil_iris_xyrs_bxgrid[imagename]['pxyr']
            self.iris_xyrs[imagepath] = self.pupil_iris_xyrs_bxgrid[imagename]['ixyr']
        
        self.heavy_augment_prob = 0.3
    
    def load_image(self, file):
        return Image.open(file).convert('L')
        
    def __getitem__(self, index):
    
        quadruplet = self.quadruplets[index]
        
        inp_img_path = quadruplet[0].strip()
        inp_mask_path = quadruplet[1].strip()
        tar_img_path = quadruplet[2].strip()
        tar_mask_path = quadruplet[3].strip()
        imp_img_path = quadruplet[4].strip()
        imp_mask_path = quadruplet[5].strip()
        tar2_img_path = quadruplet[6].strip()
        tar2_mask_path = quadruplet[7].strip()
        
        inp_img = self.load_image(inp_img_path)
        inp_mask = self.load_image(inp_mask_path)
        
        tar_img = self.load_image(tar_img_path)
        tar_mask = self.load_image(tar_mask_path)
        
        imp_img = self.load_image(imp_img_path)
        imp_mask = self.load_image(imp_mask_path)
        
        tar2_img = self.load_image(tar2_img_path)
        tar2_mask = self.load_image(tar2_mask_path)
        
        inp_img_pupil_xyr = torch.FloatTensor(self.pupil_xyrs[inp_img_path])
        inp_img_iris_xyr = torch.FloatTensor(self.iris_xyrs[inp_img_path])
        
        tar_img_pupil_xyr = torch.FloatTensor(self.pupil_xyrs[tar_img_path])
        tar_img_iris_xyr = torch.FloatTensor(self.iris_xyrs[tar_img_path])
        
        imp_img_pupil_xyr = torch.FloatTensor(self.pupil_xyrs[imp_img_path])
        imp_img_iris_xyr = torch.FloatTensor(self.iris_xyrs[imp_img_path])
        
        tar2_img_pupil_xyr = torch.FloatTensor(self.pupil_xyrs[tar2_img_path])
        tar2_img_iris_xyr = torch.FloatTensor(self.iris_xyrs[tar2_img_path])
        
        '''
        sw, sh = inp_img.size
        sw_mult = (sw/320)
        sh_mult = (self.input_size[0]/sh)
        
        bw, bh = tar_img.size
        bw_mult = (self.input_size[1]/bw)
        bh_mult = (self.input_size[0]/bh)
        
        biw, bih = imp_img.size
        biw_mult = (self.input_size[1]/biw)
        bih_mult = (self.input_size[0]/bih)
        
        inp_img_pupil_xyr[0] = (inp_img_pupil_xyr[0] * sw_mult)
        inp_img_pupil_xyr[1] = (inp_img_pupil_xyr[1] * sh_mult)
        inp_img_pupil_xyr[2] = (inp_img_pupil_xyr[2] * max(sw_mult, sh_mult))
        
        tar_img_pupil_xyr[0] = (tar_img_pupil_xyr[0] * bw_mult)
        tar_img_pupil_xyr[1] = (tar_img_pupil_xyr[1] * bh_mult)
        tar_img_pupil_xyr[2] = (tar_img_pupil_xyr[2] * max(bw_mult, bh_mult))
        
        imp_img_pupil_xyr[0] = (imp_img_pupil_xyr[0] * biw_mult)
        imp_img_pupil_xyr[1] = (imp_img_pupil_xyr[1] * bih_mult)
        imp_img_pupil_xyr[2] = (imp_img_pupil_xyr[2] * max(biw_mult, bih_mult))
    
        inp_img_iris_xyr[0] = (inp_img_iris_xyr[0] * sw_mult)
        inp_img_iris_xyr[1] = (inp_img_iris_xyr[1] * sh_mult)
        inp_img_iris_xyr[2] = (inp_img_iris_xyr[2] * max(sw_mult, sh_mult))
        
        tar_img_iris_xyr[0] = (tar_img_iris_xyr[0] * bw_mult)
        tar_img_iris_xyr[1] = (tar_img_iris_xyr[1] * bh_mult)
        tar_img_iris_xyr[2] = (tar_img_iris_xyr[2] * max(bw_mult, bh_mult))
        
        imp_img_iris_xyr[0] = (imp_img_iris_xyr[0] * biw_mult)
        imp_img_iris_xyr[1] = (imp_img_iris_xyr[1] * bih_mult)
        imp_img_iris_xyr[2] = (imp_img_iris_xyr[2] * max(biw_mult, bih_mult))
        '''
        sw, sh = inp_img.size
        bw, bh = tar_img.size
        bw2, bh2 = tar2_img.size
        iw, ih = imp_img.size
        
        if self.flip_data and random.random() < 0.5:
            inp_img = inp_img.transpose(Image.FLIP_LEFT_RIGHT) 
            inp_mask = inp_mask.transpose(Image.FLIP_LEFT_RIGHT)
            tar_img = tar_img.transpose(Image.FLIP_LEFT_RIGHT) 
            tar_mask = tar_mask.transpose(Image.FLIP_LEFT_RIGHT)
            tar2_img = tar2_img.transpose(Image.FLIP_LEFT_RIGHT) 
            tar2_mask = tar2_mask.transpose(Image.FLIP_LEFT_RIGHT)  
            inp_img_pupil_xyr[0] = sw - inp_img_pupil_xyr[0]
            tar_img_pupil_xyr[0] = bw - tar_img_pupil_xyr[0]
            tar2_img_pupil_xyr[0] = bw2 - tar2_img_pupil_xyr[0]
            inp_img_iris_xyr[0] = sw - inp_img_iris_xyr[0]
            tar_img_iris_xyr[0] = bw - tar_img_iris_xyr[0]
            tar2_img_iris_xyr[0] = bw2 - tar2_img_iris_xyr[0]
        
        if self.flip_data and random.random() < 0.5:
            imp_img = imp_img.transpose(Image.FLIP_LEFT_RIGHT)
            imp_mask = imp_mask.transpose(Image.FLIP_LEFT_RIGHT)
            imp_img_pupil_xyr[0] = iw - imp_img_pupil_xyr[0]
            imp_img_iris_xyr[0] = iw - imp_img_iris_xyr[0]
        
        if self.iris_crop:
            #inp_img.save('./check1.png')
            #print((inp_img_iris_xyr[0].item() - inp_img_iris_xyr[2].item(), inp_img_iris_xyr[1].item() - inp_img_iris_xyr[2].item(), inp_img_iris_xyr[0].item() + inp_img_iris_xyr[2].item(), inp_img_iris_xyr[1].item() + inp_img_iris_xyr[2].item()))
            inp_img = inp_img.crop((inp_img_iris_xyr[0].item() - inp_img_iris_xyr[2].item(), inp_img_iris_xyr[1].item() - inp_img_iris_xyr[2].item(), inp_img_iris_xyr[0].item() + inp_img_iris_xyr[2].item(), inp_img_iris_xyr[1].item() + inp_img_iris_xyr[2].item()))
            inp_mask = inp_mask.crop((inp_img_iris_xyr[0].item() - inp_img_iris_xyr[2].item(), inp_img_iris_xyr[1].item() - inp_img_iris_xyr[2].item(), inp_img_iris_xyr[0].item() + inp_img_iris_xyr[2].item(), inp_img_iris_xyr[1].item() + inp_img_iris_xyr[2].item()))
            sw_crop, sh_crop = inp_img.size
            inp_img_pupil_xyr[0] = inp_img_pupil_xyr[0] - inp_img_iris_xyr[0] + sw_crop/2
            inp_img_pupil_xyr[1] = inp_img_pupil_xyr[1] - inp_img_iris_xyr[1] + sh_crop/2
            inp_img_iris_xyr[0] = sw_crop/2
            inp_img_iris_xyr[1] = sh_crop/2            
            
            tar_img = tar_img.crop((tar_img_iris_xyr[0].item() - tar_img_iris_xyr[2].item(), tar_img_iris_xyr[1].item() - tar_img_iris_xyr[2].item(), tar_img_iris_xyr[0].item() + tar_img_iris_xyr[2].item(), tar_img_iris_xyr[1].item() + tar_img_iris_xyr[2].item()))
            tar_mask = tar_mask.crop((tar_img_iris_xyr[0].item() - tar_img_iris_xyr[2].item(), tar_img_iris_xyr[1].item() - tar_img_iris_xyr[2].item(), tar_img_iris_xyr[0].item() + tar_img_iris_xyr[2].item(), tar_img_iris_xyr[1].item() + tar_img_iris_xyr[2].item()))
            bw_crop, bh_crop = tar_img.size
            tar_img_pupil_xyr[0] = tar_img_pupil_xyr[0] - tar_img_iris_xyr[0] + bw_crop/2
            tar_img_pupil_xyr[1] = tar_img_pupil_xyr[1] - tar_img_iris_xyr[1] + bh_crop/2
            tar_img_iris_xyr[0] = bw_crop/2
            tar_img_iris_xyr[1] = bh_crop/2
            
            tar2_img = tar2_img.crop((tar2_img_iris_xyr[0].item() - tar2_img_iris_xyr[2].item(), tar2_img_iris_xyr[1].item() - tar2_img_iris_xyr[2].item(), tar2_img_iris_xyr[0].item() + tar2_img_iris_xyr[2].item(), tar2_img_iris_xyr[1].item() + tar2_img_iris_xyr[2].item()))
            tar2_mask = tar2_mask.crop((tar2_img_iris_xyr[0].item() - tar2_img_iris_xyr[2].item(), tar2_img_iris_xyr[1].item() - tar2_img_iris_xyr[2].item(), tar2_img_iris_xyr[0].item() + tar2_img_iris_xyr[2].item(), tar2_img_iris_xyr[1].item() + tar2_img_iris_xyr[2].item()))
            bw2_crop, bh2_crop = tar2_img.size
            tar2_img_pupil_xyr[0] = tar2_img_pupil_xyr[0] - tar2_img_iris_xyr[0] + bw2_crop/2
            tar2_img_pupil_xyr[1] = tar2_img_pupil_xyr[1] - tar2_img_iris_xyr[1] + bh2_crop/2
            tar2_img_iris_xyr[0] = bw2_crop/2
            tar2_img_iris_xyr[1] = bh2_crop/2
            
            imp_img = imp_img.crop((imp_img_iris_xyr[0].item() - imp_img_iris_xyr[2].item(), imp_img_iris_xyr[1].item() - imp_img_iris_xyr[2].item(), imp_img_iris_xyr[0].item() + imp_img_iris_xyr[2].item(), imp_img_iris_xyr[1].item() + imp_img_iris_xyr[2].item()))
            imp_mask = imp_mask.crop((imp_img_iris_xyr[0].item() - imp_img_iris_xyr[2].item(), imp_img_iris_xyr[1].item() - imp_img_iris_xyr[2].item(), imp_img_iris_xyr[0].item() + imp_img_iris_xyr[2].item(), imp_img_iris_xyr[1].item() + imp_img_iris_xyr[2].item()))
            iw_crop, ih_crop = imp_img.size
            imp_img_pupil_xyr[0] = imp_img_pupil_xyr[0] - imp_img_iris_xyr[0] + iw_crop/2
            imp_img_pupil_xyr[1] = imp_img_pupil_xyr[1] - imp_img_iris_xyr[1] + ih_crop/2
            imp_img_iris_xyr[0] = iw_crop/2
            imp_img_iris_xyr[1] = ih_crop/2
            
            '''
            inp_img_vis = np.uint8(inp_img.convert('RGB'))
            try:
                inp_img_vis = cv2.circle(inp_img_vis, (int(inp_img_pupil_xyr[0].item()),int(inp_img_pupil_xyr[1].item())), int(inp_img_pupil_xyr[2].item()), (0, 0, 255), 2)
            except:
                print("Pupil circle could not be visualized, values are: ", (int(inp_img_pupil_xyr[0].item()),int(inp_img_pupil_xyr[1].item()), int(inp_img_pupil_xyr[2].item())), ' ', inp_img_vis.shape)
                pass
            try:
                inp_img_vis = cv2.circle(inp_img_vis, (int(inp_img_iris_xyr[0].item()),int(inp_img_iris_xyr[1].item())), int(inp_img_iris_xyr[2].item()), (255, 0, 0), 2)
            except:
                print("Iris circle could not be visualized, values are: ", (int(inp_img_iris_xyr[0].item()),int(inp_img_iris_xyr[1].item()), int(inp_img_iris_xyr[2].item())), ' ', inp_img_vis.shape)
            
            Image.fromarray(inp_img_vis).save('./check2.png')
            '''
            
            sh_mult = self.input_size[0]/sh_crop
            sw_mult = self.input_size[1]/sw_crop
            inp_img = inp_img.resize((self.input_size[1], self.input_size[0]))
            inp_mask = inp_mask.resize((self.input_size[1], self.input_size[0]))
            inp_img_pupil_xyr[0] = self.input_size[1]/2 - sw_mult * (sw_crop/2 - inp_img_pupil_xyr[0])
            inp_img_pupil_xyr[1] = self.input_size[0]/2 - sh_mult * (sh_crop/2 - inp_img_pupil_xyr[1])
            inp_img_pupil_xyr[2] = max(sw_mult, sh_mult) * inp_img_pupil_xyr[2]
            inp_img_iris_xyr[0] = self.input_size[1]/2
            inp_img_iris_xyr[1] = self.input_size[0]/2
            inp_img_iris_xyr[2] = max(sw_mult, sh_mult) * inp_img_iris_xyr[2]
            
            bh_mult = self.input_size[0]/bh_crop
            bw_mult = self.input_size[1]/bw_crop
            tar_img = tar_img.resize((self.input_size[1], self.input_size[0]))
            tar_mask = tar_mask.resize((self.input_size[1], self.input_size[0]))
            tar_img_pupil_xyr[0] = self.input_size[1]/2 - bw_mult * (bw_crop/2 - tar_img_pupil_xyr[0])
            tar_img_pupil_xyr[1] = self.input_size[0]/2 - bh_mult * (bh_crop/2 - tar_img_pupil_xyr[1])
            tar_img_pupil_xyr[2] = max(bw_mult, bh_mult) * tar_img_pupil_xyr[2]
            tar_img_iris_xyr[0] = self.input_size[1]/2
            tar_img_iris_xyr[1] = self.input_size[0]/2
            tar_img_iris_xyr[2] = max(bw_mult, bh_mult) * tar_img_iris_xyr[2]
            
            bh2_mult = self.input_size[0]/bh2_crop
            bw2_mult = self.input_size[1]/bw2_crop
            tar2_img = tar2_img.resize((self.input_size[1], self.input_size[0]))
            tar2_mask = tar2_mask.resize((self.input_size[1], self.input_size[0]))
            tar2_img_pupil_xyr[0] = self.input_size[1]/2 - bw2_mult * (bw2_crop/2 - tar2_img_pupil_xyr[0])
            tar2_img_pupil_xyr[1] = self.input_size[0]/2 - bh2_mult * (bh2_crop/2 - tar2_img_pupil_xyr[1])
            tar2_img_pupil_xyr[2] = max(bw2_mult, bh2_mult) * tar2_img_pupil_xyr[2]
            tar2_img_iris_xyr[0] = self.input_size[1]/2
            tar2_img_iris_xyr[1] = self.input_size[0]/2
            tar2_img_iris_xyr[2] = max(bw2_mult, bh2_mult) * tar2_img_iris_xyr[2]
            
            ih_mult = self.input_size[0]/ih_crop
            iw_mult = self.input_size[1]/iw_crop
            imp_img = imp_img.resize((self.input_size[0], self.input_size[1]))
            imp_mask = imp_mask.resize((self.input_size[0], self.input_size[1]))
            imp_img_pupil_xyr[0] = self.input_size[1]/2 - iw_mult * (iw_crop/2 - imp_img_pupil_xyr[0])
            imp_img_pupil_xyr[1] = self.input_size[0]/2 - ih_mult * (ih_crop/2 - imp_img_pupil_xyr[1])
            imp_img_pupil_xyr[2] = max(iw_mult, ih_mult) * imp_img_pupil_xyr[2]
            imp_img_iris_xyr[0] = self.input_size[1]/2
            imp_img_iris_xyr[1] = self.input_size[0]/2
            imp_img_iris_xyr[2] = max(iw_mult, ih_mult) * imp_img_iris_xyr[2]
            
            #print('Saving ..')
            '''
            inp_img_vis = np.uint8(inp_img.convert('RGB'))
            try:
                inp_img_vis = cv2.circle(inp_img_vis, (int(inp_img_pupil_xyr[0].item()),int(inp_img_pupil_xyr[1].item())), int(inp_img_pupil_xyr[2].item()), (0, 0, 255), 2)
            except:
                print("Pupil circle could not be visualized, values are: ", (int(inp_img_pupil_xyr[0].item()),int(inp_img_pupil_xyr[1].item()), int(inp_img_pupil_xyr[2].item())), ' ', inp_img_vis.shape)
                pass
            try:
                inp_img_vis = cv2.circle(inp_img_vis, (int(inp_img_iris_xyr[0].item()),int(inp_img_iris_xyr[1].item())), int(inp_img_iris_xyr[2].item()), (255, 0, 0), 2)
            except:
                print("Iris circle could not be visualized, values are: ", (int(inp_img_iris_xyr[0].item()),int(inp_img_iris_xyr[1].item()), int(inp_img_iris_xyr[2].item())), ' ', inp_img_vis.shape)
            
            Image.fromarray(inp_img_vis).save('./check3.png')
            '''
        
        if random.random() < self.heavy_augment_prob:
            if random.random() < 0.5:
                random_choice = np.random.choice([1,2,3])
                if random_choice == 1:   
                    # sharpening
                    random_degree = np.random.choice([1,2,3,4,5])
                    if random_degree == 1:
                        inp_img = inp_img.filter(ImageFilter.DETAIL)
                        tar_img = tar_img.filter(ImageFilter.DETAIL)
                    elif random_degree == 2:
                        inp_img = inp_img.filter(ImageFilter.SHARPEN)
                        tar_img = tar_img.filter(ImageFilter.SHARPEN)
                    elif random_degree == 3:
                        inp_img = inp_img.filter(ImageFilter.EDGE_ENHANCE)
                        tar_img = tar_img.filter(ImageFilter.EDGE_ENHANCE)
                    elif random_degree == 4:
                        inp_img = inp_img.filter(ImageFilter.EDGE_ENHANCE_MORE)
                        tar_img = tar_img.filter(ImageFilter.EDGE_ENHANCE_MORE)                          
                    else:
                        random_alpha = random.uniform(0.0, 0.3)
                        random_lightness = random.uniform(0.6, 1.0)
                        aug = iaa.Sharpen(alpha=random_alpha, lightness=random_lightness)
                        inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                        inp_img = Image.fromarray(inp_img_np[0])
                        tar_img_np = aug(images = np.expand_dims(np.array(tar_img), axis=0))
                        tar_img = Image.fromarray(tar_img_np[0])
                elif random_choice == 2:
                    #blurring
                    random_degree = np.random.choice([1,2,3,4,5])
                    if random_degree == 1:
                        inp_img = inp_img.filter(ImageFilter.GaussianBlur())
                        tar_img = tar_img.filter(ImageFilter.GaussianBlur())
                    elif random_degree == 2:
                        inp_img = inp_img.filter(ImageFilter.BLUR)
                        tar_img = tar_img.filter(ImageFilter.BLUR)
                    elif random_degree == 3:
                        inp_img = inp_img.filter(ImageFilter.SMOOTH)
                        tar_img = tar_img.filter(ImageFilter.SMOOTH)
                    elif random_degree == 4:
                        inp_img = inp_img.filter(ImageFilter.SMOOTH_MORE)
                        tar_img = tar_img.filter(ImageFilter.SMOOTH_MORE)
                    else:
                        random_degree = random.randint(-179, 180)
                        aug = iaa.MotionBlur(k=3, angle=random_degree)
                        inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                        inp_img = Image.fromarray(inp_img_np[0])
                        tar_img_np = aug(images = np.expand_dims(np.array(tar_img), axis=0))
                        tar_img = Image.fromarray(tar_img_np[0])
                else:
                    if random.random() < 0.5:
                        aug = iaa.AdditiveGaussianNoise(scale=3)
                        inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                        inp_img = Image.fromarray(inp_img_np[0])
                        tar_img_np = aug(images = np.expand_dims(np.array(tar_img), axis=0))
                        tar_img = Image.fromarray(tar_img_np[0])
                    else:
                        aug = iaa.AdditiveLaplaceNoise(scale=3)
                        inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                        inp_img = Image.fromarray(inp_img_np[0])
                        tar_img_np = aug(images = np.expand_dims(np.array(tar_img), axis=0))
                        tar_img = Image.fromarray(tar_img_np[0])
            
            if random.random() < 0.5:
                if random.random() < 0.5:
                    # Basic compression and expansion
                    divider = random.uniform(1.0, 1.5)
                    cw, ch = inp_img.size
                    new_cw = int(cw/divider)
                    new_ch = int(ch/divider)
                    
                    first_choice = np.random.choice([1,2,3,4,5,6])
                    if first_choice == 1:
                        inp_img = inp_img.resize((new_cw, new_ch), Image.NEAREST)
                        tar_img = tar_img.resize((new_cw, new_ch), Image.NEAREST)
                    elif first_choice == 2:
                        inp_img = inp_img.resize((new_cw, new_ch), Image.BILINEAR)
                        tar_img = tar_img.resize((new_cw, new_ch), Image.BILINEAR)
                    elif first_choice == 3:
                        inp_img = inp_img.resize((new_cw, new_ch), Image.BICUBIC)
                        tar_img = tar_img.resize((new_cw, new_ch), Image.BICUBIC)
                    elif first_choice == 4:
                        inp_img = inp_img.resize((new_cw, new_ch), Image.LANCZOS)
                        tar_img = tar_img.resize((new_cw, new_ch), Image.LANCZOS)
                    elif first_choice == 5:
                        inp_img = inp_img.resize((new_cw, new_ch), Image.HAMMING)
                        tar_img = tar_img.resize((new_cw, new_ch), Image.HAMMING)
                    else:
                        inp_img = inp_img.resize((new_cw, new_ch), Image.BOX)
                        tar_img = tar_img.resize((new_cw, new_ch), Image.BOX)
                    
                    second_choice = np.random.choice([1,2,3,4,5,6])
                    if second_choice == 1:
                        inp_img = inp_img.resize((cw, ch), Image.NEAREST)
                        tar_img = tar_img.resize((cw, ch), Image.NEAREST)
                    elif second_choice == 2:
                        inp_img = inp_img.resize((cw, ch), Image.BILINEAR)
                        tar_img = tar_img.resize((cw, ch), Image.BILINEAR)
                    elif second_choice == 3:
                        inp_img = inp_img.resize((cw, ch), Image.BICUBIC)
                        tar_img = tar_img.resize((cw, ch), Image.BICUBIC)
                    elif second_choice == 4:
                        inp_img = inp_img.resize((cw, ch), Image.LANCZOS)
                        tar_img = tar_img.resize((cw, ch), Image.LANCZOS)
                    elif second_choice == 5:
                        inp_img = inp_img.resize((cw, ch), Image.HAMMING)
                        tar_img = tar_img.resize((cw, ch), Image.HAMMING)
                    else:
                        inp_img = inp_img.resize((cw, ch), Image.BOX)
                        tar_img = tar_img.resize((cw, ch), Image.BOX)
                else:
                    #JPEG compression
                    random_compression = random.randint(10, 50)
                    aug = iaa.JpegCompression(compression=random_compression)
                    inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                    inp_img = Image.fromarray(inp_img_np[0])
                    tar_img_np = aug(images = np.expand_dims(np.array(tar_img), axis=0))
                    tar_img = Image.fromarray(tar_img_np[0])
                         
            if random.random() < 0.5: # random contrast change
                random_degree = np.random.choice([1,2,3,4,5,6,7,8])                
                if random_degree == 1:
                    aug = iaa.GammaContrast((0.5, 2.0))
                elif random_degree == 2:
                    aug = iaa.LinearContrast((0.4, 1.6))
                elif random_degree == 3:
                    aug = iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6))
                elif random_degree == 4:
                    aug = iaa.LogContrast(gain=(0.6, 1.4)) 
                else:    
                    aug = iaa.pillike.EnhanceBrightness()
                inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                inp_img = Image.fromarray(inp_img_np[0])
                random_degree = np.random.choice([1,2,3,4,5,6,7,8])                
                if random_degree == 1:
                    aug = iaa.GammaContrast((0.5, 2.0))
                elif random_degree == 2:
                    aug = iaa.LinearContrast((0.4, 1.6))
                elif random_degree == 3:
                    aug = iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6))
                elif random_degree == 4:
                    aug = iaa.LogContrast(gain=(0.6, 1.4)) 
                else:    
                    aug = iaa.pillike.EnhanceBrightness()
                tar_img_np = aug(images = np.expand_dims(np.array(tar_img), axis=0))
                tar_img = Image.fromarray(tar_img_np[0])
        
        if random.random() < self.heavy_augment_prob:
            if random.random() < 0.5:
                random_choice = np.random.choice([1,2,3])
                if random_choice == 1:   
                    # sharpening
                    random_degree = np.random.choice([1,2,3,4,5])
                    if random_degree == 1:
                        imp_img = imp_img.filter(ImageFilter.DETAIL)
                    elif random_degree == 2:
                        imp_img = imp_img.filter(ImageFilter.SHARPEN)
                    elif random_degree == 3:
                        imp_img = imp_img.filter(ImageFilter.EDGE_ENHANCE)
                    elif random_degree == 4:
                        imp_img = imp_img.filter(ImageFilter.EDGE_ENHANCE_MORE)                            
                    else:
                        aug = iaa.Sharpen(alpha=(0.0, 0.3), lightness=(0.6, 1.0))
                        imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                        imp_img = Image.fromarray(imp_img_np[0])
                elif random_choice == 2:
                    #blurring
                    random_degree = np.random.choice([1,2,3,4,5])
                    if random_degree == 1:
                        imp_img = imp_img.filter(ImageFilter.GaussianBlur())
                    elif random_degree == 2:
                        imp_img = imp_img.filter(ImageFilter.BLUR)
                    elif random_degree == 3:
                        imp_img = imp_img.filter(ImageFilter.SMOOTH)
                    elif random_degree == 4:
                        imp_img = imp_img.filter(ImageFilter.SMOOTH_MORE)
                    else:
                        aug = iaa.MotionBlur(k=3)
                        imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                        imp_img = Image.fromarray(imp_img_np[0])
                else:
                    if random.random() < 0.5:
                        aug = iaa.AdditiveGaussianNoise(scale=3)
                        imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                        imp_img = Image.fromarray(imp_img_np[0])
                    else:
                        aug = iaa.AdditiveLaplaceNoise(scale=3)
                        imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                        imp_img = Image.fromarray(imp_img_np[0])
            if random.random() < 0.5:
                if random.random() < 0.5:
                    # Basic compression and expansion
                    divider = random.uniform(1.0, 1.5)
                    cw, ch = imp_img.size
                    new_cw = int(cw/divider)
                    new_ch = int(ch/divider)
                    
                    first_choice = np.random.choice([1,2,3,4,5,6])
                    if first_choice == 1:
                        imp_img = imp_img.resize((new_cw, new_ch), Image.NEAREST)
                    elif first_choice == 2:
                        imp_img = imp_img.resize((new_cw, new_ch), Image.BILINEAR)
                    elif first_choice == 3:
                        imp_img = imp_img.resize((new_cw, new_ch), Image.BICUBIC)
                    elif first_choice == 4:
                        imp_img = imp_img.resize((new_cw, new_ch), Image.LANCZOS)
                    elif first_choice == 5:
                        imp_img = imp_img.resize((new_cw, new_ch), Image.HAMMING)
                    else:
                        imp_img = imp_img.resize((new_cw, new_ch), Image.BOX)
                    
                    second_choice = np.random.choice([1,2,3,4,5,6])
                    if second_choice == 1:
                        imp_img = imp_img.resize((cw, ch), Image.NEAREST)
                    elif second_choice == 2:
                        imp_img = imp_img.resize((cw, ch), Image.BILINEAR)
                    elif second_choice == 3:
                        imp_img = imp_img.resize((cw, ch), Image.BICUBIC)
                    elif second_choice == 4:
                        imp_img = imp_img.resize((cw, ch), Image.LANCZOS)
                    elif second_choice == 5:
                        imp_img = imp_img.resize((cw, ch), Image.HAMMING)
                    else:
                        imp_img = imp_img.resize((cw, ch), Image.BOX)
                else:
                    #JPEG compression
                    aug = iaa.JpegCompression(compression=(10, 30))
                    imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                    imp_img = Image.fromarray(imp_img_np[0])
                         
            if random.random() < 0.5: # random contrast change
                random_degree = np.random.choice([1,2,3,4,5,6,7,8])                
                if random_degree == 1:
                    aug = iaa.GammaContrast((0.5, 2.0))
                    imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                    imp_img = Image.fromarray(imp_img_np[0])
                elif random_degree == 2:
                    aug = iaa.LinearContrast((0.4, 1.6))
                    imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                    imp_img = Image.fromarray(imp_img_np[0])
                elif random_degree == 3:
                    aug = iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6))
                    imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                    imp_img = Image.fromarray(imp_img_np[0])
                elif random_degree == 4:
                    aug = iaa.LogContrast(gain=(0.6, 1.4))
                    imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                    imp_img = Image.fromarray(imp_img_np[0])            
                else:    
                    aug = iaa.pillike.EnhanceBrightness()
                    imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                    imp_img = Image.fromarray(imp_img_np[0])         

        if self.normalize:
            inp_img_t = torch.from_numpy(np.float32(inp_img)).float()
            inp_img_mean = torch.mean(inp_img_t)
            inp_img_std = torch.std(inp_img_t)
            xform_inp_img = (inp_img_t - inp_img_mean)/inp_img_std
            xform_inp_img = xform_inp_img.unsqueeze(0)
            
            xform_inp_mask = self.transform(inp_mask)
            xform_inp_mask[xform_inp_mask<0] = -1
            xform_inp_mask[xform_inp_mask>=0] = 1
            
            tar_img_t = torch.from_numpy(np.float32(tar_img)).float()
            tar_img_mean = torch.mean(tar_img_t)
            tar_img_std = torch.std(tar_img_t)
            xform_tar_img = (tar_img_t - tar_img_mean)/tar_img_std
            xform_tar_img = xform_tar_img.unsqueeze(0)
            
            xform_tar_mask = self.transform(tar_mask)
            xform_tar_mask[xform_tar_mask<0] = -1.0
            xform_tar_mask[xform_tar_mask>=0] = 1.0
            
            tar2_img_t = torch.from_numpy(np.float32(tar2_img)).float()
            tar2_img_mean = torch.mean(tar2_img_t)
            tar2_img_std = torch.std(tar2_img_t)
            xform_tar2_img = (tar2_img_t - tar2_img_mean)/tar2_img_std
            xform_tar2_img = xform_tar2_img.unsqueeze(0)
            
            xform_tar2_mask = self.transform(tar2_mask)
            xform_tar2_mask[xform_tar2_mask<0] = -1.0
            xform_tar2_mask[xform_tar2_mask>=0] = 1.0
            
            imp_img_t = torch.from_numpy(np.float32(imp_img)).float()
            imp_img_mean = torch.mean(imp_img_t)
            imp_img_std = torch.std(imp_img_t)
            xform_imp_img = (imp_img_t - imp_img_mean)/imp_img_std
            xform_imp_img = xform_imp_img.unsqueeze(0)
            
            xform_imp_mask = self.transform(imp_mask)
            xform_imp_mask[xform_imp_mask<0] = -1.0
            xform_imp_mask[xform_imp_mask>=0] = 1.0
            return {'inp_img': xform_inp_img, 'inp_img_mean': inp_img_mean, 'inp_img_std':inp_img_std, 'tar_img' : xform_tar_img, 'tar_img_mean':tar_img_mean, 'tar_img_std':tar_img_std, 'tar2_img' : xform_tar2_img, 'tar2_img_mean':tar2_img_mean, 'tar2_img_std':tar2_img_std, 'imp_img' : xform_imp_img, 'imp_img_mean': imp_img_mean, 'imp_img_std':imp_img_std, 'inp_mask': xform_inp_mask, 'tar_mask' : xform_tar_mask, 'tar2_mask' : xform_tar2_mask, 'imp_mask' : xform_imp_mask, 'inp_img_pxyr' : inp_img_pupil_xyr, 'tar_img_pxyr' : tar_img_pupil_xyr, 'tar2_img_pxyr' : tar2_img_pupil_xyr, 'imp_img_pxyr' : imp_img_pupil_xyr, 'inp_img_ixyr': inp_img_iris_xyr, 'tar_img_ixyr' :  tar_img_iris_xyr, 'tar2_img_ixyr' :  tar2_img_iris_xyr, 'imp_img_ixyr' : imp_img_iris_xyr, 'identifier' : quadruplet[8], 'imposter_id' : quadruplet[9]}

        else:
            xform_inp_img = self.transform(inp_img)
            xform_inp_mask = self.transform(inp_mask)
            xform_inp_mask[xform_inp_mask<0] = -1
            xform_inp_mask[xform_inp_mask>=0] = 1
            
            xform_tar_img = self.transform(tar_img)
            xform_tar_mask = self.transform(tar_mask)
            xform_tar_mask[xform_tar_mask<0] = -1.0
            xform_tar_mask[xform_tar_mask>=0] = 1.0
            
            xform_tar2_img = self.transform(tar2_img)
            xform_tar2_mask = self.transform(tar2_mask)
            xform_tar2_mask[xform_tar2_mask<0] = -1.0
            xform_tar2_mask[xform_tar2_mask>=0] = 1.0
            
            xform_imp_img = self.transform(imp_img)
            xform_imp_mask = self.transform(imp_mask)
            xform_imp_mask[xform_imp_mask<0] = -1.0
            xform_imp_mask[xform_imp_mask>=0] = 1.0
            return {'inp_img': xform_inp_img, 'tar_img' : xform_tar_img, 'tar2_img' : xform_tar2_img, 'imp_img' : xform_imp_img, 'inp_mask': xform_inp_mask, 'tar_mask' : xform_tar_mask, 'tar2_mask' : xform_tar2_mask, 'imp_mask' : xform_imp_mask, 'inp_img_pxyr' : inp_img_pupil_xyr, 'tar_img_pxyr' : tar_img_pupil_xyr, 'tar2_img_pxyr' : tar2_img_pupil_xyr, 'imp_img_pxyr' : imp_img_pupil_xyr, 'inp_img_ixyr': inp_img_iris_xyr, 'tar_img_ixyr' :  tar_img_iris_xyr,  'tar2_img_ixyr' :  tar2_img_iris_xyr, 'imp_img_ixyr' : imp_img_iris_xyr, 'identifier' : quadruplet[8], 'imposter_id' : quadruplet[9]}
    
    def __len__(self):
        return len(self.quadruplets)
    
    def reset(self):
        print('Shuffling quadruplets for WSD...')
        self.quadruplets = []
        self.genuine_ids = []
        self.imposter_ids = []
        wsd_identifiers = list(self.bins_wsd.keys())
        random.shuffle(wsd_identifiers)
        for identifier in tqdm(wsd_identifiers):
            wsd_identifiers_removed = wsd_identifiers.copy()
            wsd_identifiers_removed.remove(identifier)
            bin_numbers1 = list(self.bins_wsd[identifier].keys())
            bin_numbers2 = list(self.bins_wsd[identifier].keys())
            bin_numbers1.sort()
            bin_numbers2.sort()
            for bin_num_1 in bin_numbers1:
                for bin_num_2 in bin_numbers2:
                    check = bin_num_1 > bin_num_2 if self.direction == 'bs' else bin_num_1 < bin_num_2
                    if check:
                        random.shuffle(self.bins_wsd[identifier][bin_num_1])
                        random.shuffle(self.bins_wsd[identifier][bin_num_2])
                        max_pairs = min(len(self.bins_wsd[identifier][bin_num_1]), len(self.bins_wsd[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            imposter_ind_max = 0
                            while imposter_ind_max == 0:
                                imposter_id = random.choice(wsd_identifiers_removed)
                                imposter_ind_max = len(self.bins_wsd[imposter_id][bin_num_2])
                            imagename1 = self.bins_wsd[identifier][bin_num_1][img_ind]
                            imagename2 = self.bins_wsd[identifier][bin_num_2][img_ind]
                            imagename3 = self.bins_wsd[imposter_id][bin_num_2][random.randrange(0, imposter_ind_max)]
                            imagename4 = self.bins_wsd[identifier][bin_num_2][random.randrange(0, len(self.bins_wsd[identifier][bin_num_2]))]
                            indiv_id1 = imagename1.split('_')[0]
                            indiv_id2 = imagename2.split('_')[0]
                            indiv_id3 = imagename3.split('_')[0]
                            indiv_id4 = imagename4.split('_')[0]
                            imagepath1 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id1), imagename1)
                            maskpath1 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id1), imagename1)
                            imagepath2 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id2), imagename2)
                            maskpath2 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id2), imagename2)
                            imagepath3 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id3), imagename3)
                            maskpath3 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id3), imagename3)
                            imagepath4 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id4), imagename4)
                            maskpath4 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id4), imagename4)
                            self.quadruplets.append([imagepath1, maskpath1, imagepath2, maskpath2, imagepath3, maskpath3, imagepath4, maskpath4, identifier, imposter_id])             
        
        print('Shuffling quadruplets for BXGRID...')
        bxgrid_identifiers = list(self.bins_bxgrid.keys())
        random.shuffle(bxgrid_identifiers)
        for identifier in tqdm(bxgrid_identifiers):
            bxgrid_identifiers_removed = bxgrid_identifiers.copy()
            bxgrid_identifiers_removed.remove(identifier)
            bin_numbers1 = list(self.bins_bxgrid[identifier].keys())
            bin_numbers2 = list(self.bins_bxgrid[identifier].keys())
            bin_numbers1.sort()
            bin_numbers2.sort()
            for bin_num_1 in bin_numbers1:
                for bin_num_2 in bin_numbers2:
                    check = bin_num_1 > bin_num_2 if self.direction == 'bs' else bin_num_1 < bin_num_2
                    if check:
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_1])
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_2])
                        max_pairs = min(len(self.bins_bxgrid[identifier][bin_num_1]), len(self.bins_bxgrid[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            imposter_ind_max = 0
                            while imposter_ind_max == 0:
                                imposter_id = random.choice(bxgrid_identifiers_removed)
                                imposter_ind_max = len(self.bins_bxgrid[imposter_id][bin_num_2])
                            imagename1 = self.bins_bxgrid[identifier][bin_num_1][img_ind]
                            imagename2 = self.bins_bxgrid[identifier][bin_num_2][img_ind]
                            imagename3 = self.bins_bxgrid[imposter_id][bin_num_2][random.randrange(0, imposter_ind_max)]
                            imagename4 = self.bins_bxgrid[identifier][bin_num_2][random.randrange(0, len(self.bins_bxgrid[identifier][bin_num_2]))]
                            imagepath1 = os.path.join(self.image_dir_bxgrid, imagename1)
                            maskpath1 = os.path.join(self.mask_dir_bxgrid, imagename1)
                            imagepath2 = os.path.join(self.image_dir_bxgrid, imagename2)
                            maskpath2 = os.path.join(self.mask_dir_bxgrid, imagename2)
                            imagepath3 = os.path.join(self.image_dir_bxgrid, imagename3)
                            maskpath3 = os.path.join(self.mask_dir_bxgrid, imagename3)
                            imagepath4 = os.path.join(self.image_dir_bxgrid, imagename4)
                            maskpath4 = os.path.join(self.mask_dir_bxgrid, imagename4)
                            self.quadruplets.append([imagepath1, maskpath1, imagepath2, maskpath2, imagepath3, maskpath3, imagepath4, maskpath4, identifier, imposter_id])
                            
        print('Total quadruplets: ', len(self.quadruplets))
        
        
class QuadrupletFromBinAllDatasetBothCropped(Dataset):
    def __init__(self, bins_path_wsd, parent_dir_wsd, bins_path_bxgrid, parent_dir_bxgrid, iris_crop = True, normalize = True, brightness_and_contrast = False, flip_data = True, res_mult = 1):
        super().__init__()
        self.iris_crop = iris_crop
        self.flip_data = flip_data
        self.b_n_c = brightness_and_contrast
        self.input_size=(int(256*res_mult),int(256*res_mult))
        self.input_size_nocrop = (int(480*res_mult), int(640*res_mult))
        self.res_mult = res_mult
        self.normalize = normalize
        
        self.image_dir_wsd = os.path.join(parent_dir_wsd, 'WarsawData2')
        self.mask_dir_wsd = os.path.join(parent_dir_wsd, 'WarsawMasksCoarse')        
        
        with open(bins_path_wsd, 'rb') as wsdbinsFile:
            self.bins_wsd = pkl.load(wsdbinsFile)
        
        self.image_dir_bxgrid = os.path.join(parent_dir_bxgrid, 'iris')
        self.mask_dir_bxgrid = os.path.join(parent_dir_bxgrid, 'masks_coarse')       
        
        with open(bins_path_bxgrid, 'rb') as bxgridbinsFile:
            self.bins_bxgrid = pkl.load(bxgridbinsFile)
        '''    
        n_problems = 0
        for identifier in tqdm(self.bins_wsd.keys()):
             for bin_num in self.bins_wsd[identifier].keys():
                 for imagename in self.bins_wsd[identifier][bin_num]:
                     indiv_id = imagename.split('_')[0]
                     imagepath = os.path.join(os.path.join(self.image_dir_wsd, indiv_id), imagename)
                     maskpath = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id), imagename)
                     image = Image.open(imagepath).convert("L")
                     if np.mean(np.array(image)) == 0 or np.std(np.array(image)) <= 0.01:
                         self.bins_wsd[identifier][bin_num].remove(imagename)
                         n_problems += 1
                         continue
                     mask = Image.open(maskpath).convert("L")
                     if np.mean(np.array(mask)) <= 10:
                         self.bins_wsd[identifier][bin_num].remove(imagename)
                         n_problems += 1
                         continue
                         
        for identifier in tqdm(self.bins_bxgrid.keys()):
             for bin_num in self.bins_bxgrid[identifier].keys():
                 for imagename in self.bins_bxgrid[identifier][bin_num]:
                     imagepath = os.path.join(self.image_dir_bxgrid, imagename)
                     maskpath = os.path.join(self.mask_dir_bxgrid, imagename)
                     image = Image.open(imagepath).convert("L")
                     if np.mean(np.array(image)) == 0 or np.std(np.array(image)) <= 0.01:
                         self.bins_bxgrid[identifier][bin_num].remove(imagename)
                         n_problems += 1
                         continue
                     mask = Image.open(maskpath).convert("L")
                     if np.mean(np.array(mask)) <= 10:
                         self.bins_bxgrid[identifier][bin_num].remove(imagename)
                         n_problems += 1
                         continue
        
        print("Problems found: ", n_problems)          
        '''                
        
        self.all_ids = list(self.bins_wsd.keys()) + list(self.bins_bxgrid.keys())
        
        print('Initializing quadruplets for Warsaw Pupil Dynamics...')
        self.quadruplets = []
        self.genuine_ids = []
        self.imposter_ids = []
        wsd_identifiers = list(self.bins_wsd.keys())
        random.shuffle(wsd_identifiers)
        for identifier in tqdm(wsd_identifiers):
            wsd_identifiers_removed = wsd_identifiers.copy()
            wsd_identifiers_removed.remove(identifier)
            bin_numbers1 = list(self.bins_wsd[identifier].keys())
            bin_numbers2 = list(self.bins_wsd[identifier].keys())
            bin_numbers1.sort()
            bin_numbers2.sort()
            for bin_num_1 in bin_numbers1:
                for bin_num_2 in bin_numbers2:
                    if bin_num_1 != bin_num_2:
                        random.shuffle(self.bins_wsd[identifier][bin_num_1])
                        random.shuffle(self.bins_wsd[identifier][bin_num_2])
                        max_pairs = min(len(self.bins_wsd[identifier][bin_num_1]), len(self.bins_wsd[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            imposter_ind_max = 0
                            while imposter_ind_max == 0:
                                imposter_id = random.choice(wsd_identifiers_removed)
                                imposter_ind_max = len(self.bins_wsd[imposter_id][bin_num_2])
                            imagename1 = self.bins_wsd[identifier][bin_num_1][img_ind]
                            imagename2 = self.bins_wsd[identifier][bin_num_2][img_ind]
                            imagename3 = self.bins_wsd[imposter_id][bin_num_2][random.randrange(0, imposter_ind_max)]
                            imagename4 = self.bins_wsd[identifier][bin_num_2][random.randrange(0, len(self.bins_wsd[identifier][bin_num_2]))]
                            indiv_id1 = imagename1.split('_')[0]
                            indiv_id2 = imagename2.split('_')[0]
                            indiv_id3 = imagename3.split('_')[0]
                            indiv_id4 = imagename4.split('_')[0]
                            imagepath1 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id1), imagename1)
                            maskpath1 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id1), imagename1)
                            imagepath2 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id2), imagename2)
                            maskpath2 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id2), imagename2)
                            imagepath3 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id3), imagename3)
                            maskpath3 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id3), imagename3)
                            imagepath4 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id4), imagename4)
                            maskpath4 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id4), imagename4)
                            self.quadruplets.append([imagepath1, maskpath1, imagepath2, maskpath2, imagepath3, maskpath3, imagepath4, maskpath4, identifier, imposter_id])             
        
        print('Initializing quadruplets for BXGRID...')
        bxgrid_identifiers = list(self.bins_bxgrid.keys())
        random.shuffle(bxgrid_identifiers)
        for identifier in tqdm(bxgrid_identifiers):
            bxgrid_identifiers_removed = bxgrid_identifiers.copy()
            bxgrid_identifiers_removed.remove(identifier)
            bin_numbers1 = list(self.bins_bxgrid[identifier].keys())
            bin_numbers2 = list(self.bins_bxgrid[identifier].keys())
            bin_numbers1.sort()
            bin_numbers2.sort()
            for bin_num_1 in bin_numbers1:
                for bin_num_2 in bin_numbers2:
                    if bin_num_1 != bin_num_2:
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_1])
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_2])
                        max_pairs = min(len(self.bins_bxgrid[identifier][bin_num_1]), len(self.bins_bxgrid[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            imposter_ind_max = 0
                            while imposter_ind_max == 0:
                                imposter_id = random.choice(bxgrid_identifiers_removed)
                                imposter_ind_max = len(self.bins_bxgrid[imposter_id][bin_num_2])
                            imagename1 = self.bins_bxgrid[identifier][bin_num_1][img_ind]
                            imagename2 = self.bins_bxgrid[identifier][bin_num_2][img_ind]
                            imagename3 = self.bins_bxgrid[imposter_id][bin_num_2][random.randrange(0, imposter_ind_max)]
                            imagename4 = self.bins_bxgrid[identifier][bin_num_2][random.randrange(0, len(self.bins_bxgrid[identifier][bin_num_2]))]
                            imagepath1 = os.path.join(self.image_dir_bxgrid, imagename1)
                            maskpath1 = os.path.join(self.mask_dir_bxgrid, imagename1)
                            imagepath2 = os.path.join(self.image_dir_bxgrid, imagename2)
                            maskpath2 = os.path.join(self.mask_dir_bxgrid, imagename2)
                            imagepath3 = os.path.join(self.image_dir_bxgrid, imagename3)
                            maskpath3 = os.path.join(self.mask_dir_bxgrid, imagename3)
                            imagepath4 = os.path.join(self.image_dir_bxgrid, imagename4)
                            maskpath4 = os.path.join(self.mask_dir_bxgrid, imagename4)
                            self.quadruplets.append([imagepath1, maskpath1, imagepath2, maskpath2, imagepath3, maskpath3, imagepath4, maskpath4, identifier, imposter_id])
                            
        print('Total quadruplets: ', len(self.quadruplets))
        

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5, ))
        ])
        
        self.pupil_iris_xyrs_path_wsd = os.path.join(parent_dir_wsd, 'pupil_iris_xyrs_new.pkl')
        with open(self.pupil_iris_xyrs_path_wsd, 'rb') as pupilirisxyrsFile:
            self.pupil_iris_xyrs_wsd = pkl.load(pupilirisxyrsFile)
        
        self.pupil_iris_xyrs_path_bxgrid = os.path.join(parent_dir_bxgrid, 'pupil_iris_xyrs_new.pkl')
        with open(self.pupil_iris_xyrs_path_bxgrid, 'rb') as pupilirisxyrsFile:
            self.pupil_iris_xyrs_bxgrid = pkl.load(pupilirisxyrsFile)
        
        self.pupil_xyrs = {}
        self.iris_xyrs = {}
        
        for imagename in self.pupil_iris_xyrs_wsd.keys():
            imagepath = os.path.join(self.image_dir_wsd, imagename + '.bmp')
            self.pupil_xyrs[imagepath] = self.pupil_iris_xyrs_wsd[imagename]['pxyr']
            self.iris_xyrs[imagepath] = self.pupil_iris_xyrs_wsd[imagename]['ixyr']
        
        for imagename in self.pupil_iris_xyrs_bxgrid.keys():
            imagepath = os.path.join(self.image_dir_bxgrid, imagename + '.png')
            self.pupil_xyrs[imagepath] = self.pupil_iris_xyrs_bxgrid[imagename]['pxyr']
            self.iris_xyrs[imagepath] = self.pupil_iris_xyrs_bxgrid[imagename]['ixyr']
        
        self.heavy_augment_prob = 0.3
    
    def load_image(self, file):
        return Image.open(file).convert('L')
        
    def __getitem__(self, index):
    
        quadruplet = self.quadruplets[index]
        
        inp_img_path = quadruplet[0].strip()
        inp_mask_path = quadruplet[1].strip()
        tar_img_path = quadruplet[2].strip()
        tar_mask_path = quadruplet[3].strip()
        imp_img_path = quadruplet[4].strip()
        imp_mask_path = quadruplet[5].strip()
        tar2_img_path = quadruplet[6].strip()
        tar2_mask_path = quadruplet[7].strip()
        
        inp_img = self.load_image(inp_img_path)
        inp_mask = self.load_image(inp_mask_path)
        
        tar_img = self.load_image(tar_img_path)
        tar_mask = self.load_image(tar_mask_path)
        
        imp_img = self.load_image(imp_img_path)
        imp_mask = self.load_image(imp_mask_path)
        
        tar2_img = self.load_image(tar2_img_path)
        tar2_mask = self.load_image(tar2_mask_path)
        
        inp_img_pupil_xyr = torch.FloatTensor(self.pupil_xyrs[inp_img_path])
        inp_img_iris_xyr = torch.FloatTensor(self.iris_xyrs[inp_img_path])
        
        tar_img_pupil_xyr = torch.FloatTensor(self.pupil_xyrs[tar_img_path])
        tar_img_iris_xyr = torch.FloatTensor(self.iris_xyrs[tar_img_path])
        
        imp_img_pupil_xyr = torch.FloatTensor(self.pupil_xyrs[imp_img_path])
        imp_img_iris_xyr = torch.FloatTensor(self.iris_xyrs[imp_img_path])
        
        tar2_img_pupil_xyr = torch.FloatTensor(self.pupil_xyrs[tar2_img_path])
        tar2_img_iris_xyr = torch.FloatTensor(self.iris_xyrs[tar2_img_path])
        
        '''
        sw, sh = inp_img.size
        sw_mult = (sw/320)
        sh_mult = (self.input_size[0]/sh)
        
        bw, bh = tar_img.size
        bw_mult = (self.input_size[1]/bw)
        bh_mult = (self.input_size[0]/bh)
        
        biw, bih = imp_img.size
        biw_mult = (self.input_size[1]/biw)
        bih_mult = (self.input_size[0]/bih)
        
        inp_img_pupil_xyr[0] = (inp_img_pupil_xyr[0] * sw_mult)
        inp_img_pupil_xyr[1] = (inp_img_pupil_xyr[1] * sh_mult)
        inp_img_pupil_xyr[2] = (inp_img_pupil_xyr[2] * max(sw_mult, sh_mult))
        
        tar_img_pupil_xyr[0] = (tar_img_pupil_xyr[0] * bw_mult)
        tar_img_pupil_xyr[1] = (tar_img_pupil_xyr[1] * bh_mult)
        tar_img_pupil_xyr[2] = (tar_img_pupil_xyr[2] * max(bw_mult, bh_mult))
        
        imp_img_pupil_xyr[0] = (imp_img_pupil_xyr[0] * biw_mult)
        imp_img_pupil_xyr[1] = (imp_img_pupil_xyr[1] * bih_mult)
        imp_img_pupil_xyr[2] = (imp_img_pupil_xyr[2] * max(biw_mult, bih_mult))
    
        inp_img_iris_xyr[0] = (inp_img_iris_xyr[0] * sw_mult)
        inp_img_iris_xyr[1] = (inp_img_iris_xyr[1] * sh_mult)
        inp_img_iris_xyr[2] = (inp_img_iris_xyr[2] * max(sw_mult, sh_mult))
        
        tar_img_iris_xyr[0] = (tar_img_iris_xyr[0] * bw_mult)
        tar_img_iris_xyr[1] = (tar_img_iris_xyr[1] * bh_mult)
        tar_img_iris_xyr[2] = (tar_img_iris_xyr[2] * max(bw_mult, bh_mult))
        
        imp_img_iris_xyr[0] = (imp_img_iris_xyr[0] * biw_mult)
        imp_img_iris_xyr[1] = (imp_img_iris_xyr[1] * bih_mult)
        imp_img_iris_xyr[2] = (imp_img_iris_xyr[2] * max(biw_mult, bih_mult))
        '''
        sw, sh = inp_img.size
        bw, bh = tar_img.size
        bw2, bh2 = tar2_img.size
        iw, ih = imp_img.size
        
        if self.flip_data and random.random() < 0.5:
            inp_img = inp_img.transpose(Image.FLIP_LEFT_RIGHT) 
            inp_mask = inp_mask.transpose(Image.FLIP_LEFT_RIGHT)
            tar_img = tar_img.transpose(Image.FLIP_LEFT_RIGHT) 
            tar_mask = tar_mask.transpose(Image.FLIP_LEFT_RIGHT)
            tar2_img = tar2_img.transpose(Image.FLIP_LEFT_RIGHT) 
            tar2_mask = tar2_mask.transpose(Image.FLIP_LEFT_RIGHT)  
            inp_img_pupil_xyr[0] = sw - inp_img_pupil_xyr[0]
            tar_img_pupil_xyr[0] = bw - tar_img_pupil_xyr[0]
            tar2_img_pupil_xyr[0] = bw2 - tar2_img_pupil_xyr[0]
            inp_img_iris_xyr[0] = sw - inp_img_iris_xyr[0]
            tar_img_iris_xyr[0] = bw - tar_img_iris_xyr[0]
            tar2_img_iris_xyr[0] = bw2 - tar2_img_iris_xyr[0]
        
        if self.flip_data and random.random() < 0.5:
            imp_img = imp_img.transpose(Image.FLIP_LEFT_RIGHT)
            imp_mask = imp_mask.transpose(Image.FLIP_LEFT_RIGHT)
            imp_img_pupil_xyr[0] = iw - imp_img_pupil_xyr[0]
            imp_img_iris_xyr[0] = iw - imp_img_iris_xyr[0]
        
        if self.iris_crop:
            #inp_img.save('./check1.png')
            #print((inp_img_iris_xyr[0].item() - inp_img_iris_xyr[2].item(), inp_img_iris_xyr[1].item() - inp_img_iris_xyr[2].item(), inp_img_iris_xyr[0].item() + inp_img_iris_xyr[2].item(), inp_img_iris_xyr[1].item() + inp_img_iris_xyr[2].item()))
            inp_img = inp_img.crop((inp_img_iris_xyr[0].item() - inp_img_iris_xyr[2].item(), inp_img_iris_xyr[1].item() - inp_img_iris_xyr[2].item(), inp_img_iris_xyr[0].item() + inp_img_iris_xyr[2].item(), inp_img_iris_xyr[1].item() + inp_img_iris_xyr[2].item()))
            inp_mask = inp_mask.crop((inp_img_iris_xyr[0].item() - inp_img_iris_xyr[2].item(), inp_img_iris_xyr[1].item() - inp_img_iris_xyr[2].item(), inp_img_iris_xyr[0].item() + inp_img_iris_xyr[2].item(), inp_img_iris_xyr[1].item() + inp_img_iris_xyr[2].item()))
            sw_crop, sh_crop = inp_img.size
            inp_img_pupil_xyr[0] = inp_img_pupil_xyr[0] - inp_img_iris_xyr[0] + sw_crop/2
            inp_img_pupil_xyr[1] = inp_img_pupil_xyr[1] - inp_img_iris_xyr[1] + sh_crop/2
            inp_img_iris_xyr[0] = sw_crop/2
            inp_img_iris_xyr[1] = sh_crop/2            
            
            tar_img = tar_img.crop((tar_img_iris_xyr[0].item() - tar_img_iris_xyr[2].item(), tar_img_iris_xyr[1].item() - tar_img_iris_xyr[2].item(), tar_img_iris_xyr[0].item() + tar_img_iris_xyr[2].item(), tar_img_iris_xyr[1].item() + tar_img_iris_xyr[2].item()))
            tar_mask = tar_mask.crop((tar_img_iris_xyr[0].item() - tar_img_iris_xyr[2].item(), tar_img_iris_xyr[1].item() - tar_img_iris_xyr[2].item(), tar_img_iris_xyr[0].item() + tar_img_iris_xyr[2].item(), tar_img_iris_xyr[1].item() + tar_img_iris_xyr[2].item()))
            bw_crop, bh_crop = tar_img.size
            tar_img_pupil_xyr[0] = tar_img_pupil_xyr[0] - tar_img_iris_xyr[0] + bw_crop/2
            tar_img_pupil_xyr[1] = tar_img_pupil_xyr[1] - tar_img_iris_xyr[1] + bh_crop/2
            tar_img_iris_xyr[0] = bw_crop/2
            tar_img_iris_xyr[1] = bh_crop/2
            
            tar2_img = tar2_img.crop((tar2_img_iris_xyr[0].item() - tar2_img_iris_xyr[2].item(), tar2_img_iris_xyr[1].item() - tar2_img_iris_xyr[2].item(), tar2_img_iris_xyr[0].item() + tar2_img_iris_xyr[2].item(), tar2_img_iris_xyr[1].item() + tar2_img_iris_xyr[2].item()))
            tar2_mask = tar2_mask.crop((tar2_img_iris_xyr[0].item() - tar2_img_iris_xyr[2].item(), tar2_img_iris_xyr[1].item() - tar2_img_iris_xyr[2].item(), tar2_img_iris_xyr[0].item() + tar2_img_iris_xyr[2].item(), tar2_img_iris_xyr[1].item() + tar2_img_iris_xyr[2].item()))
            bw2_crop, bh2_crop = tar2_img.size
            tar2_img_pupil_xyr[0] = tar2_img_pupil_xyr[0] - tar2_img_iris_xyr[0] + bw2_crop/2
            tar2_img_pupil_xyr[1] = tar2_img_pupil_xyr[1] - tar2_img_iris_xyr[1] + bh2_crop/2
            tar2_img_iris_xyr[0] = bw2_crop/2
            tar2_img_iris_xyr[1] = bh2_crop/2
            
            imp_img = imp_img.crop((imp_img_iris_xyr[0].item() - imp_img_iris_xyr[2].item(), imp_img_iris_xyr[1].item() - imp_img_iris_xyr[2].item(), imp_img_iris_xyr[0].item() + imp_img_iris_xyr[2].item(), imp_img_iris_xyr[1].item() + imp_img_iris_xyr[2].item()))
            imp_mask = imp_mask.crop((imp_img_iris_xyr[0].item() - imp_img_iris_xyr[2].item(), imp_img_iris_xyr[1].item() - imp_img_iris_xyr[2].item(), imp_img_iris_xyr[0].item() + imp_img_iris_xyr[2].item(), imp_img_iris_xyr[1].item() + imp_img_iris_xyr[2].item()))
            iw_crop, ih_crop = imp_img.size
            imp_img_pupil_xyr[0] = imp_img_pupil_xyr[0] - imp_img_iris_xyr[0] + iw_crop/2
            imp_img_pupil_xyr[1] = imp_img_pupil_xyr[1] - imp_img_iris_xyr[1] + ih_crop/2
            imp_img_iris_xyr[0] = iw_crop/2
            imp_img_iris_xyr[1] = ih_crop/2
            
            '''
            inp_img_vis = np.uint8(inp_img.convert('RGB'))
            try:
                inp_img_vis = cv2.circle(inp_img_vis, (int(inp_img_pupil_xyr[0].item()),int(inp_img_pupil_xyr[1].item())), int(inp_img_pupil_xyr[2].item()), (0, 0, 255), 2)
            except:
                print("Pupil circle could not be visualized, values are: ", (int(inp_img_pupil_xyr[0].item()),int(inp_img_pupil_xyr[1].item()), int(inp_img_pupil_xyr[2].item())), ' ', inp_img_vis.shape)
                pass
            try:
                inp_img_vis = cv2.circle(inp_img_vis, (int(inp_img_iris_xyr[0].item()),int(inp_img_iris_xyr[1].item())), int(inp_img_iris_xyr[2].item()), (255, 0, 0), 2)
            except:
                print("Iris circle could not be visualized, values are: ", (int(inp_img_iris_xyr[0].item()),int(inp_img_iris_xyr[1].item()), int(inp_img_iris_xyr[2].item())), ' ', inp_img_vis.shape)
            
            Image.fromarray(inp_img_vis).save('./check2.png')
            '''
            
            sh_mult = self.input_size[0]/sh_crop
            sw_mult = self.input_size[1]/sw_crop
            inp_img = inp_img.resize((self.input_size[1], self.input_size[0]))
            inp_mask = inp_mask.resize((self.input_size[1], self.input_size[0]))
            inp_img_pupil_xyr[0] = self.input_size[1]/2 - sw_mult * (sw_crop/2 - inp_img_pupil_xyr[0])
            inp_img_pupil_xyr[1] = self.input_size[0]/2 - sh_mult * (sh_crop/2 - inp_img_pupil_xyr[1])
            inp_img_pupil_xyr[2] = max(sw_mult, sh_mult) * inp_img_pupil_xyr[2]
            inp_img_iris_xyr[0] = self.input_size[1]/2
            inp_img_iris_xyr[1] = self.input_size[0]/2
            inp_img_iris_xyr[2] = max(sw_mult, sh_mult) * inp_img_iris_xyr[2]
            
            bh_mult = self.input_size[0]/bh_crop
            bw_mult = self.input_size[1]/bw_crop
            tar_img = tar_img.resize((self.input_size[1], self.input_size[0]))
            tar_mask = tar_mask.resize((self.input_size[1], self.input_size[0]))
            tar_img_pupil_xyr[0] = self.input_size[1]/2 - bw_mult * (bw_crop/2 - tar_img_pupil_xyr[0])
            tar_img_pupil_xyr[1] = self.input_size[0]/2 - bh_mult * (bh_crop/2 - tar_img_pupil_xyr[1])
            tar_img_pupil_xyr[2] = max(bw_mult, bh_mult) * tar_img_pupil_xyr[2]
            tar_img_iris_xyr[0] = self.input_size[1]/2
            tar_img_iris_xyr[1] = self.input_size[0]/2
            tar_img_iris_xyr[2] = max(bw_mult, bh_mult) * tar_img_iris_xyr[2]
            
            bh2_mult = self.input_size[0]/bh2_crop
            bw2_mult = self.input_size[1]/bw2_crop
            tar2_img = tar2_img.resize((self.input_size[1], self.input_size[0]))
            tar2_mask = tar2_mask.resize((self.input_size[1], self.input_size[0]))
            tar2_img_pupil_xyr[0] = self.input_size[1]/2 - bw2_mult * (bw2_crop/2 - tar2_img_pupil_xyr[0])
            tar2_img_pupil_xyr[1] = self.input_size[0]/2 - bh2_mult * (bh2_crop/2 - tar2_img_pupil_xyr[1])
            tar2_img_pupil_xyr[2] = max(bw2_mult, bh2_mult) * tar2_img_pupil_xyr[2]
            tar2_img_iris_xyr[0] = self.input_size[1]/2
            tar2_img_iris_xyr[1] = self.input_size[0]/2
            tar2_img_iris_xyr[2] = max(bw2_mult, bh2_mult) * tar2_img_iris_xyr[2]
            
            ih_mult = self.input_size[0]/ih_crop
            iw_mult = self.input_size[1]/iw_crop
            imp_img = imp_img.resize((self.input_size[0], self.input_size[1]))
            imp_mask = imp_mask.resize((self.input_size[0], self.input_size[1]))
            imp_img_pupil_xyr[0] = self.input_size[1]/2 - iw_mult * (iw_crop/2 - imp_img_pupil_xyr[0])
            imp_img_pupil_xyr[1] = self.input_size[0]/2 - ih_mult * (ih_crop/2 - imp_img_pupil_xyr[1])
            imp_img_pupil_xyr[2] = max(iw_mult, ih_mult) * imp_img_pupil_xyr[2]
            imp_img_iris_xyr[0] = self.input_size[1]/2
            imp_img_iris_xyr[1] = self.input_size[0]/2
            imp_img_iris_xyr[2] = max(iw_mult, ih_mult) * imp_img_iris_xyr[2]
            
            #print('Saving ..')
            '''
            inp_img_vis = np.uint8(inp_img.convert('RGB'))
            try:
                inp_img_vis = cv2.circle(inp_img_vis, (int(inp_img_pupil_xyr[0].item()),int(inp_img_pupil_xyr[1].item())), int(inp_img_pupil_xyr[2].item()), (0, 0, 255), 2)
            except:
                print("Pupil circle could not be visualized, values are: ", (int(inp_img_pupil_xyr[0].item()),int(inp_img_pupil_xyr[1].item()), int(inp_img_pupil_xyr[2].item())), ' ', inp_img_vis.shape)
                pass
            try:
                inp_img_vis = cv2.circle(inp_img_vis, (int(inp_img_iris_xyr[0].item()),int(inp_img_iris_xyr[1].item())), int(inp_img_iris_xyr[2].item()), (255, 0, 0), 2)
            except:
                print("Iris circle could not be visualized, values are: ", (int(inp_img_iris_xyr[0].item()),int(inp_img_iris_xyr[1].item()), int(inp_img_iris_xyr[2].item())), ' ', inp_img_vis.shape)
            
            Image.fromarray(inp_img_vis).save('./check3.png')
            '''
        else:
            sw, sh = inp_img.size
            sh_mult = self.input_size_nocrop[0]/sh
            sw_mult = self.input_size_nocrop[1]/sw
            
            inp_img = inp_img.resize((self.input_size_nocrop[1], self.input_size_nocrop[0]))
            inp_mask = inp_mask.resize((self.input_size_nocrop[1], self.input_size_nocrop[0]))
            inp_img_pupil_xyr[0] = self.input_size_nocrop[1]/2 - sw_mult * (sw/2 - inp_img_pupil_xyr[0])
            inp_img_pupil_xyr[1] = self.input_size_nocrop[0]/2 - sh_mult * (sh/2 - inp_img_pupil_xyr[1])
            inp_img_pupil_xyr[2] = max(sw_mult, sh_mult) * inp_img_pupil_xyr[2]
            inp_img_iris_xyr[0] = self.input_size_nocrop[1]/2 - sw_mult * (sw/2 - inp_img_iris_xyr[0])
            inp_img_iris_xyr[1] = self.input_size_nocrop[0]/2 - sh_mult * (sh/2 - inp_img_iris_xyr[1])
            inp_img_iris_xyr[2] = max(sw_mult, sh_mult) * inp_img_iris_xyr[2]
            
            bw, bh = tar_img.size
            bh_mult = self.input_size_nocrop[0]/bh
            bw_mult = self.input_size_nocrop[1]/bw
            
            tar_img = tar_img.resize((self.input_size_nocrop[1], self.input_size_nocrop[0]))
            tar_mask = tar_mask.resize((self.input_size_nocrop[1], self.input_size_nocrop[0]))
            tar_img_pupil_xyr[0] = self.input_size_nocrop[1]/2 - bw_mult * (bw/2 - tar_img_pupil_xyr[0])
            tar_img_pupil_xyr[1] = self.input_size_nocrop[0]/2 - bh_mult * (bh/2 - tar_img_pupil_xyr[1])
            tar_img_pupil_xyr[2] = max(bw_mult, bh_mult) * tar_img_pupil_xyr[2]
            tar_img_iris_xyr[0] = self.input_size_nocrop[1]/2 - bw_mult * (bw/2 - tar_img_iris_xyr[0])
            tar_img_iris_xyr[1] = self.input_size_nocrop[0]/2 - bh_mult * (bh/2 - tar_img_iris_xyr[1])
            tar_img_iris_xyr[2] = max(bw_mult, bh_mult) * tar_img_iris_xyr[2]
            
            b2w, b2h = tar2_img.size
            b2h_mult = self.input_size_nocrop[0]/b2h
            b2w_mult = self.input_size_nocrop[1]/b2w
            
            tar2_img = tar2_img.resize((self.input_size_nocrop[1], self.input_size_nocrop[0]))
            tar2_mask = tar2_mask.resize((self.input_size_nocrop[1], self.input_size_nocrop[0]))
            tar2_img_pupil_xyr[0] = self.input_size_nocrop[1]/2 - b2w_mult * (b2w/2 - tar2_img_pupil_xyr[0])
            tar2_img_pupil_xyr[1] = self.input_size_nocrop[0]/2 - b2h_mult * (b2h/2 - tar2_img_pupil_xyr[1])
            tar2_img_pupil_xyr[2] = max(b2w_mult, b2h_mult) * tar2_img_pupil_xyr[2]
            tar2_img_iris_xyr[0] = self.input_size_nocrop[1]/2 - b2w_mult * (b2w/2 - tar2_img_iris_xyr[0])
            tar2_img_iris_xyr[1] = self.input_size_nocrop[0]/2 - b2h_mult * (b2h/2 - tar2_img_iris_xyr[1])
            tar2_img_iris_xyr[2] = max(b2w_mult, b2h_mult) * tar2_img_iris_xyr[2]
            
            iw, ih = imp_img.size
            ih_mult = self.input_size_nocrop[0]/ih
            iw_mult = self.input_size_nocrop[1]/iw
            
            imp_img = imp_img.resize((self.input_size_nocrop[1], self.input_size_nocrop[0]))
            imp_mask = imp_mask.resize((self.input_size_nocrop[1], self.input_size_nocrop[0]))
            imp_img_pupil_xyr[0] = self.input_size_nocrop[1]/2 - iw_mult * (iw/2 - imp_img_pupil_xyr[0])
            imp_img_pupil_xyr[1] = self.input_size_nocrop[0]/2 - ih_mult * (ih/2 - imp_img_pupil_xyr[1])
            imp_img_pupil_xyr[2] = max(iw_mult, ih_mult) * imp_img_pupil_xyr[2]
            imp_img_iris_xyr[0] = self.input_size_nocrop[1]/2 - iw_mult * (iw/2 - imp_img_iris_xyr[0])
            imp_img_iris_xyr[1] = self.input_size_nocrop[0]/2 - ih_mult * (ih/2 - imp_img_iris_xyr[1])
            imp_img_iris_xyr[2] = max(iw_mult, ih_mult) * imp_img_iris_xyr[2]
            
        if random.random() < self.heavy_augment_prob:
            if random.random() < 0.5:
                random_choice = np.random.choice([1,2,3,4,5])
                if random_choice == 1:   
                    # sharpening
                    random_degree = np.random.choice([1,2,3,4,5])
                    if random_degree == 1:
                        inp_img = inp_img.filter(ImageFilter.DETAIL)
                        tar_img = tar_img.filter(ImageFilter.DETAIL)
                    elif random_degree == 2:
                        inp_img = inp_img.filter(ImageFilter.SHARPEN)
                        tar_img = tar_img.filter(ImageFilter.SHARPEN)
                    elif random_degree == 3:
                        inp_img = inp_img.filter(ImageFilter.EDGE_ENHANCE)
                        tar_img = tar_img.filter(ImageFilter.EDGE_ENHANCE)
                    elif random_degree == 4:
                        inp_img = inp_img.filter(ImageFilter.EDGE_ENHANCE_MORE)
                        tar_img = tar_img.filter(ImageFilter.EDGE_ENHANCE_MORE)                          
                    else:
                        random_alpha = random.uniform(0.0, 0.3)
                        random_lightness = random.uniform(0.6, 1.0)
                        aug = iaa.Sharpen(alpha=random_alpha, lightness=random_lightness)
                        inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                        inp_img = Image.fromarray(inp_img_np[0])
                        tar_img_np = aug(images = np.expand_dims(np.array(tar_img), axis=0))
                        tar_img = Image.fromarray(tar_img_np[0])
                elif random_choice == 2:
                    #blurring
                    random_degree = np.random.choice([1,2,3,4,5])
                    if random_degree == 1:
                        inp_img = inp_img.filter(ImageFilter.GaussianBlur())
                        tar_img = tar_img.filter(ImageFilter.GaussianBlur())
                    elif random_degree == 2:
                        inp_img = inp_img.filter(ImageFilter.BLUR)
                        tar_img = tar_img.filter(ImageFilter.BLUR)
                    elif random_degree == 3:
                        inp_img = inp_img.filter(ImageFilter.SMOOTH)
                        tar_img = tar_img.filter(ImageFilter.SMOOTH)
                    elif random_degree == 4:
                        inp_img = inp_img.filter(ImageFilter.SMOOTH_MORE)
                        tar_img = tar_img.filter(ImageFilter.SMOOTH_MORE)
                    else:
                        random_degree = random.randint(-179, 180)
                        aug = iaa.MotionBlur(k=3, angle=random_degree)
                        inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                        inp_img = Image.fromarray(inp_img_np[0])
                        tar_img_np = aug(images = np.expand_dims(np.array(tar_img), axis=0))
                        tar_img = Image.fromarray(tar_img_np[0])
                elif random_choice == 3:
                    # Basic compression and expansion
                    divider = random.uniform(1.0, 1.5)
                    cw, ch = inp_img.size
                    new_cw = int(cw/divider)
                    new_ch = int(ch/divider)
                    
                    first_choice = np.random.choice([1,2,3,4,5,6])
                    if first_choice == 1:
                        inp_img = inp_img.resize((new_cw, new_ch), Image.NEAREST)
                        tar_img = tar_img.resize((new_cw, new_ch), Image.NEAREST)
                    elif first_choice == 2:
                        inp_img = inp_img.resize((new_cw, new_ch), Image.BILINEAR)
                        tar_img = tar_img.resize((new_cw, new_ch), Image.BILINEAR)
                    elif first_choice == 3:
                        inp_img = inp_img.resize((new_cw, new_ch), Image.BICUBIC)
                        tar_img = tar_img.resize((new_cw, new_ch), Image.BICUBIC)
                    elif first_choice == 4:
                        inp_img = inp_img.resize((new_cw, new_ch), Image.LANCZOS)
                        tar_img = tar_img.resize((new_cw, new_ch), Image.LANCZOS)
                    elif first_choice == 5:
                        inp_img = inp_img.resize((new_cw, new_ch), Image.HAMMING)
                        tar_img = tar_img.resize((new_cw, new_ch), Image.HAMMING)
                    else:
                        inp_img = inp_img.resize((new_cw, new_ch), Image.BOX)
                        tar_img = tar_img.resize((new_cw, new_ch), Image.BOX)
                    
                    second_choice = np.random.choice([1,2,3,4,5,6])
                    if second_choice == 1:
                        inp_img = inp_img.resize((cw, ch), Image.NEAREST)
                        tar_img = tar_img.resize((cw, ch), Image.NEAREST)
                    elif second_choice == 2:
                        inp_img = inp_img.resize((cw, ch), Image.BILINEAR)
                        tar_img = tar_img.resize((cw, ch), Image.BILINEAR)
                    elif second_choice == 3:
                        inp_img = inp_img.resize((cw, ch), Image.BICUBIC)
                        tar_img = tar_img.resize((cw, ch), Image.BICUBIC)
                    elif second_choice == 4:
                        inp_img = inp_img.resize((cw, ch), Image.LANCZOS)
                        tar_img = tar_img.resize((cw, ch), Image.LANCZOS)
                    elif second_choice == 5:
                        inp_img = inp_img.resize((cw, ch), Image.HAMMING)
                        tar_img = tar_img.resize((cw, ch), Image.HAMMING)
                    else:
                        inp_img = inp_img.resize((cw, ch), Image.BOX)
                        tar_img = tar_img.resize((cw, ch), Image.BOX)
                elif random_choice == 4:
                    #JPEG compression
                    random_compression = random.randint(10, 50)
                    aug = iaa.JpegCompression(compression=random_compression)
                    inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                    inp_img = Image.fromarray(inp_img_np[0])
                    tar_img_np = aug(images = np.expand_dims(np.array(tar_img), axis=0))
                    tar_img = Image.fromarray(tar_img_np[0])
                else:
                    if random.random() < 0.5:
                        aug = iaa.AdditiveGaussianNoise(scale=3)
                        inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                        inp_img = Image.fromarray(inp_img_np[0])
                        tar_img_np = aug(images = np.expand_dims(np.array(tar_img), axis=0))
                        tar_img = Image.fromarray(tar_img_np[0])
                    else:
                        aug = iaa.AdditiveLaplaceNoise(scale=3)
                        inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                        inp_img = Image.fromarray(inp_img_np[0])
                        tar_img_np = aug(images = np.expand_dims(np.array(tar_img), axis=0))
                        tar_img = Image.fromarray(tar_img_np[0])
                       
            if random.random() < 0.5: # random contrast change
                random_degree = np.random.choice([1,2,3,4,5,6,7,8])                
                if random_degree == 1:
                    aug = iaa.GammaContrast((0.5, 2.0))
                elif random_degree == 2:
                    aug = iaa.LinearContrast((0.4, 1.6))
                elif random_degree == 3:
                    aug = iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6))
                elif random_degree == 4:
                    aug = iaa.LogContrast(gain=(0.6, 1.4)) 
                else:    
                    aug = iaa.pillike.EnhanceBrightness()
                inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                inp_img = Image.fromarray(inp_img_np[0])
                random_degree = np.random.choice([1,2,3,4,5,6,7,8])                
                if random_degree == 1:
                    aug = iaa.GammaContrast((0.5, 2.0))
                elif random_degree == 2:
                    aug = iaa.LinearContrast((0.4, 1.6))
                elif random_degree == 3:
                    aug = iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6))
                elif random_degree == 4:
                    aug = iaa.LogContrast(gain=(0.6, 1.4)) 
                else:    
                    aug = iaa.pillike.EnhanceBrightness()
                tar_img_np = aug(images = np.expand_dims(np.array(tar_img), axis=0))
                tar_img = Image.fromarray(tar_img_np[0])
        
        if random.random() < self.heavy_augment_prob:
            if random.random() < 0.5:
                random_choice = np.random.choice([1,2,3,4,5])
                if random_choice == 1:   
                    # sharpening
                    random_degree = np.random.choice([1,2,3,4,5])
                    if random_degree == 1:
                        imp_img = imp_img.filter(ImageFilter.DETAIL)
                    elif random_degree == 2:
                        imp_img = imp_img.filter(ImageFilter.SHARPEN)
                    elif random_degree == 3:
                        imp_img = imp_img.filter(ImageFilter.EDGE_ENHANCE)
                    elif random_degree == 4:
                        imp_img = imp_img.filter(ImageFilter.EDGE_ENHANCE_MORE)                            
                    else:
                        aug = iaa.Sharpen(alpha=(0.0, 0.3), lightness=(0.6, 1.0))
                        imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                        imp_img = Image.fromarray(imp_img_np[0])
                elif random_choice == 2:
                    #blurring
                    random_degree = np.random.choice([1,2,3,4,5])
                    if random_degree == 1:
                        imp_img = imp_img.filter(ImageFilter.GaussianBlur())
                    elif random_degree == 2:
                        imp_img = imp_img.filter(ImageFilter.BLUR)
                    elif random_degree == 3:
                        imp_img = imp_img.filter(ImageFilter.SMOOTH)
                    elif random_degree == 4:
                        imp_img = imp_img.filter(ImageFilter.SMOOTH_MORE)
                    else:
                        aug = iaa.MotionBlur(k=3)
                        imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                        imp_img = Image.fromarray(imp_img_np[0])
                elif random_choice == 3:
                    # Basic compression and expansion
                    divider = random.uniform(1.0, 1.5)
                    cw, ch = imp_img.size
                    new_cw = int(cw/divider)
                    new_ch = int(ch/divider)
                    
                    first_choice = np.random.choice([1,2,3,4,5,6])
                    if first_choice == 1:
                        imp_img = imp_img.resize((new_cw, new_ch), Image.NEAREST)
                    elif first_choice == 2:
                        imp_img = imp_img.resize((new_cw, new_ch), Image.BILINEAR)
                    elif first_choice == 3:
                        imp_img = imp_img.resize((new_cw, new_ch), Image.BICUBIC)
                    elif first_choice == 4:
                        imp_img = imp_img.resize((new_cw, new_ch), Image.LANCZOS)
                    elif first_choice == 5:
                        imp_img = imp_img.resize((new_cw, new_ch), Image.HAMMING)
                    else:
                        imp_img = imp_img.resize((new_cw, new_ch), Image.BOX)
                    
                    second_choice = np.random.choice([1,2,3,4,5,6])
                    if second_choice == 1:
                        imp_img = imp_img.resize((cw, ch), Image.NEAREST)
                    elif second_choice == 2:
                        imp_img = imp_img.resize((cw, ch), Image.BILINEAR)
                    elif second_choice == 3:
                        imp_img = imp_img.resize((cw, ch), Image.BICUBIC)
                    elif second_choice == 4:
                        imp_img = imp_img.resize((cw, ch), Image.LANCZOS)
                    elif second_choice == 5:
                        imp_img = imp_img.resize((cw, ch), Image.HAMMING)
                    else:
                        imp_img = imp_img.resize((cw, ch), Image.BOX)
                elif random_choice == 4:
                    #JPEG compression
                    aug = iaa.JpegCompression(compression=(10, 50))
                    imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                    imp_img = Image.fromarray(imp_img_np[0])
                else:
                    if random.random() < 0.5:
                        aug = iaa.AdditiveGaussianNoise(scale=3)
                        imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                        imp_img = Image.fromarray(imp_img_np[0])
                    else:
                        aug = iaa.AdditiveLaplaceNoise(scale=3)
                        imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                        imp_img = Image.fromarray(imp_img_np[0])
                
                         
            if random.random() < 0.5: # random contrast change
                random_degree = np.random.choice([1,2,3,4,5,6,7,8])                
                if random_degree == 1:
                    aug = iaa.GammaContrast((0.5, 2.0))
                    imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                    imp_img = Image.fromarray(imp_img_np[0])
                elif random_degree == 2:
                    aug = iaa.LinearContrast((0.4, 1.6))
                    imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                    imp_img = Image.fromarray(imp_img_np[0])
                elif random_degree == 3:
                    aug = iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6))
                    imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                    imp_img = Image.fromarray(imp_img_np[0])
                elif random_degree == 4:
                    aug = iaa.LogContrast(gain=(0.6, 1.4))
                    imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                    imp_img = Image.fromarray(imp_img_np[0])            
                else:    
                    aug = iaa.pillike.EnhanceBrightness()
                    imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                    imp_img = Image.fromarray(imp_img_np[0])
                    
        if self.normalize:
            inp_img_t = torch.from_numpy(np.float32(inp_img)).float()
            inp_img_mean = torch.mean(inp_img_t)
            inp_img_std = torch.std(inp_img_t)
            xform_inp_img = torch.clamp(torch.nan_to_num((inp_img_t - inp_img_mean)/inp_img_std, nan=4, posinf=4, neginf=-4), -4, 4)
            xform_inp_img = xform_inp_img.unsqueeze(0)
            
            xform_inp_mask = self.transform(inp_mask)
            xform_inp_mask[xform_inp_mask<0] = -1
            xform_inp_mask[xform_inp_mask>=0] = 1
            
            tar_img_t = torch.from_numpy(np.float32(tar_img)).float()
            tar_img_mean = torch.mean(tar_img_t)
            tar_img_std = torch.std(tar_img_t)
            xform_tar_img = torch.clamp(torch.nan_to_num((tar_img_t - tar_img_mean)/tar_img_std, nan=4, posinf=4, neginf=-4), -4, 4)
            xform_tar_img = xform_tar_img.unsqueeze(0)
            
            xform_tar_mask = self.transform(tar_mask)
            xform_tar_mask[xform_tar_mask<0] = -1.0
            xform_tar_mask[xform_tar_mask>=0] = 1.0
            
            tar2_img_t = torch.from_numpy(np.float32(tar2_img)).float()
            tar2_img_mean = torch.mean(tar2_img_t)
            tar2_img_std = torch.std(tar2_img_t)
            xform_tar2_img = torch.clamp(torch.nan_to_num((tar2_img_t - tar2_img_mean)/tar2_img_std, nan=4, posinf=4, neginf=-4), -4, 4)
            xform_tar2_img = xform_tar2_img.unsqueeze(0)
            
            xform_tar2_mask = self.transform(tar2_mask)
            xform_tar2_mask[xform_tar2_mask<0] = -1.0
            xform_tar2_mask[xform_tar2_mask>=0] = 1.0
            
            imp_img_t = torch.from_numpy(np.float32(imp_img)).float()
            imp_img_mean = torch.mean(imp_img_t)
            imp_img_std = torch.std(imp_img_t)
            xform_imp_img = torch.clamp(torch.nan_to_num((imp_img_t - imp_img_mean)/imp_img_std, nan=4, posinf=4, neginf=-4), -4, 4)
            xform_imp_img = xform_imp_img.unsqueeze(0)
            
            xform_imp_mask = self.transform(imp_mask)
            xform_imp_mask[xform_imp_mask<0] = -1.0
            xform_imp_mask[xform_imp_mask>=0] = 1.0
            return {'inp_img': xform_inp_img, 'inp_img_mean': inp_img_mean, 'inp_img_std':inp_img_std, 'tar_img' : xform_tar_img, 'tar_img_mean':tar_img_mean, 'tar_img_std':tar_img_std, 'tar2_img' : xform_tar2_img, 'tar2_img_mean':tar2_img_mean, 'tar2_img_std':tar2_img_std, 'imp_img' : xform_imp_img, 'imp_img_mean': imp_img_mean, 'imp_img_std':imp_img_std, 'inp_mask': xform_inp_mask, 'tar_mask' : xform_tar_mask, 'tar2_mask' : xform_tar2_mask, 'imp_mask' : xform_imp_mask, 'inp_img_pxyr' : inp_img_pupil_xyr, 'tar_img_pxyr' : tar_img_pupil_xyr, 'tar2_img_pxyr' : tar2_img_pupil_xyr, 'imp_img_pxyr' : imp_img_pupil_xyr, 'inp_img_ixyr': inp_img_iris_xyr, 'tar_img_ixyr' :  tar_img_iris_xyr, 'tar2_img_ixyr' :  tar2_img_iris_xyr, 'imp_img_ixyr' : imp_img_iris_xyr, 'identifier' : quadruplet[8], 'imposter_id' : quadruplet[9]}

        else:
            xform_inp_img = self.transform(inp_img)
            xform_inp_mask = self.transform(inp_mask)
            xform_inp_mask[xform_inp_mask<0] = -1
            xform_inp_mask[xform_inp_mask>=0] = 1
            
            xform_tar_img = self.transform(tar_img)
            xform_tar_mask = self.transform(tar_mask)
            xform_tar_mask[xform_tar_mask<0] = -1.0
            xform_tar_mask[xform_tar_mask>=0] = 1.0
            
            xform_tar2_img = self.transform(tar2_img)
            xform_tar2_mask = self.transform(tar2_mask)
            xform_tar2_mask[xform_tar2_mask<0] = -1.0
            xform_tar2_mask[xform_tar2_mask>=0] = 1.0
            
            xform_imp_img = self.transform(imp_img)
            xform_imp_mask = self.transform(imp_mask)
            xform_imp_mask[xform_imp_mask<0] = -1.0
            xform_imp_mask[xform_imp_mask>=0] = 1.0
            return {'inp_img': xform_inp_img, 'tar_img' : xform_tar_img, 'tar2_img' : xform_tar2_img, 'imp_img' : xform_imp_img, 'inp_mask': xform_inp_mask, 'tar_mask' : xform_tar_mask, 'tar2_mask' : xform_tar2_mask, 'imp_mask' : xform_imp_mask, 'inp_img_pxyr' : inp_img_pupil_xyr, 'tar_img_pxyr' : tar_img_pupil_xyr, 'tar2_img_pxyr' : tar2_img_pupil_xyr, 'imp_img_pxyr' : imp_img_pupil_xyr, 'inp_img_ixyr': inp_img_iris_xyr, 'tar_img_ixyr' :  tar_img_iris_xyr,  'tar2_img_ixyr' :  tar2_img_iris_xyr, 'imp_img_ixyr' : imp_img_iris_xyr, 'identifier' : quadruplet[8], 'imposter_id' : quadruplet[9]}
    
    def __len__(self):
        return len(self.quadruplets)
    
    def reset(self):
        print('Shuffling quadruplets for WSD...')
        self.quadruplets = []
        self.genuine_ids = []
        self.imposter_ids = []
        wsd_identifiers = list(self.bins_wsd.keys())
        random.shuffle(wsd_identifiers)
        for identifier in tqdm(wsd_identifiers):
            wsd_identifiers_removed = wsd_identifiers.copy()
            wsd_identifiers_removed.remove(identifier)
            bin_numbers1 = list(self.bins_wsd[identifier].keys())
            bin_numbers2 = list(self.bins_wsd[identifier].keys())
            bin_numbers1.sort()
            bin_numbers2.sort()
            for bin_num_1 in bin_numbers1:
                for bin_num_2 in bin_numbers2:
                    if bin_num_1 != bin_num_2:
                        random.shuffle(self.bins_wsd[identifier][bin_num_1])
                        random.shuffle(self.bins_wsd[identifier][bin_num_2])
                        max_pairs = min(len(self.bins_wsd[identifier][bin_num_1]), len(self.bins_wsd[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            imposter_ind_max = 0
                            while imposter_ind_max == 0:
                                imposter_id = random.choice(wsd_identifiers_removed)
                                imposter_ind_max = len(self.bins_wsd[imposter_id][bin_num_2])
                            imagename1 = self.bins_wsd[identifier][bin_num_1][img_ind]
                            imagename2 = self.bins_wsd[identifier][bin_num_2][img_ind]
                            imagename3 = self.bins_wsd[imposter_id][bin_num_2][random.randrange(0, imposter_ind_max)]
                            imagename4 = self.bins_wsd[identifier][bin_num_2][random.randrange(0, len(self.bins_wsd[identifier][bin_num_2]))]
                            indiv_id1 = imagename1.split('_')[0]
                            indiv_id2 = imagename2.split('_')[0]
                            indiv_id3 = imagename3.split('_')[0]
                            indiv_id4 = imagename4.split('_')[0]
                            imagepath1 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id1), imagename1)
                            maskpath1 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id1), imagename1)
                            imagepath2 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id2), imagename2)
                            maskpath2 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id2), imagename2)
                            imagepath3 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id3), imagename3)
                            maskpath3 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id3), imagename3)
                            imagepath4 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id4), imagename4)
                            maskpath4 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id4), imagename4)
                            self.quadruplets.append([imagepath1, maskpath1, imagepath2, maskpath2, imagepath3, maskpath3, imagepath4, maskpath4, identifier, imposter_id])             
        
        print('Shuffling quadruplets for BXGRID...')
        bxgrid_identifiers = list(self.bins_bxgrid.keys())
        random.shuffle(bxgrid_identifiers)
        for identifier in tqdm(bxgrid_identifiers):
            bxgrid_identifiers_removed = bxgrid_identifiers.copy()
            bxgrid_identifiers_removed.remove(identifier)
            bin_numbers1 = list(self.bins_bxgrid[identifier].keys())
            bin_numbers2 = list(self.bins_bxgrid[identifier].keys())
            bin_numbers1.sort()
            bin_numbers2.sort()
            for bin_num_1 in bin_numbers1:
                for bin_num_2 in bin_numbers2:
                    if bin_num_1 != bin_num_2:
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_1])
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_2])
                        max_pairs = min(len(self.bins_bxgrid[identifier][bin_num_1]), len(self.bins_bxgrid[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            imposter_ind_max = 0
                            while imposter_ind_max == 0:
                                imposter_id = random.choice(bxgrid_identifiers_removed)
                                imposter_ind_max = len(self.bins_bxgrid[imposter_id][bin_num_2])
                            imagename1 = self.bins_bxgrid[identifier][bin_num_1][img_ind]
                            imagename2 = self.bins_bxgrid[identifier][bin_num_2][img_ind]
                            imagename3 = self.bins_bxgrid[imposter_id][bin_num_2][random.randrange(0, imposter_ind_max)]
                            imagename4 = self.bins_bxgrid[identifier][bin_num_2][random.randrange(0, len(self.bins_bxgrid[identifier][bin_num_2]))]
                            imagepath1 = os.path.join(self.image_dir_bxgrid, imagename1)
                            maskpath1 = os.path.join(self.mask_dir_bxgrid, imagename1)
                            imagepath2 = os.path.join(self.image_dir_bxgrid, imagename2)
                            maskpath2 = os.path.join(self.mask_dir_bxgrid, imagename2)
                            imagepath3 = os.path.join(self.image_dir_bxgrid, imagename3)
                            maskpath3 = os.path.join(self.mask_dir_bxgrid, imagename3)
                            imagepath4 = os.path.join(self.image_dir_bxgrid, imagename4)
                            maskpath4 = os.path.join(self.mask_dir_bxgrid, imagename4)
                            self.quadruplets.append([imagepath1, maskpath1, imagepath2, maskpath2, imagepath3, maskpath3, imagepath4, maskpath4, identifier, imposter_id])
                            
        print('Total quadruplets: ', len(self.quadruplets))

class QuadrupletFromBinAllDatasetBothUncropped(Dataset):
    def __init__(self, bins_path_wsd, parent_dir_wsd, bins_path_bxgrid, parent_dir_bxgrid, brightness_and_contrast = False, flip_data = True, res_mult = 1):
        super().__init__()
        self.flip_data = flip_data
        self.b_n_c = brightness_and_contrast
        self.input_size=(int(256*res_mult),int(256*res_mult))
        self.input_size_nocrop = (int(480*res_mult), int(640*res_mult))
        self.res_mult = res_mult
        
        self.image_dir_wsd = os.path.join(parent_dir_wsd, 'WarsawData2')
        self.mask_dir_wsd = os.path.join(parent_dir_wsd, 'WarsawMasksCoarse')        
        
        with open(bins_path_wsd, 'rb') as wsdbinsFile:
            self.bins_wsd = pkl.load(wsdbinsFile)
        
        self.image_dir_bxgrid = os.path.join(parent_dir_bxgrid, 'iris')
        self.mask_dir_bxgrid = os.path.join(parent_dir_bxgrid, 'masks_coarse')       
        
        with open(bins_path_bxgrid, 'rb') as bxgridbinsFile:
            self.bins_bxgrid = pkl.load(bxgridbinsFile)        
        
        self.all_ids = list(self.bins_wsd.keys()) + list(self.bins_bxgrid.keys())
        
        print('Initializing quadruplets for Warsaw Pupil Dynamics...')
        self.quadruplets = []
        self.genuine_ids = []
        self.imposter_ids = []
        wsd_identifiers = list(self.bins_wsd.keys())
        random.shuffle(wsd_identifiers)
        for identifier in tqdm(wsd_identifiers):
            wsd_identifiers_removed = wsd_identifiers.copy()
            wsd_identifiers_removed.remove(identifier)
            bin_numbers1 = list(self.bins_wsd[identifier].keys())
            bin_numbers2 = list(self.bins_wsd[identifier].keys())
            bin_numbers1.sort()
            bin_numbers2.sort()
            for bin_num_1 in bin_numbers1:
                for bin_num_2 in bin_numbers2:
                    if bin_num_1 != bin_num_2:
                        random.shuffle(self.bins_wsd[identifier][bin_num_1])
                        random.shuffle(self.bins_wsd[identifier][bin_num_2])
                        max_pairs = min(len(self.bins_wsd[identifier][bin_num_1]), len(self.bins_wsd[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            imposter_ind_max = 0
                            while imposter_ind_max == 0:
                                imposter_id = random.choice(wsd_identifiers_removed)
                                imposter_ind_max = len(self.bins_wsd[imposter_id][bin_num_2])
                            imagename1 = self.bins_wsd[identifier][bin_num_1][img_ind]
                            imagename2 = self.bins_wsd[identifier][bin_num_2][img_ind]
                            imagename3 = self.bins_wsd[imposter_id][bin_num_2][random.randrange(0, imposter_ind_max)]
                            imagename4 = self.bins_wsd[identifier][bin_num_2][random.randrange(0, len(self.bins_wsd[identifier][bin_num_2]))]
                            indiv_id1 = imagename1.split('_')[0]
                            indiv_id2 = imagename2.split('_')[0]
                            indiv_id3 = imagename3.split('_')[0]
                            indiv_id4 = imagename4.split('_')[0]
                            imagepath1 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id1), imagename1)
                            maskpath1 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id1), imagename1)
                            imagepath2 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id2), imagename2)
                            maskpath2 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id2), imagename2)
                            imagepath3 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id3), imagename3)
                            maskpath3 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id3), imagename3)
                            imagepath4 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id4), imagename4)
                            maskpath4 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id4), imagename4)
                            self.quadruplets.append([imagepath1, maskpath1, imagepath2, maskpath2, imagepath3, maskpath3, imagepath4, maskpath4, identifier, imposter_id])             
        
        print('Initializing quadruplets for BXGRID...')
        bxgrid_identifiers = list(self.bins_bxgrid.keys())
        random.shuffle(bxgrid_identifiers)
        for identifier in tqdm(bxgrid_identifiers):
            bxgrid_identifiers_removed = bxgrid_identifiers.copy()
            bxgrid_identifiers_removed.remove(identifier)
            bin_numbers1 = list(self.bins_bxgrid[identifier].keys())
            bin_numbers2 = list(self.bins_bxgrid[identifier].keys())
            bin_numbers1.sort()
            bin_numbers2.sort()
            for bin_num_1 in bin_numbers1:
                for bin_num_2 in bin_numbers2:
                    if bin_num_1 != bin_num_2:
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_1])
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_2])
                        max_pairs = min(len(self.bins_bxgrid[identifier][bin_num_1]), len(self.bins_bxgrid[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            imposter_ind_max = 0
                            while imposter_ind_max == 0:
                                imposter_id = random.choice(bxgrid_identifiers_removed)
                                imposter_ind_max = len(self.bins_bxgrid[imposter_id][bin_num_2])
                            imagename1 = self.bins_bxgrid[identifier][bin_num_1][img_ind]
                            imagename2 = self.bins_bxgrid[identifier][bin_num_2][img_ind]
                            imagename3 = self.bins_bxgrid[imposter_id][bin_num_2][random.randrange(0, imposter_ind_max)]
                            imagename4 = self.bins_bxgrid[identifier][bin_num_2][random.randrange(0, len(self.bins_bxgrid[identifier][bin_num_2]))]
                            imagepath1 = os.path.join(self.image_dir_bxgrid, imagename1)
                            maskpath1 = os.path.join(self.mask_dir_bxgrid, imagename1)
                            imagepath2 = os.path.join(self.image_dir_bxgrid, imagename2)
                            maskpath2 = os.path.join(self.mask_dir_bxgrid, imagename2)
                            imagepath3 = os.path.join(self.image_dir_bxgrid, imagename3)
                            maskpath3 = os.path.join(self.mask_dir_bxgrid, imagename3)
                            imagepath4 = os.path.join(self.image_dir_bxgrid, imagename4)
                            maskpath4 = os.path.join(self.mask_dir_bxgrid, imagename4)
                            self.quadruplets.append([imagepath1, maskpath1, imagepath2, maskpath2, imagepath3, maskpath3, imagepath4, maskpath4, identifier, imposter_id])
                            
        print('Total quadruplets: ', len(self.quadruplets))
        

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5, ))
        ])
        
        self.pupil_iris_xyrs_path_wsd = os.path.join(parent_dir_wsd, 'pupil_iris_xyrs_new.pkl')
        with open(self.pupil_iris_xyrs_path_wsd, 'rb') as pupilirisxyrsFile:
            self.pupil_iris_xyrs_wsd = pkl.load(pupilirisxyrsFile)
        
        self.pupil_iris_xyrs_path_bxgrid = os.path.join(parent_dir_bxgrid, 'pupil_iris_xyrs_new.pkl')
        with open(self.pupil_iris_xyrs_path_bxgrid, 'rb') as pupilirisxyrsFile:
            self.pupil_iris_xyrs_bxgrid = pkl.load(pupilirisxyrsFile)
        
        self.pupil_xyrs = {}
        self.iris_xyrs = {}
        
        for imagename in self.pupil_iris_xyrs_wsd.keys():
            imagepath = os.path.join(self.image_dir_wsd, imagename + '.bmp')
            self.pupil_xyrs[imagepath] = self.pupil_iris_xyrs_wsd[imagename]['pxyr']
            self.iris_xyrs[imagepath] = self.pupil_iris_xyrs_wsd[imagename]['ixyr']
        
        for imagename in self.pupil_iris_xyrs_bxgrid.keys():
            imagepath = os.path.join(self.image_dir_bxgrid, imagename + '.png')
            self.pupil_xyrs[imagepath] = self.pupil_iris_xyrs_bxgrid[imagename]['pxyr']
            self.iris_xyrs[imagepath] = self.pupil_iris_xyrs_bxgrid[imagename]['ixyr']
        
        self.heavy_augment_prob = 0.3
    
    def load_image(self, file):
        return Image.open(file).convert('L')
        
    def __getitem__(self, index):
    
        quadruplet = self.quadruplets[index]
        
        inp_img_path = quadruplet[0].strip()
        inp_mask_path = quadruplet[1].strip()
        tar_img_path = quadruplet[2].strip()
        tar_mask_path = quadruplet[3].strip()
        imp_img_path = quadruplet[4].strip()
        imp_mask_path = quadruplet[5].strip()
        tar2_img_path = quadruplet[6].strip()
        tar2_mask_path = quadruplet[7].strip()
        
        inp_img = self.load_image(inp_img_path)
        inp_mask = self.load_image(inp_mask_path)
        
        tar_img = self.load_image(tar_img_path)
        tar_mask = self.load_image(tar_mask_path)
        
        imp_img = self.load_image(imp_img_path)
        imp_mask = self.load_image(imp_mask_path)
        
        tar2_img = self.load_image(tar2_img_path)
        tar2_mask = self.load_image(tar2_mask_path)
        
        inp_img_pupil_xyr = torch.FloatTensor(self.pupil_xyrs[inp_img_path])
        inp_img_iris_xyr = torch.FloatTensor(self.iris_xyrs[inp_img_path])
        
        tar_img_pupil_xyr = torch.FloatTensor(self.pupil_xyrs[tar_img_path])
        tar_img_iris_xyr = torch.FloatTensor(self.iris_xyrs[tar_img_path])
        
        imp_img_pupil_xyr = torch.FloatTensor(self.pupil_xyrs[imp_img_path])
        imp_img_iris_xyr = torch.FloatTensor(self.iris_xyrs[imp_img_path])
        
        tar2_img_pupil_xyr = torch.FloatTensor(self.pupil_xyrs[tar2_img_path])
        tar2_img_iris_xyr = torch.FloatTensor(self.iris_xyrs[tar2_img_path])
        
        sw, sh = inp_img.size
        bw, bh = tar_img.size
        bw2, bh2 = tar2_img.size
        iw, ih = imp_img.size
        
        if self.flip_data and random.random() < 0.5:
            inp_img = inp_img.transpose(Image.FLIP_LEFT_RIGHT) 
            inp_mask = inp_mask.transpose(Image.FLIP_LEFT_RIGHT)
            tar_img = tar_img.transpose(Image.FLIP_LEFT_RIGHT) 
            tar_mask = tar_mask.transpose(Image.FLIP_LEFT_RIGHT)
            tar2_img = tar2_img.transpose(Image.FLIP_LEFT_RIGHT) 
            tar2_mask = tar2_mask.transpose(Image.FLIP_LEFT_RIGHT)  
            inp_img_pupil_xyr[0] = sw - inp_img_pupil_xyr[0]
            tar_img_pupil_xyr[0] = bw - tar_img_pupil_xyr[0]
            tar2_img_pupil_xyr[0] = bw2 - tar2_img_pupil_xyr[0]
            inp_img_iris_xyr[0] = sw - inp_img_iris_xyr[0]
            tar_img_iris_xyr[0] = bw - tar_img_iris_xyr[0]
            tar2_img_iris_xyr[0] = bw2 - tar2_img_iris_xyr[0]
        
        if self.flip_data and random.random() < 0.5:
            imp_img = imp_img.transpose(Image.FLIP_LEFT_RIGHT)
            imp_mask = imp_mask.transpose(Image.FLIP_LEFT_RIGHT)
            imp_img_pupil_xyr[0] = iw - imp_img_pupil_xyr[0]
            imp_img_iris_xyr[0] = iw - imp_img_iris_xyr[0]
        
        
        sw, sh = inp_img.size
        sh_mult = self.input_size_nocrop[0]/sh
        sw_mult = self.input_size_nocrop[1]/sw
        
        inp_img = inp_img.resize((self.input_size_nocrop[1], self.input_size_nocrop[0]))
        inp_mask = inp_mask.resize((self.input_size_nocrop[1], self.input_size_nocrop[0]))
        inp_img_pupil_xyr[0] = self.input_size_nocrop[1]/2 - sw_mult * (sw/2 - inp_img_pupil_xyr[0])
        inp_img_pupil_xyr[1] = self.input_size_nocrop[0]/2 - sh_mult * (sh/2 - inp_img_pupil_xyr[1])
        inp_img_pupil_xyr[2] = max(sw_mult, sh_mult) * inp_img_pupil_xyr[2]
        inp_img_iris_xyr[0] = self.input_size_nocrop[1]/2 - sw_mult * (sw/2 - inp_img_iris_xyr[0])
        inp_img_iris_xyr[1] = self.input_size_nocrop[0]/2 - sh_mult * (sh/2 - inp_img_iris_xyr[1])
        inp_img_iris_xyr[2] = max(sw_mult, sh_mult) * inp_img_iris_xyr[2]
        
        bw, bh = tar_img.size
        bh_mult = self.input_size_nocrop[0]/bh
        bw_mult = self.input_size_nocrop[1]/bw
        
        tar_img = tar_img.resize((self.input_size_nocrop[1], self.input_size_nocrop[0]))
        tar_mask = tar_mask.resize((self.input_size_nocrop[1], self.input_size_nocrop[0]))
        tar_img_pupil_xyr[0] = self.input_size_nocrop[1]/2 - bw_mult * (bw/2 - tar_img_pupil_xyr[0])
        tar_img_pupil_xyr[1] = self.input_size_nocrop[0]/2 - bh_mult * (bh/2 - tar_img_pupil_xyr[1])
        tar_img_pupil_xyr[2] = max(bw_mult, bh_mult) * tar_img_pupil_xyr[2]
        tar_img_iris_xyr[0] = self.input_size_nocrop[1]/2 - bw_mult * (bw/2 - tar_img_iris_xyr[0])
        tar_img_iris_xyr[1] = self.input_size_nocrop[0]/2 - bh_mult * (bh/2 - tar_img_iris_xyr[1])
        tar_img_iris_xyr[2] = max(bw_mult, bh_mult) * tar_img_iris_xyr[2]
        
        b2w, b2h = tar2_img.size
        b2h_mult = self.input_size_nocrop[0]/b2h
        b2w_mult = self.input_size_nocrop[1]/b2w
        
        tar2_img = tar2_img.resize((self.input_size_nocrop[1], self.input_size_nocrop[0]))
        tar2_mask = tar2_mask.resize((self.input_size_nocrop[1], self.input_size_nocrop[0]))
        tar2_img_pupil_xyr[0] = self.input_size_nocrop[1]/2 - b2w_mult * (b2w/2 - tar2_img_pupil_xyr[0])
        tar2_img_pupil_xyr[1] = self.input_size_nocrop[0]/2 - b2h_mult * (b2h/2 - tar2_img_pupil_xyr[1])
        tar2_img_pupil_xyr[2] = max(b2w_mult, b2h_mult) * tar2_img_pupil_xyr[2]
        tar2_img_iris_xyr[0] = self.input_size_nocrop[1]/2 - b2w_mult * (b2w/2 - tar2_img_iris_xyr[0])
        tar2_img_iris_xyr[1] = self.input_size_nocrop[0]/2 - b2h_mult * (b2h/2 - tar2_img_iris_xyr[1])
        tar2_img_iris_xyr[2] = max(b2w_mult, b2h_mult) * tar2_img_iris_xyr[2]
        
        iw, ih = imp_img.size
        ih_mult = self.input_size_nocrop[0]/ih
        iw_mult = self.input_size_nocrop[1]/iw
        
        imp_img = imp_img.resize((self.input_size_nocrop[1], self.input_size_nocrop[0]))
        imp_mask = imp_mask.resize((self.input_size_nocrop[1], self.input_size_nocrop[0]))
        imp_img_pupil_xyr[0] = self.input_size_nocrop[1]/2 - iw_mult * (iw/2 - imp_img_pupil_xyr[0])
        imp_img_pupil_xyr[1] = self.input_size_nocrop[0]/2 - ih_mult * (ih/2 - imp_img_pupil_xyr[1])
        imp_img_pupil_xyr[2] = max(iw_mult, ih_mult) * imp_img_pupil_xyr[2]
        imp_img_iris_xyr[0] = self.input_size_nocrop[1]/2 - iw_mult * (iw/2 - imp_img_iris_xyr[0])
        imp_img_iris_xyr[1] = self.input_size_nocrop[0]/2 - ih_mult * (ih/2 - imp_img_iris_xyr[1])
        imp_img_iris_xyr[2] = max(iw_mult, ih_mult) * imp_img_iris_xyr[2]
            
        if random.random() < self.heavy_augment_prob:
            if random.random() < 0.5:
                random_choice = np.random.choice([1,2,3,4,5])
                if random_choice == 1:   
                    # sharpening
                    random_degree = np.random.choice([1,2,3,4,5])
                    if random_degree == 1:
                        inp_img = inp_img.filter(ImageFilter.DETAIL)
                        tar_img = tar_img.filter(ImageFilter.DETAIL)
                    elif random_degree == 2:
                        inp_img = inp_img.filter(ImageFilter.SHARPEN)
                        tar_img = tar_img.filter(ImageFilter.SHARPEN)
                    elif random_degree == 3:
                        inp_img = inp_img.filter(ImageFilter.EDGE_ENHANCE)
                        tar_img = tar_img.filter(ImageFilter.EDGE_ENHANCE)
                    elif random_degree == 4:
                        inp_img = inp_img.filter(ImageFilter.EDGE_ENHANCE_MORE)
                        tar_img = tar_img.filter(ImageFilter.EDGE_ENHANCE_MORE)                          
                    else:
                        random_alpha = random.uniform(0.0, 0.3)
                        random_lightness = random.uniform(0.6, 1.0)
                        aug = iaa.Sharpen(alpha=random_alpha, lightness=random_lightness)
                        inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                        inp_img = Image.fromarray(inp_img_np[0])
                        tar_img_np = aug(images = np.expand_dims(np.array(tar_img), axis=0))
                        tar_img = Image.fromarray(tar_img_np[0])
                elif random_choice == 2:
                    #blurring
                    random_degree = np.random.choice([1,2,3,4,5])
                    if random_degree == 1:
                        inp_img = inp_img.filter(ImageFilter.GaussianBlur())
                        tar_img = tar_img.filter(ImageFilter.GaussianBlur())
                    elif random_degree == 2:
                        inp_img = inp_img.filter(ImageFilter.BLUR)
                        tar_img = tar_img.filter(ImageFilter.BLUR)
                    elif random_degree == 3:
                        inp_img = inp_img.filter(ImageFilter.SMOOTH)
                        tar_img = tar_img.filter(ImageFilter.SMOOTH)
                    elif random_degree == 4:
                        inp_img = inp_img.filter(ImageFilter.SMOOTH_MORE)
                        tar_img = tar_img.filter(ImageFilter.SMOOTH_MORE)
                    else:
                        random_degree = random.randint(-179, 180)
                        aug = iaa.MotionBlur(k=3, angle=random_degree)
                        inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                        inp_img = Image.fromarray(inp_img_np[0])
                        tar_img_np = aug(images = np.expand_dims(np.array(tar_img), axis=0))
                        tar_img = Image.fromarray(tar_img_np[0])
                elif random_choice == 3:
                    # Basic compression and expansion
                    divider = random.uniform(1.0, 1.5)
                    cw, ch = inp_img.size
                    new_cw = int(cw/divider)
                    new_ch = int(ch/divider)
                    
                    first_choice = np.random.choice([1,2,3,4,5,6])
                    if first_choice == 1:
                        inp_img = inp_img.resize((new_cw, new_ch), Image.NEAREST)
                        tar_img = tar_img.resize((new_cw, new_ch), Image.NEAREST)
                    elif first_choice == 2:
                        inp_img = inp_img.resize((new_cw, new_ch), Image.BILINEAR)
                        tar_img = tar_img.resize((new_cw, new_ch), Image.BILINEAR)
                    elif first_choice == 3:
                        inp_img = inp_img.resize((new_cw, new_ch), Image.BICUBIC)
                        tar_img = tar_img.resize((new_cw, new_ch), Image.BICUBIC)
                    elif first_choice == 4:
                        inp_img = inp_img.resize((new_cw, new_ch), Image.LANCZOS)
                        tar_img = tar_img.resize((new_cw, new_ch), Image.LANCZOS)
                    elif first_choice == 5:
                        inp_img = inp_img.resize((new_cw, new_ch), Image.HAMMING)
                        tar_img = tar_img.resize((new_cw, new_ch), Image.HAMMING)
                    else:
                        inp_img = inp_img.resize((new_cw, new_ch), Image.BOX)
                        tar_img = tar_img.resize((new_cw, new_ch), Image.BOX)
                    
                    second_choice = np.random.choice([1,2,3,4,5,6])
                    if second_choice == 1:
                        inp_img = inp_img.resize((cw, ch), Image.NEAREST)
                        tar_img = tar_img.resize((cw, ch), Image.NEAREST)
                    elif second_choice == 2:
                        inp_img = inp_img.resize((cw, ch), Image.BILINEAR)
                        tar_img = tar_img.resize((cw, ch), Image.BILINEAR)
                    elif second_choice == 3:
                        inp_img = inp_img.resize((cw, ch), Image.BICUBIC)
                        tar_img = tar_img.resize((cw, ch), Image.BICUBIC)
                    elif second_choice == 4:
                        inp_img = inp_img.resize((cw, ch), Image.LANCZOS)
                        tar_img = tar_img.resize((cw, ch), Image.LANCZOS)
                    elif second_choice == 5:
                        inp_img = inp_img.resize((cw, ch), Image.HAMMING)
                        tar_img = tar_img.resize((cw, ch), Image.HAMMING)
                    else:
                        inp_img = inp_img.resize((cw, ch), Image.BOX)
                        tar_img = tar_img.resize((cw, ch), Image.BOX)
                elif random_choice == 4:
                    #JPEG compression
                    random_compression = random.randint(10, 50)
                    aug = iaa.JpegCompression(compression=random_compression)
                    inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                    inp_img = Image.fromarray(inp_img_np[0])
                    tar_img_np = aug(images = np.expand_dims(np.array(tar_img), axis=0))
                    tar_img = Image.fromarray(tar_img_np[0])
                else:
                    if random.random() < 0.5:
                        aug = iaa.AdditiveGaussianNoise(scale=3)
                        inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                        inp_img = Image.fromarray(inp_img_np[0])
                        tar_img_np = aug(images = np.expand_dims(np.array(tar_img), axis=0))
                        tar_img = Image.fromarray(tar_img_np[0])
                    else:
                        aug = iaa.AdditiveLaplaceNoise(scale=3)
                        inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                        inp_img = Image.fromarray(inp_img_np[0])
                        tar_img_np = aug(images = np.expand_dims(np.array(tar_img), axis=0))
                        tar_img = Image.fromarray(tar_img_np[0])
                       
            if random.random() < 0.5: # random contrast change
                random_degree = np.random.choice([1,2,3,4,5,6,7,8])                
                if random_degree == 1:
                    aug = iaa.GammaContrast((0.5, 2.0))
                elif random_degree == 2:
                    aug = iaa.LinearContrast((0.4, 1.6))
                elif random_degree == 3:
                    aug = iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6))
                elif random_degree == 4:
                    aug = iaa.LogContrast(gain=(0.6, 1.4)) 
                else:    
                    aug = iaa.pillike.EnhanceBrightness()
                inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                inp_img = Image.fromarray(inp_img_np[0])
                random_degree = np.random.choice([1,2,3,4,5,6,7,8])                
                if random_degree == 1:
                    aug = iaa.GammaContrast((0.5, 2.0))
                elif random_degree == 2:
                    aug = iaa.LinearContrast((0.4, 1.6))
                elif random_degree == 3:
                    aug = iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6))
                elif random_degree == 4:
                    aug = iaa.LogContrast(gain=(0.6, 1.4)) 
                else:    
                    aug = iaa.pillike.EnhanceBrightness()
                tar_img_np = aug(images = np.expand_dims(np.array(tar_img), axis=0))
                tar_img = Image.fromarray(tar_img_np[0])
        
        if random.random() < self.heavy_augment_prob:
            if random.random() < 0.5:
                random_choice = np.random.choice([1,2,3,4,5])
                if random_choice == 1:   
                    # sharpening
                    random_degree = np.random.choice([1,2,3,4,5])
                    if random_degree == 1:
                        imp_img = imp_img.filter(ImageFilter.DETAIL)
                    elif random_degree == 2:
                        imp_img = imp_img.filter(ImageFilter.SHARPEN)
                    elif random_degree == 3:
                        imp_img = imp_img.filter(ImageFilter.EDGE_ENHANCE)
                    elif random_degree == 4:
                        imp_img = imp_img.filter(ImageFilter.EDGE_ENHANCE_MORE)                            
                    else:
                        aug = iaa.Sharpen(alpha=(0.0, 0.3), lightness=(0.6, 1.0))
                        imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                        imp_img = Image.fromarray(imp_img_np[0])
                elif random_choice == 2:
                    #blurring
                    random_degree = np.random.choice([1,2,3,4,5])
                    if random_degree == 1:
                        imp_img = imp_img.filter(ImageFilter.GaussianBlur())
                    elif random_degree == 2:
                        imp_img = imp_img.filter(ImageFilter.BLUR)
                    elif random_degree == 3:
                        imp_img = imp_img.filter(ImageFilter.SMOOTH)
                    elif random_degree == 4:
                        imp_img = imp_img.filter(ImageFilter.SMOOTH_MORE)
                    else:
                        aug = iaa.MotionBlur(k=3)
                        imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                        imp_img = Image.fromarray(imp_img_np[0])
                elif random_choice == 3:
                    # Basic compression and expansion
                    divider = random.uniform(1.0, 1.5)
                    cw, ch = imp_img.size
                    new_cw = int(cw/divider)
                    new_ch = int(ch/divider)
                    
                    first_choice = np.random.choice([1,2,3,4,5,6])
                    if first_choice == 1:
                        imp_img = imp_img.resize((new_cw, new_ch), Image.NEAREST)
                    elif first_choice == 2:
                        imp_img = imp_img.resize((new_cw, new_ch), Image.BILINEAR)
                    elif first_choice == 3:
                        imp_img = imp_img.resize((new_cw, new_ch), Image.BICUBIC)
                    elif first_choice == 4:
                        imp_img = imp_img.resize((new_cw, new_ch), Image.LANCZOS)
                    elif first_choice == 5:
                        imp_img = imp_img.resize((new_cw, new_ch), Image.HAMMING)
                    else:
                        imp_img = imp_img.resize((new_cw, new_ch), Image.BOX)
                    
                    second_choice = np.random.choice([1,2,3,4,5,6])
                    if second_choice == 1:
                        imp_img = imp_img.resize((cw, ch), Image.NEAREST)
                    elif second_choice == 2:
                        imp_img = imp_img.resize((cw, ch), Image.BILINEAR)
                    elif second_choice == 3:
                        imp_img = imp_img.resize((cw, ch), Image.BICUBIC)
                    elif second_choice == 4:
                        imp_img = imp_img.resize((cw, ch), Image.LANCZOS)
                    elif second_choice == 5:
                        imp_img = imp_img.resize((cw, ch), Image.HAMMING)
                    else:
                        imp_img = imp_img.resize((cw, ch), Image.BOX)
                elif random_choice == 4:
                    #JPEG compression
                    aug = iaa.JpegCompression(compression=(10, 50))
                    imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                    imp_img = Image.fromarray(imp_img_np[0])
                else:
                    if random.random() < 0.5:
                        aug = iaa.AdditiveGaussianNoise(scale=3)
                        imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                        imp_img = Image.fromarray(imp_img_np[0])
                    else:
                        aug = iaa.AdditiveLaplaceNoise(scale=3)
                        imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                        imp_img = Image.fromarray(imp_img_np[0])
                
                         
            if random.random() < 0.5: # random contrast change
                random_degree = np.random.choice([1,2,3,4,5,6,7,8])                
                if random_degree == 1:
                    aug = iaa.GammaContrast((0.5, 2.0))
                    imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                    imp_img = Image.fromarray(imp_img_np[0])
                elif random_degree == 2:
                    aug = iaa.LinearContrast((0.4, 1.6))
                    imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                    imp_img = Image.fromarray(imp_img_np[0])
                elif random_degree == 3:
                    aug = iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6))
                    imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                    imp_img = Image.fromarray(imp_img_np[0])
                elif random_degree == 4:
                    aug = iaa.LogContrast(gain=(0.6, 1.4))
                    imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                    imp_img = Image.fromarray(imp_img_np[0])            
                else:    
                    aug = iaa.pillike.EnhanceBrightness()
                    imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                    imp_img = Image.fromarray(imp_img_np[0])
                    
        inp_img_t = torch.from_numpy(np.float32(inp_img)).float()
        xform_inp_img = inp_img_t.unsqueeze(0)
        
        xform_inp_mask = self.transform(inp_mask)
        xform_inp_mask[xform_inp_mask<0] = -1
        xform_inp_mask[xform_inp_mask>=0] = 1
        
        tar_img_t = torch.from_numpy(np.float32(tar_img)).float()
        xform_tar_img = tar_img_t.unsqueeze(0)
        
        xform_tar_mask = self.transform(tar_mask)
        xform_tar_mask[xform_tar_mask<0] = -1.0
        xform_tar_mask[xform_tar_mask>=0] = 1.0
        
        tar2_img_t = torch.from_numpy(np.float32(tar2_img)).float()
        xform_tar2_img = tar2_img_t.unsqueeze(0)
        
        xform_tar2_mask = self.transform(tar2_mask)
        xform_tar2_mask[xform_tar2_mask<0] = -1.0
        xform_tar2_mask[xform_tar2_mask>=0] = 1.0
        
        imp_img_t = torch.from_numpy(np.float32(imp_img)).float()
        xform_imp_img = imp_img_t.unsqueeze(0)
        
        xform_imp_mask = self.transform(imp_mask)
        xform_imp_mask[xform_imp_mask<0] = -1.0
        xform_imp_mask[xform_imp_mask>=0] = 1.0
        return {'inp_img': xform_inp_img, 'tar_img' : xform_tar_img, 'tar2_img' : xform_tar2_img, 'imp_img' : xform_imp_img, 'inp_mask': xform_inp_mask, 'tar_mask' : xform_tar_mask, 'tar2_mask' : xform_tar2_mask, 'imp_mask' : xform_imp_mask, 'inp_img_pxyr' : inp_img_pupil_xyr, 'tar_img_pxyr' : tar_img_pupil_xyr, 'tar2_img_pxyr' : tar2_img_pupil_xyr, 'imp_img_pxyr' : imp_img_pupil_xyr, 'inp_img_ixyr': inp_img_iris_xyr, 'tar_img_ixyr' :  tar_img_iris_xyr, 'tar2_img_ixyr' :  tar2_img_iris_xyr, 'imp_img_ixyr' : imp_img_iris_xyr, 'identifier' : quadruplet[8], 'imposter_id' : quadruplet[9]}

  
    def __len__(self):
        return len(self.quadruplets)
    
    def reset(self):
        print('Shuffling quadruplets for WSD...')
        self.quadruplets = []
        self.genuine_ids = []
        self.imposter_ids = []
        wsd_identifiers = list(self.bins_wsd.keys())
        random.shuffle(wsd_identifiers)
        for identifier in tqdm(wsd_identifiers):
            wsd_identifiers_removed = wsd_identifiers.copy()
            wsd_identifiers_removed.remove(identifier)
            bin_numbers1 = list(self.bins_wsd[identifier].keys())
            bin_numbers2 = list(self.bins_wsd[identifier].keys())
            bin_numbers1.sort()
            bin_numbers2.sort()
            for bin_num_1 in bin_numbers1:
                for bin_num_2 in bin_numbers2:
                    if bin_num_1 != bin_num_2:
                        random.shuffle(self.bins_wsd[identifier][bin_num_1])
                        random.shuffle(self.bins_wsd[identifier][bin_num_2])
                        max_pairs = min(len(self.bins_wsd[identifier][bin_num_1]), len(self.bins_wsd[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            imposter_ind_max = 0
                            while imposter_ind_max == 0:
                                imposter_id = random.choice(wsd_identifiers_removed)
                                imposter_ind_max = len(self.bins_wsd[imposter_id][bin_num_2])
                            imagename1 = self.bins_wsd[identifier][bin_num_1][img_ind]
                            imagename2 = self.bins_wsd[identifier][bin_num_2][img_ind]
                            imagename3 = self.bins_wsd[imposter_id][bin_num_2][random.randrange(0, imposter_ind_max)]
                            imagename4 = self.bins_wsd[identifier][bin_num_2][random.randrange(0, len(self.bins_wsd[identifier][bin_num_2]))]
                            indiv_id1 = imagename1.split('_')[0]
                            indiv_id2 = imagename2.split('_')[0]
                            indiv_id3 = imagename3.split('_')[0]
                            indiv_id4 = imagename4.split('_')[0]
                            imagepath1 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id1), imagename1)
                            maskpath1 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id1), imagename1)
                            imagepath2 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id2), imagename2)
                            maskpath2 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id2), imagename2)
                            imagepath3 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id3), imagename3)
                            maskpath3 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id3), imagename3)
                            imagepath4 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id4), imagename4)
                            maskpath4 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id4), imagename4)
                            self.quadruplets.append([imagepath1, maskpath1, imagepath2, maskpath2, imagepath3, maskpath3, imagepath4, maskpath4, identifier, imposter_id])             
        
        print('Shuffling quadruplets for BXGRID...')
        bxgrid_identifiers = list(self.bins_bxgrid.keys())
        random.shuffle(bxgrid_identifiers)
        for identifier in tqdm(bxgrid_identifiers):
            bxgrid_identifiers_removed = bxgrid_identifiers.copy()
            bxgrid_identifiers_removed.remove(identifier)
            bin_numbers1 = list(self.bins_bxgrid[identifier].keys())
            bin_numbers2 = list(self.bins_bxgrid[identifier].keys())
            bin_numbers1.sort()
            bin_numbers2.sort()
            for bin_num_1 in bin_numbers1:
                for bin_num_2 in bin_numbers2:
                    if bin_num_1 != bin_num_2:
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_1])
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_2])
                        max_pairs = min(len(self.bins_bxgrid[identifier][bin_num_1]), len(self.bins_bxgrid[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            imposter_ind_max = 0
                            while imposter_ind_max == 0:
                                imposter_id = random.choice(bxgrid_identifiers_removed)
                                imposter_ind_max = len(self.bins_bxgrid[imposter_id][bin_num_2])
                            imagename1 = self.bins_bxgrid[identifier][bin_num_1][img_ind]
                            imagename2 = self.bins_bxgrid[identifier][bin_num_2][img_ind]
                            imagename3 = self.bins_bxgrid[imposter_id][bin_num_2][random.randrange(0, imposter_ind_max)]
                            imagename4 = self.bins_bxgrid[identifier][bin_num_2][random.randrange(0, len(self.bins_bxgrid[identifier][bin_num_2]))]
                            imagepath1 = os.path.join(self.image_dir_bxgrid, imagename1)
                            maskpath1 = os.path.join(self.mask_dir_bxgrid, imagename1)
                            imagepath2 = os.path.join(self.image_dir_bxgrid, imagename2)
                            maskpath2 = os.path.join(self.mask_dir_bxgrid, imagename2)
                            imagepath3 = os.path.join(self.image_dir_bxgrid, imagename3)
                            maskpath3 = os.path.join(self.mask_dir_bxgrid, imagename3)
                            imagepath4 = os.path.join(self.image_dir_bxgrid, imagename4)
                            maskpath4 = os.path.join(self.mask_dir_bxgrid, imagename4)
                            self.quadruplets.append([imagepath1, maskpath1, imagepath2, maskpath2, imagepath3, maskpath3, imagepath4, maskpath4, identifier, imposter_id])
                            
        print('Total quadruplets: ', len(self.quadruplets))

class QuadrupletFromBinAllDatasetBothPolarized(Dataset):
    def __init__(self, bins_path_wsd, parent_dir_wsd, bins_path_bxgrid, parent_dir_bxgrid, iris_crop=True, normalize=True, brightness_and_contrast = False, flip_data = True, res_mult=1):
        super().__init__()
        self.iris_crop = iris_crop
        self.flip_data = flip_data
        self.b_n_c = brightness_and_contrast
        self.input_size=(int(256*res_mult),int(256*res_mult))
        self.res_mult = res_mult
        self.normalize = normalize
        
        self.image_dir_wsd = os.path.join(parent_dir_wsd, 'WarsawDataPolar')
        self.mask_dir_wsd = os.path.join(parent_dir_wsd, 'WarsawMaskPolar')            
        
        with open(bins_path_wsd, 'rb') as wsdbinsFile:
            self.bins_wsd = pkl.load(wsdbinsFile)
        
        self.image_dir_bxgrid = os.path.join(parent_dir_bxgrid, 'images_polar')
        self.mask_dir_bxgrid = os.path.join(parent_dir_bxgrid, 'masks_polar')       
        
        with open(bins_path_bxgrid, 'rb') as bxgridbinsFile:
            self.bins_bxgrid = pkl.load(bxgridbinsFile)
        
        self.all_ids = list(self.bins_wsd.keys()) + list(self.bins_bxgrid.keys())
        
        print('Initializing quadruplets for Warsaw Pupil Dynamics...')
        self.quadruplets = []
        self.genuine_ids = []
        self.imposter_ids = []
        wsd_identifiers = list(self.bins_wsd.keys())
        random.shuffle(wsd_identifiers)
        for identifier in tqdm(wsd_identifiers):
            wsd_identifiers_removed = wsd_identifiers.copy()
            wsd_identifiers_removed.remove(identifier)
            bin_numbers1 = list(self.bins_wsd[identifier].keys())
            bin_numbers2 = list(self.bins_wsd[identifier].keys())
            bin_numbers1.sort()
            bin_numbers2.sort()
            for bin_num_1 in bin_numbers1:
                for bin_num_2 in bin_numbers2:
                    if bin_num_1 != bin_num_2:
                        random.shuffle(self.bins_wsd[identifier][bin_num_1])
                        random.shuffle(self.bins_wsd[identifier][bin_num_2])
                        max_pairs = min(len(self.bins_wsd[identifier][bin_num_1]), len(self.bins_wsd[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            imposter_ind_max = 0
                            while imposter_ind_max == 0:
                                imposter_id = random.choice(wsd_identifiers_removed)
                                imposter_ind_max = len(self.bins_wsd[imposter_id][bin_num_2])
                            imagename1 = self.bins_wsd[identifier][bin_num_1][img_ind]
                            imagename2 = self.bins_wsd[identifier][bin_num_2][img_ind]
                            imagename3 = self.bins_wsd[imposter_id][bin_num_2][random.randrange(0, imposter_ind_max)]
                            imagename4 = self.bins_wsd[identifier][bin_num_2][random.randrange(0, len(self.bins_wsd[identifier][bin_num_2]))]
                            indiv_id1 = imagename1.split('_')[0]
                            indiv_id2 = imagename2.split('_')[0]
                            indiv_id3 = imagename3.split('_')[0]
                            indiv_id4 = imagename4.split('_')[0]
                            imagepath1 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id1), imagename1)
                            maskpath1 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id1), imagename1)
                            imagepath2 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id2), imagename2)
                            maskpath2 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id2), imagename2)
                            imagepath3 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id3), imagename3)
                            maskpath3 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id3), imagename3)
                            imagepath4 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id4), imagename4)
                            maskpath4 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id4), imagename4)
                            self.quadruplets.append([imagepath1, maskpath1, imagepath2, maskpath2, imagepath3, maskpath3, imagepath4, maskpath4, identifier, imposter_id])             
        
        print('Initializing quadruplets for BXGRID...')
        bxgrid_identifiers = list(self.bins_bxgrid.keys())
        random.shuffle(bxgrid_identifiers)
        for identifier in tqdm(bxgrid_identifiers):
            bxgrid_identifiers_removed = bxgrid_identifiers.copy()
            bxgrid_identifiers_removed.remove(identifier)
            bin_numbers1 = list(self.bins_bxgrid[identifier].keys())
            bin_numbers2 = list(self.bins_bxgrid[identifier].keys())
            bin_numbers1.sort()
            bin_numbers2.sort()
            for bin_num_1 in bin_numbers1:
                for bin_num_2 in bin_numbers2:
                    if bin_num_1 != bin_num_2:
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_1])
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_2])
                        max_pairs = min(len(self.bins_bxgrid[identifier][bin_num_1]), len(self.bins_bxgrid[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            imposter_ind_max = 0
                            while imposter_ind_max == 0:
                                imposter_id = random.choice(bxgrid_identifiers_removed)
                                imposter_ind_max = len(self.bins_bxgrid[imposter_id][bin_num_2])
                            imagename1 = self.bins_bxgrid[identifier][bin_num_1][img_ind]
                            imagename2 = self.bins_bxgrid[identifier][bin_num_2][img_ind]
                            imagename3 = self.bins_bxgrid[imposter_id][bin_num_2][random.randrange(0, imposter_ind_max)]
                            imagename4 = self.bins_bxgrid[identifier][bin_num_2][random.randrange(0, len(self.bins_bxgrid[identifier][bin_num_2]))]
                            imagepath1 = os.path.join(self.image_dir_bxgrid, imagename1)
                            maskpath1 = os.path.join(self.mask_dir_bxgrid, imagename1)
                            imagepath2 = os.path.join(self.image_dir_bxgrid, imagename2)
                            maskpath2 = os.path.join(self.mask_dir_bxgrid, imagename2)
                            imagepath3 = os.path.join(self.image_dir_bxgrid, imagename3)
                            maskpath3 = os.path.join(self.mask_dir_bxgrid, imagename3)
                            imagepath4 = os.path.join(self.image_dir_bxgrid, imagename4)
                            maskpath4 = os.path.join(self.mask_dir_bxgrid, imagename4)
                            self.quadruplets.append([imagepath1, maskpath1, imagepath2, maskpath2, imagepath3, maskpath3, imagepath4, maskpath4, identifier, imposter_id])
                            
        print('Total quadruplets: ', len(self.quadruplets))
        

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5, ))
        ])
        
        self.pupil_iris_xyrs_path_wsd = os.path.join(parent_dir_wsd, 'pupil_iris_xyrs_new.pkl')
        with open(self.pupil_iris_xyrs_path_wsd, 'rb') as pupilirisxyrsFile:
            self.pupil_iris_xyrs_wsd = pkl.load(pupilirisxyrsFile)
        
        self.pupil_iris_xyrs_path_bxgrid = os.path.join(parent_dir_bxgrid, 'pupil_iris_xyrs_new.pkl')
        with open(self.pupil_iris_xyrs_path_bxgrid, 'rb') as pupilirisxyrsFile:
            self.pupil_iris_xyrs_bxgrid = pkl.load(pupilirisxyrsFile)
            
        print(self.pupil_iris_xyrs_wsd)
        print(self.pupil_iris_xyrs_bxgrid)
        
        self.pupil_xyrs = {}
        self.iris_xyrs = {}
        
        for imagename in self.pupil_iris_xyrs_wsd.keys():
            indiv_id = imagename.split('_')[0]
            imagepath = os.path.join(os.path.join(self.image_dir_wsd, indiv_id), imagename)
            self.pupil_xyrs[imagepath] = self.pupil_iris_xyrs_wsd[imagename]['pxyr']
            self.iris_xyrs[imagepath] = self.pupil_iris_xyrs_wsd[imagename]['ixyr']
        
        for imagename in self.pupil_iris_xyrs_bxgrid.keys():
            imagepath = os.path.join(self.image_dir_bxgrid, imagename)
            self.pupil_xyrs[imagepath] = self.pupil_iris_xyrs_bxgrid[imagename]['pxyr']
            self.iris_xyrs[imagepath] = self.pupil_iris_xyrs_bxgrid[imagename]['ixyr']
        
        self.heavy_augment_prob = 0.3
    
    def load_image(self, file):
        return Image.open(file).convert('L')
        
    def __getitem__(self, index):
    
        quadruplet = self.quadruplets[index]
        
        inp_img_path = quadruplet[0].strip()
        inp_mask_path = quadruplet[1].strip()
        tar_img_path = quadruplet[2].strip()
        tar_mask_path = quadruplet[3].strip()
        imp_img_path = quadruplet[4].strip()
        imp_mask_path = quadruplet[5].strip()
        tar2_img_path = quadruplet[6].strip()
        tar2_mask_path = quadruplet[7].strip()
        
        inp_img = self.load_image(inp_img_path)
        inp_mask = self.load_image(inp_mask_path)
        
        tar_img = self.load_image(tar_img_path)
        tar_mask = self.load_image(tar_mask_path)
        
        imp_img = self.load_image(imp_img_path)
        imp_mask = self.load_image(imp_mask_path)
        
        tar2_img = self.load_image(tar2_img_path)
        tar2_mask = self.load_image(tar2_mask_path)
        
        inp_img_pupil_xyr = torch.FloatTensor(self.pupil_xyrs[inp_img_path])
        inp_img_iris_xyr = torch.FloatTensor(self.iris_xyrs[inp_img_path])
        
        tar_img_pupil_xyr = torch.FloatTensor(self.pupil_xyrs[tar_img_path])
        tar_img_iris_xyr = torch.FloatTensor(self.iris_xyrs[tar_img_path])
        
        imp_img_pupil_xyr = torch.FloatTensor(self.pupil_xyrs[imp_img_path])
        imp_img_iris_xyr = torch.FloatTensor(self.iris_xyrs[imp_img_path])
        
        tar2_img_pupil_xyr = torch.FloatTensor(self.pupil_xyrs[tar2_img_path])
        tar2_img_iris_xyr = torch.FloatTensor(self.iris_xyrs[tar2_img_path])
        
        '''
        sw, sh = inp_img.size
        sw_mult = (sw/320)
        sh_mult = (self.input_size[0]/sh)
        
        bw, bh = tar_img.size
        bw_mult = (self.input_size[1]/bw)
        bh_mult = (self.input_size[0]/bh)
        
        biw, bih = imp_img.size
        biw_mult = (self.input_size[1]/biw)
        bih_mult = (self.input_size[0]/bih)
        
        inp_img_pupil_xyr[0] = (inp_img_pupil_xyr[0] * sw_mult)
        inp_img_pupil_xyr[1] = (inp_img_pupil_xyr[1] * sh_mult)
        inp_img_pupil_xyr[2] = (inp_img_pupil_xyr[2] * max(sw_mult, sh_mult))
        
        tar_img_pupil_xyr[0] = (tar_img_pupil_xyr[0] * bw_mult)
        tar_img_pupil_xyr[1] = (tar_img_pupil_xyr[1] * bh_mult)
        tar_img_pupil_xyr[2] = (tar_img_pupil_xyr[2] * max(bw_mult, bh_mult))
        
        imp_img_pupil_xyr[0] = (imp_img_pupil_xyr[0] * biw_mult)
        imp_img_pupil_xyr[1] = (imp_img_pupil_xyr[1] * bih_mult)
        imp_img_pupil_xyr[2] = (imp_img_pupil_xyr[2] * max(biw_mult, bih_mult))
    
        inp_img_iris_xyr[0] = (inp_img_iris_xyr[0] * sw_mult)
        inp_img_iris_xyr[1] = (inp_img_iris_xyr[1] * sh_mult)
        inp_img_iris_xyr[2] = (inp_img_iris_xyr[2] * max(sw_mult, sh_mult))
        
        tar_img_iris_xyr[0] = (tar_img_iris_xyr[0] * bw_mult)
        tar_img_iris_xyr[1] = (tar_img_iris_xyr[1] * bh_mult)
        tar_img_iris_xyr[2] = (tar_img_iris_xyr[2] * max(bw_mult, bh_mult))
        
        imp_img_iris_xyr[0] = (imp_img_iris_xyr[0] * biw_mult)
        imp_img_iris_xyr[1] = (imp_img_iris_xyr[1] * bih_mult)
        imp_img_iris_xyr[2] = (imp_img_iris_xyr[2] * max(biw_mult, bih_mult))
        '''
        sw, sh = inp_img.size
        bw, bh = tar_img.size
        bw2, bh2 = tar2_img.size
        iw, ih = imp_img.size
        
        if random.random() < self.heavy_augment_prob:
            if random.random() < 0.5:
                random_choice = np.random.choice([1,2,3,4,5])
                if random_choice == 1:   
                    # sharpening
                    random_degree = np.random.choice([1,2,3,4,5])
                    if random_degree == 1:
                        inp_img = inp_img.filter(ImageFilter.DETAIL)
                        tar_img = tar_img.filter(ImageFilter.DETAIL)
                    elif random_degree == 2:
                        inp_img = inp_img.filter(ImageFilter.SHARPEN)
                        tar_img = tar_img.filter(ImageFilter.SHARPEN)
                    elif random_degree == 3:
                        inp_img = inp_img.filter(ImageFilter.EDGE_ENHANCE)
                        tar_img = tar_img.filter(ImageFilter.EDGE_ENHANCE)
                    elif random_degree == 4:
                        inp_img = inp_img.filter(ImageFilter.EDGE_ENHANCE_MORE)
                        tar_img = tar_img.filter(ImageFilter.EDGE_ENHANCE_MORE)                          
                    else:
                        random_alpha = random.uniform(0.0, 0.3)
                        random_lightness = random.uniform(0.6, 1.0)
                        aug = iaa.Sharpen(alpha=random_alpha, lightness=random_lightness)
                        inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                        inp_img = Image.fromarray(inp_img_np[0])
                        tar_img_np = aug(images = np.expand_dims(np.array(tar_img), axis=0))
                        tar_img = Image.fromarray(tar_img_np[0])
                elif random_choice == 2:
                    #blurring
                    random_degree = np.random.choice([1,2,3,4,5])
                    if random_degree == 1:
                        inp_img = inp_img.filter(ImageFilter.GaussianBlur())
                        tar_img = tar_img.filter(ImageFilter.GaussianBlur())
                    elif random_degree == 2:
                        inp_img = inp_img.filter(ImageFilter.BLUR)
                        tar_img = tar_img.filter(ImageFilter.BLUR)
                    elif random_degree == 3:
                        inp_img = inp_img.filter(ImageFilter.SMOOTH)
                        tar_img = tar_img.filter(ImageFilter.SMOOTH)
                    elif random_degree == 4:
                        inp_img = inp_img.filter(ImageFilter.SMOOTH_MORE)
                        tar_img = tar_img.filter(ImageFilter.SMOOTH_MORE)
                    else:
                        random_degree = random.randint(-179, 180)
                        aug = iaa.MotionBlur(k=3, angle=random_degree)
                        inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                        inp_img = Image.fromarray(inp_img_np[0])
                        tar_img_np = aug(images = np.expand_dims(np.array(tar_img), axis=0))
                        tar_img = Image.fromarray(tar_img_np[0])
                elif random_choice == 3:
                    # Basic compression and expansion
                    divider = random.uniform(1.0, 1.5)
                    cw, ch = inp_img.size
                    new_cw = int(cw/divider)
                    new_ch = int(ch/divider)
                    
                    first_choice = np.random.choice([1,2,3,4,5,6])
                    if first_choice == 1:
                        inp_img = inp_img.resize((new_cw, new_ch), Image.NEAREST)
                        tar_img = tar_img.resize((new_cw, new_ch), Image.NEAREST)
                    elif first_choice == 2:
                        inp_img = inp_img.resize((new_cw, new_ch), Image.BILINEAR)
                        tar_img = tar_img.resize((new_cw, new_ch), Image.BILINEAR)
                    elif first_choice == 3:
                        inp_img = inp_img.resize((new_cw, new_ch), Image.BICUBIC)
                        tar_img = tar_img.resize((new_cw, new_ch), Image.BICUBIC)
                    elif first_choice == 4:
                        inp_img = inp_img.resize((new_cw, new_ch), Image.LANCZOS)
                        tar_img = tar_img.resize((new_cw, new_ch), Image.LANCZOS)
                    elif first_choice == 5:
                        inp_img = inp_img.resize((new_cw, new_ch), Image.HAMMING)
                        tar_img = tar_img.resize((new_cw, new_ch), Image.HAMMING)
                    else:
                        inp_img = inp_img.resize((new_cw, new_ch), Image.BOX)
                        tar_img = tar_img.resize((new_cw, new_ch), Image.BOX)
                    
                    second_choice = np.random.choice([1,2,3,4,5,6])
                    if second_choice == 1:
                        inp_img = inp_img.resize((cw, ch), Image.NEAREST)
                        tar_img = tar_img.resize((cw, ch), Image.NEAREST)
                    elif second_choice == 2:
                        inp_img = inp_img.resize((cw, ch), Image.BILINEAR)
                        tar_img = tar_img.resize((cw, ch), Image.BILINEAR)
                    elif second_choice == 3:
                        inp_img = inp_img.resize((cw, ch), Image.BICUBIC)
                        tar_img = tar_img.resize((cw, ch), Image.BICUBIC)
                    elif second_choice == 4:
                        inp_img = inp_img.resize((cw, ch), Image.LANCZOS)
                        tar_img = tar_img.resize((cw, ch), Image.LANCZOS)
                    elif second_choice == 5:
                        inp_img = inp_img.resize((cw, ch), Image.HAMMING)
                        tar_img = tar_img.resize((cw, ch), Image.HAMMING)
                    else:
                        inp_img = inp_img.resize((cw, ch), Image.BOX)
                        tar_img = tar_img.resize((cw, ch), Image.BOX)
                elif random_choice == 4:
                    #JPEG compression
                    random_compression = random.randint(10, 50)
                    aug = iaa.JpegCompression(compression=random_compression)
                    inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                    inp_img = Image.fromarray(inp_img_np[0])
                    tar_img_np = aug(images = np.expand_dims(np.array(tar_img), axis=0))
                    tar_img = Image.fromarray(tar_img_np[0])
                else:
                    if random.random() < 0.5:
                        aug = iaa.AdditiveGaussianNoise(scale=3)
                        inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                        inp_img = Image.fromarray(inp_img_np[0])
                        tar_img_np = aug(images = np.expand_dims(np.array(tar_img), axis=0))
                        tar_img = Image.fromarray(tar_img_np[0])
                    else:
                        aug = iaa.AdditiveLaplaceNoise(scale=3)
                        inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                        inp_img = Image.fromarray(inp_img_np[0])
                        tar_img_np = aug(images = np.expand_dims(np.array(tar_img), axis=0))
                        tar_img = Image.fromarray(tar_img_np[0])
                       
            if random.random() < 0.5: # random contrast change
                
                random_degree = np.random.choice([1,2,3,4,5,6,7,8])                
                if random_degree == 1:
                    aug = iaa.GammaContrast((0.5, 2.0))
                elif random_degree == 2:
                    aug = iaa.LinearContrast((0.4, 1.6))
                elif random_degree == 3:
                    aug = iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6))
                elif random_degree == 4:
                    aug = iaa.LogContrast(gain=(0.6, 1.4)) 
                else:    
                    aug = iaa.pillike.EnhanceBrightness()
                inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                inp_img = Image.fromarray(inp_img_np[0])
                
                random_degree = np.random.choice([1,2,3,4,5,6,7,8])                
                if random_degree == 1:
                    aug = iaa.GammaContrast((0.5, 2.0))
                elif random_degree == 2:
                    aug = iaa.LinearContrast((0.4, 1.6))
                elif random_degree == 3:
                    aug = iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6))
                elif random_degree == 4:
                    aug = iaa.LogContrast(gain=(0.6, 1.4)) 
                else:    
                    aug = iaa.pillike.EnhanceBrightness()
                tar_img_np = aug(images = np.expand_dims(np.array(tar_img), axis=0))
                tar_img = Image.fromarray(tar_img_np[0])
        
        if random.random() < self.heavy_augment_prob:
            if random.random() < 0.5:
                random_choice = np.random.choice([1,2,3,4,5])
                if random_choice == 1:   
                    # sharpening
                    random_degree = np.random.choice([1,2,3,4,5])
                    if random_degree == 1:
                        imp_img = imp_img.filter(ImageFilter.DETAIL)
                    elif random_degree == 2:
                        imp_img = imp_img.filter(ImageFilter.SHARPEN)
                    elif random_degree == 3:
                        imp_img = imp_img.filter(ImageFilter.EDGE_ENHANCE)
                    elif random_degree == 4:
                        imp_img = imp_img.filter(ImageFilter.EDGE_ENHANCE_MORE)                            
                    else:
                        aug = iaa.Sharpen(alpha=(0.0, 0.3), lightness=(0.6, 1.0))
                        imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                        imp_img = Image.fromarray(imp_img_np[0])
                elif random_choice == 2:
                    #blurring
                    random_degree = np.random.choice([1,2,3,4,5])
                    if random_degree == 1:
                        imp_img = imp_img.filter(ImageFilter.GaussianBlur())
                    elif random_degree == 2:
                        imp_img = imp_img.filter(ImageFilter.BLUR)
                    elif random_degree == 3:
                        imp_img = imp_img.filter(ImageFilter.SMOOTH)
                    elif random_degree == 4:
                        imp_img = imp_img.filter(ImageFilter.SMOOTH_MORE)
                    else:
                        aug = iaa.MotionBlur(k=3)
                        imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                        imp_img = Image.fromarray(imp_img_np[0])
                elif random_choice == 3:
                    # Basic compression and expansion
                    divider = random.uniform(1.0, 1.5)
                    cw, ch = imp_img.size
                    new_cw = int(cw/divider)
                    new_ch = int(ch/divider)
                    
                    first_choice = np.random.choice([1,2,3,4,5,6])
                    if first_choice == 1:
                        imp_img = imp_img.resize((new_cw, new_ch), Image.NEAREST)
                    elif first_choice == 2:
                        imp_img = imp_img.resize((new_cw, new_ch), Image.BILINEAR)
                    elif first_choice == 3:
                        imp_img = imp_img.resize((new_cw, new_ch), Image.BICUBIC)
                    elif first_choice == 4:
                        imp_img = imp_img.resize((new_cw, new_ch), Image.LANCZOS)
                    elif first_choice == 5:
                        imp_img = imp_img.resize((new_cw, new_ch), Image.HAMMING)
                    else:
                        imp_img = imp_img.resize((new_cw, new_ch), Image.BOX)
                    
                    second_choice = np.random.choice([1,2,3,4,5,6])
                    if second_choice == 1:
                        imp_img = imp_img.resize((cw, ch), Image.NEAREST)
                    elif second_choice == 2:
                        imp_img = imp_img.resize((cw, ch), Image.BILINEAR)
                    elif second_choice == 3:
                        imp_img = imp_img.resize((cw, ch), Image.BICUBIC)
                    elif second_choice == 4:
                        imp_img = imp_img.resize((cw, ch), Image.LANCZOS)
                    elif second_choice == 5:
                        imp_img = imp_img.resize((cw, ch), Image.HAMMING)
                    else:
                        imp_img = imp_img.resize((cw, ch), Image.BOX)
                elif random_choice == 4:
                    #JPEG compression
                    aug = iaa.JpegCompression(compression=(10, 50))
                    imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                    imp_img = Image.fromarray(imp_img_np[0])
                else:
                    if random.random() < 0.5:
                        aug = iaa.AdditiveGaussianNoise(scale=3)
                        imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                        imp_img = Image.fromarray(imp_img_np[0])
                    else:
                        aug = iaa.AdditiveLaplaceNoise(scale=3)
                        imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                        imp_img = Image.fromarray(imp_img_np[0])
                
                         
            if random.random() < 0.5: # random contrast change
                random_degree = np.random.choice([1,2,3,4,5,6,7,8])                
                if random_degree == 1:
                    aug = iaa.GammaContrast((0.5, 2.0))
                    imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                    imp_img = Image.fromarray(imp_img_np[0])
                elif random_degree == 2:
                    aug = iaa.LinearContrast((0.4, 1.6))
                    imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                    imp_img = Image.fromarray(imp_img_np[0])
                elif random_degree == 3:
                    aug = iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6))
                    imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                    imp_img = Image.fromarray(imp_img_np[0])
                elif random_degree == 4:
                    aug = iaa.LogContrast(gain=(0.6, 1.4))
                    imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                    imp_img = Image.fromarray(imp_img_np[0])            
                else:    
                    aug = iaa.pillike.EnhanceBrightness()
                    imp_img_np = aug(images = np.expand_dims(np.array(imp_img), axis=0))
                    imp_img = Image.fromarray(imp_img_np[0])
                    
        if self.normalize:
            inp_img_t = torch.from_numpy(np.float32(inp_img)).float()
            inp_img_mean = torch.mean(inp_img_t)
            inp_img_std = torch.std(inp_img_t)
            xform_inp_img = (inp_img_t - inp_img_mean)/inp_img_std
            xform_inp_img = xform_inp_img.unsqueeze(0)
            
            xform_inp_mask = self.transform(inp_mask)
            xform_inp_mask[xform_inp_mask<0] = -1
            xform_inp_mask[xform_inp_mask>=0] = 1
            
            tar_img_t = torch.from_numpy(np.float32(tar_img)).float()
            tar_img_mean = torch.mean(tar_img_t)
            tar_img_std = torch.std(tar_img_t)
            xform_tar_img = (tar_img_t - tar_img_mean)/tar_img_std
            xform_tar_img = xform_tar_img.unsqueeze(0)
            
            xform_tar_mask = self.transform(tar_mask)
            xform_tar_mask[xform_tar_mask<0] = -1.0
            xform_tar_mask[xform_tar_mask>=0] = 1.0
            
            tar2_img_t = torch.from_numpy(np.float32(tar2_img)).float()
            tar2_img_mean = torch.mean(tar2_img_t)
            tar2_img_std = torch.std(tar2_img_t)
            xform_tar2_img = (tar2_img_t - tar2_img_mean)/tar2_img_std
            xform_tar2_img = xform_tar2_img.unsqueeze(0)
            
            xform_tar2_mask = self.transform(tar2_mask)
            xform_tar2_mask[xform_tar2_mask<0] = -1.0
            xform_tar2_mask[xform_tar2_mask>=0] = 1.0
            
            imp_img_t = torch.from_numpy(np.float32(imp_img)).float()
            imp_img_mean = torch.mean(imp_img_t)
            imp_img_std = torch.std(imp_img_t)
            xform_imp_img = (imp_img_t - imp_img_mean)/imp_img_std
            xform_imp_img = xform_imp_img.unsqueeze(0)
            
            xform_imp_mask = self.transform(imp_mask)
            xform_imp_mask[xform_imp_mask<0] = -1.0
            xform_imp_mask[xform_imp_mask>=0] = 1.0
            return {'inp_img': xform_inp_img, 'inp_img_mean': inp_img_mean, 'inp_img_std':inp_img_std, 'tar_img' : xform_tar_img, 'tar_img_mean':tar_img_mean, 'tar_img_std':tar_img_std, 'tar2_img' : xform_tar2_img, 'tar2_img_mean':tar2_img_mean, 'tar2_img_std':tar2_img_std, 'imp_img' : xform_imp_img, 'imp_img_mean': imp_img_mean, 'imp_img_std':imp_img_std, 'inp_mask': xform_inp_mask, 'tar_mask' : xform_tar_mask, 'tar2_mask' : xform_tar2_mask, 'imp_mask' : xform_imp_mask, 'inp_img_pxyr' : inp_img_pupil_xyr, 'tar_img_pxyr' : tar_img_pupil_xyr, 'tar2_img_pxyr' : tar2_img_pupil_xyr, 'imp_img_pxyr' : imp_img_pupil_xyr, 'inp_img_ixyr': inp_img_iris_xyr, 'tar_img_ixyr' :  tar_img_iris_xyr, 'tar2_img_ixyr' :  tar2_img_iris_xyr, 'imp_img_ixyr' : imp_img_iris_xyr, 'identifier' : quadruplet[8], 'imposter_id' : quadruplet[9]}

        else:
            xform_inp_img = self.transform(inp_img)
            xform_inp_mask = self.transform(inp_mask)
            xform_inp_mask[xform_inp_mask<0] = -1
            xform_inp_mask[xform_inp_mask>=0] = 1
            
            xform_tar_img = self.transform(tar_img)
            xform_tar_mask = self.transform(tar_mask)
            xform_tar_mask[xform_tar_mask<0] = -1.0
            xform_tar_mask[xform_tar_mask>=0] = 1.0
            
            xform_tar2_img = self.transform(tar2_img)
            xform_tar2_mask = self.transform(tar2_mask)
            xform_tar2_mask[xform_tar2_mask<0] = -1.0
            xform_tar2_mask[xform_tar2_mask>=0] = 1.0
            
            xform_imp_img = self.transform(imp_img)
            xform_imp_mask = self.transform(imp_mask)
            xform_imp_mask[xform_imp_mask<0] = -1.0
            xform_imp_mask[xform_imp_mask>=0] = 1.0
            return {'inp_img': xform_inp_img, 'tar_img' : xform_tar_img, 'tar2_img' : xform_tar2_img, 'imp_img' : xform_imp_img, 'inp_mask': xform_inp_mask, 'tar_mask' : xform_tar_mask, 'tar2_mask' : xform_tar2_mask, 'imp_mask' : xform_imp_mask, 'inp_img_pxyr' : inp_img_pupil_xyr, 'tar_img_pxyr' : tar_img_pupil_xyr, 'tar2_img_pxyr' : tar2_img_pupil_xyr, 'imp_img_pxyr' : imp_img_pupil_xyr, 'inp_img_ixyr': inp_img_iris_xyr, 'tar_img_ixyr' :  tar_img_iris_xyr,  'tar2_img_ixyr' :  tar2_img_iris_xyr, 'imp_img_ixyr' : imp_img_iris_xyr, 'identifier' : quadruplet[8], 'imposter_id' : quadruplet[9]}
    
    def __len__(self):
        return len(self.quadruplets)
    
    def reset(self):
        print('Shuffling quadruplets for WSD...')
        self.quadruplets = []
        self.genuine_ids = []
        self.imposter_ids = []
        wsd_identifiers = list(self.bins_wsd.keys())
        random.shuffle(wsd_identifiers)
        for identifier in tqdm(wsd_identifiers):
            wsd_identifiers_removed = wsd_identifiers.copy()
            wsd_identifiers_removed.remove(identifier)
            bin_numbers1 = list(self.bins_wsd[identifier].keys())
            bin_numbers2 = list(self.bins_wsd[identifier].keys())
            bin_numbers1.sort()
            bin_numbers2.sort()
            for bin_num_1 in bin_numbers1:
                for bin_num_2 in bin_numbers2:
                    if bin_num_1 != bin_num_2:
                        random.shuffle(self.bins_wsd[identifier][bin_num_1])
                        random.shuffle(self.bins_wsd[identifier][bin_num_2])
                        max_pairs = min(len(self.bins_wsd[identifier][bin_num_1]), len(self.bins_wsd[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            imposter_ind_max = 0
                            while imposter_ind_max == 0:
                                imposter_id = random.choice(wsd_identifiers_removed)
                                imposter_ind_max = len(self.bins_wsd[imposter_id][bin_num_2])
                            imagename1 = self.bins_wsd[identifier][bin_num_1][img_ind]
                            imagename2 = self.bins_wsd[identifier][bin_num_2][img_ind]
                            imagename3 = self.bins_wsd[imposter_id][bin_num_2][random.randrange(0, imposter_ind_max)]
                            imagename4 = self.bins_wsd[identifier][bin_num_2][random.randrange(0, len(self.bins_wsd[identifier][bin_num_2]))]
                            indiv_id1 = imagename1.split('_')[0]
                            indiv_id2 = imagename2.split('_')[0]
                            indiv_id3 = imagename3.split('_')[0]
                            indiv_id4 = imagename4.split('_')[0]
                            imagepath1 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id1), imagename1)
                            maskpath1 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id1), imagename1)
                            imagepath2 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id2), imagename2)
                            maskpath2 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id2), imagename2)
                            imagepath3 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id3), imagename3)
                            maskpath3 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id3), imagename3)
                            imagepath4 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id4), imagename4)
                            maskpath4 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id4), imagename4)
                            self.quadruplets.append([imagepath1, maskpath1, imagepath2, maskpath2, imagepath3, maskpath3, imagepath4, maskpath4, identifier, imposter_id])             
        
        print('Shuffling quadruplets for BXGRID...')
        bxgrid_identifiers = list(self.bins_bxgrid.keys())
        random.shuffle(bxgrid_identifiers)
        for identifier in tqdm(bxgrid_identifiers):
            bxgrid_identifiers_removed = bxgrid_identifiers.copy()
            bxgrid_identifiers_removed.remove(identifier)
            bin_numbers1 = list(self.bins_bxgrid[identifier].keys())
            bin_numbers2 = list(self.bins_bxgrid[identifier].keys())
            bin_numbers1.sort()
            bin_numbers2.sort()
            for bin_num_1 in bin_numbers1:
                for bin_num_2 in bin_numbers2:
                    if bin_num_1 != bin_num_2:
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_1])
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_2])
                        max_pairs = min(len(self.bins_bxgrid[identifier][bin_num_1]), len(self.bins_bxgrid[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            imposter_ind_max = 0
                            while imposter_ind_max == 0:
                                imposter_id = random.choice(bxgrid_identifiers_removed)
                                imposter_ind_max = len(self.bins_bxgrid[imposter_id][bin_num_2])
                            imagename1 = self.bins_bxgrid[identifier][bin_num_1][img_ind]
                            imagename2 = self.bins_bxgrid[identifier][bin_num_2][img_ind]
                            imagename3 = self.bins_bxgrid[imposter_id][bin_num_2][random.randrange(0, imposter_ind_max)]
                            imagename4 = self.bins_bxgrid[identifier][bin_num_2][random.randrange(0, len(self.bins_bxgrid[identifier][bin_num_2]))]
                            imagepath1 = os.path.join(self.image_dir_bxgrid, imagename1)
                            maskpath1 = os.path.join(self.mask_dir_bxgrid, imagename1)
                            imagepath2 = os.path.join(self.image_dir_bxgrid, imagename2)
                            maskpath2 = os.path.join(self.mask_dir_bxgrid, imagename2)
                            imagepath3 = os.path.join(self.image_dir_bxgrid, imagename3)
                            maskpath3 = os.path.join(self.mask_dir_bxgrid, imagename3)
                            imagepath4 = os.path.join(self.image_dir_bxgrid, imagename4)
                            maskpath4 = os.path.join(self.mask_dir_bxgrid, imagename4)
                            self.quadruplets.append([imagepath1, maskpath1, imagepath2, maskpath2, imagepath3, maskpath3, imagepath4, maskpath4, identifier, imposter_id])
                            
        print('Total quadruplets: ', len(self.quadruplets))
    
class PairFromBinAllDatasetBoth(Dataset):
    def __init__(self, bins_path_wsd, parent_dir_wsd, bins_path_bxgrid, parent_dir_bxgrid, normalize=True, flip_data = True, res_mult=1):
        super().__init__()
        
        self.flip_data = flip_data   
        self.input_size=(int(240*res_mult),int(320*res_mult))
        self.res_mult = res_mult
        self.normalize = normalize
        
        self.image_dir_wsd = os.path.join(parent_dir_wsd, 'WarsawData2')
        self.mask_dir_wsd = os.path.join(parent_dir_wsd, 'WarsawMask2')            
        
        with open(bins_path_wsd, 'rb') as wsdbinsFile:
            self.bins_wsd = pkl.load(wsdbinsFile)
        
        self.image_dir_bxgrid = os.path.join(parent_dir_bxgrid, 'iris')
        self.mask_dir_bxgrid = os.path.join(parent_dir_bxgrid, 'mask')       
        
        with open(bins_path_bxgrid, 'rb') as bxgridbinsFile:
            self.bins_bxgrid = pkl.load(bxgridbinsFile)
        
        self.all_ids = list(self.bins_wsd.keys()) + list(self.bins_bxgrid.keys())
        
        print('Initializing pairs for Warsaw Pupil Dynamics...')
        self.pairs = []
        wsd_identifiers = list(self.bins_wsd.keys())
        random.shuffle(wsd_identifiers)
        for identifier in tqdm(wsd_identifiers):
            bin_numbers1 = list(self.bins_wsd[identifier].keys())
            bin_numbers1.sort()
            bin_numbers2 = list(self.bins_wsd[identifier].keys())
            bin_numbers2.sort()
            for bin_num_1 in bin_numbers1:
                for bin_num_2 in bin_numbers2:
                    if bin_num_1 != bin_num_2:
                        max_pairs = min(len(self.bins_wsd[identifier][bin_num_1]), len(self.bins_wsd[identifier][bin_num_2]))
                        random.shuffle(self.bins_wsd[identifier][bin_num_1])
                        random.shuffle(self.bins_wsd[identifier][bin_num_2])
                        for img_ind in range(max_pairs):
                            imagename1 = self.bins_wsd[identifier][bin_num_1][img_ind]
                            imagename2 = self.bins_wsd[identifier][bin_num_2][img_ind]
                            indiv_id1 = imagename1.split('_')[0]
                            indiv_id2 = imagename2.split('_')[0]
                            imagepath1 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id1), imagename1)
                            maskpath1 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id1), imagename1)
                            imagepath2 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id2), imagename2)
                            maskpath2 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id2), imagename2)
                            self.pairs.append([imagepath1, maskpath1, imagepath2, maskpath2, identifier])
                        random.shuffle(self.bins_wsd[identifier][bin_num_1])
                        random.shuffle(self.bins_wsd[identifier][bin_num_2])
                        for img_ind in range(max_pairs):
                            imagename1 = self.bins_wsd[identifier][bin_num_2][img_ind]
                            imagename2 = self.bins_wsd[identifier][bin_num_1][img_ind]
                            indiv_id1 = imagename1.split('_')[0]
                            indiv_id2 = imagename2.split('_')[0]
                            imagepath1 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id1), imagename1)
                            maskpath1 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id1), imagename1)
                            imagepath2 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id2), imagename2)
                            maskpath2 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id2), imagename2)
                            self.pairs.append([imagepath1, maskpath1, imagepath2, maskpath2, identifier])                   
        
        print('Initializing pairs for BXGRID...')
        bxgrid_identifiers = list(self.bins_bxgrid.keys())
        random.shuffle(bxgrid_identifiers)
        for identifier in tqdm(bxgrid_identifiers):
            bin_numbers1 = list(self.bins_bxgrid[identifier].keys())
            bin_numbers1.sort()
            bin_numbers2 = list(self.bins_bxgrid[identifier].keys())
            bin_numbers2.sort()
            for bin_num_1 in bin_numbers1:
                for bin_num_2 in bin_numbers2:
                    if bin_num_1 != bin_num_2:
                        max_pairs = min(len(self.bins_bxgrid[identifier][bin_num_1]), len(self.bins_bxgrid[identifier][bin_num_2]))
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_1])
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_2])
                        for img_ind in range(max_pairs):
                            imagename1 = self.bins_bxgrid[identifier][bin_num_1][img_ind]
                            imagename2 = self.bins_bxgrid[identifier][bin_num_2][img_ind]
                            imagepath1 = os.path.join(self.image_dir_bxgrid, imagename1)
                            maskpath1 = os.path.join(self.mask_dir_bxgrid, imagename1)
                            imagepath2 = os.path.join(self.image_dir_bxgrid, imagename2)
                            maskpath2 = os.path.join(self.mask_dir_bxgrid, imagename2)
                            self.pairs.append([imagepath1, maskpath1, imagepath2, maskpath2, identifier])
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_1])
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_2])
                        for img_ind in range(max_pairs):
                            imagename1 = self.bins_bxgrid[identifier][bin_num_2][img_ind]
                            imagename2 = self.bins_bxgrid[identifier][bin_num_1][img_ind]
                            imagepath1 = os.path.join(self.image_dir_bxgrid, imagename1)
                            maskpath1 = os.path.join(self.mask_dir_bxgrid, imagename1)
                            imagepath2 = os.path.join(self.image_dir_bxgrid, imagename2)
                            maskpath2 = os.path.join(self.mask_dir_bxgrid, imagename2)
                            self.pairs.append([imagepath1, maskpath1, imagepath2, maskpath2, identifier])
        
        random.shuffle(self.pairs)                 
        print('Total pairs: ', len(self.pairs))
        
        
        if self.normalize:
            self.transform = transforms.Compose([
                transforms.Resize([self.input_size[0], self.input_size[1]]),
                transforms.ToTensor(),
                transforms.Normalize((0.5, ), (0.5, ))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize([self.input_size[0], self.input_size[1]]),
                transforms.ToTensor()
            ])
        
        self.pupil_iris_xyrs_path_wsd = os.path.join(parent_dir_wsd, 'pupil_iris_xyrs2.pkl')
        with open(self.pupil_iris_xyrs_path_wsd, 'rb') as pupilirisxyrsFile:
            self.pupil_iris_xyrs_wsd = pkl.load(pupilirisxyrsFile)
        
        self.pupil_iris_xyrs_path_bxgrid = os.path.join(parent_dir_bxgrid, 'pupil_iris_xyrs.pkl')
        with open(self.pupil_iris_xyrs_path_bxgrid, 'rb') as pupilirisxyrsFile:
            self.pupil_iris_xyrs_bxgrid = pkl.load(pupilirisxyrsFile)
        
        self.pupil_xyrs = {}
        self.iris_xyrs = {}
        
        for imagename in self.pupil_iris_xyrs_wsd.keys():
            indiv_id = imagename.split('_')[0]
            imagepath = os.path.join(os.path.join(self.image_dir_wsd, indiv_id), imagename)
            self.pupil_xyrs[imagepath] = self.pupil_iris_xyrs_wsd[imagename]['pxyr']
            self.iris_xyrs[imagepath] = self.pupil_iris_xyrs_wsd[imagename]['ixyr']
        
        for imagename in self.pupil_iris_xyrs_bxgrid.keys():
            imagepath = os.path.join(self.image_dir_bxgrid, imagename)
            self.pupil_xyrs[imagepath] = self.pupil_iris_xyrs_bxgrid[imagename]['pxyr']
            self.iris_xyrs[imagepath] = self.pupil_iris_xyrs_bxgrid[imagename]['ixyr']
        
       
    
    def load_image(self, file):
        return Image.open(file).convert('L')
        
    def __getitem__(self, index):
    
        pair = self.pairs[index]
        
        inp_img_path = pair[0].strip()
        inp_mask_path = pair[1].strip()
        tar_img_path = pair[2].strip()
        tar_mask_path = pair[3].strip()
        
        inp_img = self.load_image(inp_img_path)
        inp_mask = self.load_image(inp_mask_path)
        
        tar_img = self.load_image(tar_img_path)
        tar_mask = self.load_image(tar_mask_path)
        
        inp_img_pupil_xyr = torch.FloatTensor(self.pupil_xyrs[inp_img_path])
        inp_img_iris_xyr = torch.FloatTensor(self.iris_xyrs[inp_img_path])
        
        tar_img_pupil_xyr = torch.FloatTensor(self.pupil_xyrs[tar_img_path])
        tar_img_iris_xyr = torch.FloatTensor(self.iris_xyrs[tar_img_path])
        
        sw, sh = inp_img.size
        sw_mult = (self.input_size[1]/sw)
        sh_mult = (self.input_size[0]/sh)
        
        bw, bh = tar_img.size
        bw_mult = (self.input_size[1]/bw)
        bh_mult = (self.input_size[0]/bh)
        
        inp_img_pupil_xyr[0] = (inp_img_pupil_xyr[0] * sw_mult)
        inp_img_pupil_xyr[1] = (inp_img_pupil_xyr[1] * sh_mult)
        inp_img_pupil_xyr[2] = (inp_img_pupil_xyr[2] * max(sw_mult, sh_mult))
        
        tar_img_pupil_xyr[0] = (tar_img_pupil_xyr[0] * bw_mult)
        tar_img_pupil_xyr[1] = (tar_img_pupil_xyr[1] * bh_mult)
        tar_img_pupil_xyr[2] = (tar_img_pupil_xyr[2] * max(bw_mult, bh_mult))
    
        inp_img_iris_xyr[0] = (inp_img_iris_xyr[0] * sw_mult)
        inp_img_iris_xyr[1] = (inp_img_iris_xyr[1] * sh_mult)
        inp_img_iris_xyr[2] = (inp_img_iris_xyr[2] * max(sw_mult, sh_mult))
        
        tar_img_iris_xyr[0] = (tar_img_iris_xyr[0] * bw_mult)
        tar_img_iris_xyr[1] = (tar_img_iris_xyr[1] * bh_mult)
        tar_img_iris_xyr[2] = (tar_img_iris_xyr[2] * max(bw_mult, bh_mult))
        
        if self.flip_data and random.random() < 0.5:
            inp_img = inp_img.transpose(Image.FLIP_LEFT_RIGHT) 
            inp_mask = inp_mask.transpose(Image.FLIP_LEFT_RIGHT)
            tar_img = tar_img.transpose(Image.FLIP_LEFT_RIGHT) 
            tar_mask = tar_mask.transpose(Image.FLIP_LEFT_RIGHT)
            inp_img_pupil_xyr[0] = self.input_size[1] - inp_img_pupil_xyr[0]
            tar_img_pupil_xyr[0] = self.input_size[1] - tar_img_pupil_xyr[0]
            inp_img_iris_xyr[0] = self.input_size[1] - inp_img_iris_xyr[0]
            tar_img_iris_xyr[0] = self.input_size[1] - tar_img_iris_xyr[0]
        
        xform_inp_img = self.transform(inp_img)
        xform_inp_mask = self.transform(inp_mask)
        if self.normalize:
            xform_inp_mask[xform_inp_mask<0] = -1
        else:
            xform_inp_mask[xform_inp_mask<0] = 0
        xform_inp_mask[xform_inp_mask>=0] = 1
        
        xform_tar_img = self.transform(tar_img)
        xform_tar_mask = self.transform(tar_mask)
        if self.normalize:
            xform_tar_mask[xform_tar_mask<0] = -1
        else:
            xform_tar_mask[xform_tar_mask<0] = 0
        xform_tar_mask[xform_tar_mask>=0] = 1
        
        return {'inp_img': xform_inp_img, 'tar_img' : xform_tar_img, 'inp_mask': xform_inp_mask, 'tar_mask' : xform_tar_mask, 'inp_img_pxyr' : inp_img_pupil_xyr, 'tar_img_pxyr' : tar_img_pupil_xyr, 'inp_img_ixyr': inp_img_iris_xyr, 'tar_img_ixyr' :  tar_img_iris_xyr, 'identifier':pair[4].strip() }
    
    
    def __len__(self):
        return len(self.pairs)
    
    def reset(self):
        print('Shuffling pairs for Warsaw Pupil Dynamics...')
        self.pairs = []
        wsd_identifiers = list(self.bins_wsd.keys())
        random.shuffle(wsd_identifiers)
        for identifier in tqdm(wsd_identifiers):
            bin_numbers1 = list(self.bins_wsd[identifier].keys())
            bin_numbers1.sort()
            bin_numbers2 = list(self.bins_wsd[identifier].keys())
            bin_numbers2.sort()
            for bin_num_1 in bin_numbers1:
                for bin_num_2 in bin_numbers2:
                    if bin_num_1 != bin_num_2:
                        max_pairs = min(len(self.bins_wsd[identifier][bin_num_1]), len(self.bins_wsd[identifier][bin_num_2]))
                        random.shuffle(self.bins_wsd[identifier][bin_num_1])
                        random.shuffle(self.bins_wsd[identifier][bin_num_2])
                        for img_ind in range(max_pairs):
                            imagename1 = self.bins_wsd[identifier][bin_num_1][img_ind]
                            imagename2 = self.bins_wsd[identifier][bin_num_2][img_ind]
                            indiv_id1 = imagename1.split('_')[0]
                            indiv_id2 = imagename2.split('_')[0]
                            imagepath1 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id1), imagename1)
                            maskpath1 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id1), imagename1)
                            imagepath2 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id2), imagename2)
                            maskpath2 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id2), imagename2)
                            self.pairs.append([imagepath1, maskpath1, imagepath2, maskpath2, identifier])
                        random.shuffle(self.bins_wsd[identifier][bin_num_1])
                        random.shuffle(self.bins_wsd[identifier][bin_num_2])
                        for img_ind in range(max_pairs):
                            imagename1 = self.bins_wsd[identifier][bin_num_2][img_ind]
                            imagename2 = self.bins_wsd[identifier][bin_num_1][img_ind]
                            indiv_id1 = imagename1.split('_')[0]
                            indiv_id2 = imagename2.split('_')[0]
                            imagepath1 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id1), imagename1)
                            maskpath1 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id1), imagename1)
                            imagepath2 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id2), imagename2)
                            maskpath2 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id2), imagename2)
                            self.pairs.append([imagepath1, maskpath1, imagepath2, maskpath2, identifier])                   
        
        print('Shuffling pairs for BXGRID...')
        bxgrid_identifiers = list(self.bins_bxgrid.keys())
        random.shuffle(bxgrid_identifiers)
        for identifier in tqdm(bxgrid_identifiers):
            bin_numbers1 = list(self.bins_bxgrid[identifier].keys())
            bin_numbers1.sort()
            bin_numbers2 = list(self.bins_bxgrid[identifier].keys())
            bin_numbers2.sort()
            for bin_num_1 in bin_numbers1:
                for bin_num_2 in bin_numbers2:
                    if bin_num_1 != bin_num_2:
                        max_pairs = min(len(self.bins_bxgrid[identifier][bin_num_1]), len(self.bins_bxgrid[identifier][bin_num_2]))
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_1])
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_2])
                        for img_ind in range(max_pairs):
                            imagename1 = self.bins_bxgrid[identifier][bin_num_1][img_ind]
                            imagename2 = self.bins_bxgrid[identifier][bin_num_2][img_ind]
                            imagepath1 = os.path.join(self.image_dir_bxgrid, imagename1)
                            maskpath1 = os.path.join(self.mask_dir_bxgrid, imagename1)
                            imagepath2 = os.path.join(self.image_dir_bxgrid, imagename2)
                            maskpath2 = os.path.join(self.mask_dir_bxgrid, imagename2)
                            self.pairs.append([imagepath1, maskpath1, imagepath2, maskpath2, identifier])
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_1])
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_2])
                        for img_ind in range(max_pairs):
                            imagename1 = self.bins_bxgrid[identifier][bin_num_2][img_ind]
                            imagename2 = self.bins_bxgrid[identifier][bin_num_1][img_ind]
                            imagepath1 = os.path.join(self.image_dir_bxgrid, imagename1)
                            maskpath1 = os.path.join(self.mask_dir_bxgrid, imagename1)
                            imagepath2 = os.path.join(self.image_dir_bxgrid, imagename2)
                            maskpath2 = os.path.join(self.mask_dir_bxgrid, imagename2)
                            self.pairs.append([imagepath1, maskpath1, imagepath2, maskpath2, identifier])
        
        random.shuffle(self.pairs)                 
        print('Total pairs: ', len(self.pairs))
        
class PairFromBinAllDatasetBothCropped(Dataset):
    def __init__(self, bins_path_wsd, parent_dir_wsd, bins_path_bxgrid, parent_dir_bxgrid, iris_crop = True, normalize = True, brightness_and_contrast = False, flip_data = True, res_mult = 1):
        super().__init__()
        self.iris_crop = iris_crop
        self.flip_data = flip_data
        self.b_n_c = brightness_and_contrast
        self.input_size=(int(256*res_mult),int(256*res_mult))
        self.input_size_nocrop = (int(480*res_mult), int(640*res_mult))
        self.res_mult = res_mult
        self.normalize = normalize
        
        self.image_dir_wsd = os.path.join(parent_dir_wsd, 'WarsawData2')
        self.mask_dir_wsd = os.path.join(parent_dir_wsd, 'WarsawMasksCoarse')        
        
        with open(bins_path_wsd, 'rb') as wsdbinsFile:
            self.bins_wsd = pkl.load(wsdbinsFile)
        
        self.image_dir_bxgrid = os.path.join(parent_dir_bxgrid, 'iris')
        self.mask_dir_bxgrid = os.path.join(parent_dir_bxgrid, 'masks_coarse')       
        
        with open(bins_path_bxgrid, 'rb') as bxgridbinsFile:
            self.bins_bxgrid = pkl.load(bxgridbinsFile)
        
        self.all_ids = list(self.bins_wsd.keys()) + list(self.bins_bxgrid.keys())
        
        print('Initializing pairs for Warsaw Pupil Dynamics...')
        self.pairs = []
        self.genuine_ids = []
        self.imposter_ids = []
        wsd_identifiers = list(self.bins_wsd.keys())
        random.shuffle(wsd_identifiers)
        for identifier in tqdm(wsd_identifiers):
            wsd_identifiers_removed = wsd_identifiers.copy()
            wsd_identifiers_removed.remove(identifier)
            bin_numbers1 = list(self.bins_wsd[identifier].keys())
            bin_numbers2 = list(self.bins_wsd[identifier].keys())
            bin_numbers1.sort()
            bin_numbers2.sort()
            for bin_num_1 in bin_numbers1:
                for bin_num_2 in bin_numbers2:
                    if bin_num_1 != bin_num_2:
                        random.shuffle(self.bins_wsd[identifier][bin_num_1])
                        random.shuffle(self.bins_wsd[identifier][bin_num_2])
                        max_pairs = min(len(self.bins_wsd[identifier][bin_num_1]), len(self.bins_wsd[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            imagename1 = self.bins_wsd[identifier][bin_num_1][img_ind]
                            imagename2 = self.bins_wsd[identifier][bin_num_2][img_ind]
                            indiv_id1 = imagename1.split('_')[0]
                            indiv_id2 = imagename2.split('_')[0]
                            imagepath1 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id1), imagename1)
                            maskpath1 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id1), imagename1)
                            imagepath2 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id2), imagename2)
                            maskpath2 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id2), imagename2)
                            self.pairs.append([imagepath1, maskpath1, imagepath2, maskpath2, identifier])        
        
        print('Initializing pairs for BXGRID...')
        bxgrid_identifiers = list(self.bins_bxgrid.keys())
        random.shuffle(bxgrid_identifiers)
        for identifier in tqdm(bxgrid_identifiers):
            bxgrid_identifiers_removed = bxgrid_identifiers.copy()
            bxgrid_identifiers_removed.remove(identifier)
            bin_numbers1 = list(self.bins_bxgrid[identifier].keys())
            bin_numbers2 = list(self.bins_bxgrid[identifier].keys())
            bin_numbers1.sort()
            bin_numbers2.sort()
            for bin_num_1 in bin_numbers1:
                for bin_num_2 in bin_numbers2:
                    if bin_num_1 != bin_num_2:
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_1])
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_2])
                        max_pairs = min(len(self.bins_bxgrid[identifier][bin_num_1]), len(self.bins_bxgrid[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            imagename1 = self.bins_bxgrid[identifier][bin_num_1][img_ind]
                            imagename2 = self.bins_bxgrid[identifier][bin_num_2][img_ind]
                            imagepath1 = os.path.join(self.image_dir_bxgrid, imagename1)
                            maskpath1 = os.path.join(self.mask_dir_bxgrid, imagename1)
                            imagepath2 = os.path.join(self.image_dir_bxgrid, imagename2)
                            maskpath2 = os.path.join(self.mask_dir_bxgrid, imagename2)
                            self.pairs.append([imagepath1, maskpath1, imagepath2, maskpath2, identifier])
                            
        print('Total pairs: ', len(self.pairs))
        

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5, ))
        ])
        
        self.pupil_iris_xyrs_path_wsd = os.path.join(parent_dir_wsd, 'pupil_iris_xyrs_new.pkl')
        with open(self.pupil_iris_xyrs_path_wsd, 'rb') as pupilirisxyrsFile:
            self.pupil_iris_xyrs_wsd = pkl.load(pupilirisxyrsFile)
        
        self.pupil_iris_xyrs_path_bxgrid = os.path.join(parent_dir_bxgrid, 'pupil_iris_xyrs_new.pkl')
        with open(self.pupil_iris_xyrs_path_bxgrid, 'rb') as pupilirisxyrsFile:
            self.pupil_iris_xyrs_bxgrid = pkl.load(pupilirisxyrsFile)
        
        self.pupil_xyrs = {}
        self.iris_xyrs = {}
        
        for imagename in self.pupil_iris_xyrs_wsd.keys():
            imagepath = os.path.join(self.image_dir_wsd, imagename + '.bmp')
            self.pupil_xyrs[imagepath] = self.pupil_iris_xyrs_wsd[imagename]['pxyr']
            self.iris_xyrs[imagepath] = self.pupil_iris_xyrs_wsd[imagename]['ixyr']
        
        for imagename in self.pupil_iris_xyrs_bxgrid.keys():
            imagepath = os.path.join(self.image_dir_bxgrid, imagename + '.png')
            self.pupil_xyrs[imagepath] = self.pupil_iris_xyrs_bxgrid[imagename]['pxyr']
            self.iris_xyrs[imagepath] = self.pupil_iris_xyrs_bxgrid[imagename]['ixyr']
        
        self.heavy_augment_prob = 0.3
    
    def load_image(self, file):
        return Image.open(file).convert('L')
        
    def __getitem__(self, index):
    
        pair = self.pairs[index]
        
        inp_img_path = pair[0].strip()
        inp_mask_path = pair[1].strip()
        tar_img_path = pair[2].strip()
        tar_mask_path = pair[3].strip()
        
        inp_img = self.load_image(inp_img_path)
        inp_mask = self.load_image(inp_mask_path)
        
        tar_img = self.load_image(tar_img_path)
        tar_mask = self.load_image(tar_mask_path)
        
        inp_img_pupil_xyr = torch.FloatTensor(self.pupil_xyrs[inp_img_path])
        inp_img_iris_xyr = torch.FloatTensor(self.iris_xyrs[inp_img_path])
        
        tar_img_pupil_xyr = torch.FloatTensor(self.pupil_xyrs[tar_img_path])
        tar_img_iris_xyr = torch.FloatTensor(self.iris_xyrs[tar_img_path])
        
        sw, sh = inp_img.size
        bw, bh = tar_img.size
        
        if self.flip_data and random.random() < 0.5:
            inp_img = inp_img.transpose(Image.FLIP_LEFT_RIGHT) 
            inp_mask = inp_mask.transpose(Image.FLIP_LEFT_RIGHT)
            tar_img = tar_img.transpose(Image.FLIP_LEFT_RIGHT) 
            tar_mask = tar_mask.transpose(Image.FLIP_LEFT_RIGHT)
            inp_img_pupil_xyr[0] = sw - inp_img_pupil_xyr[0]
            tar_img_pupil_xyr[0] = bw - tar_img_pupil_xyr[0]
            inp_img_iris_xyr[0] = sw - inp_img_iris_xyr[0]
            tar_img_iris_xyr[0] = bw - tar_img_iris_xyr[0]
        
        if self.iris_crop:
            inp_img = inp_img.crop((inp_img_iris_xyr[0].item() - inp_img_iris_xyr[2].item(), inp_img_iris_xyr[1].item() - inp_img_iris_xyr[2].item(), inp_img_iris_xyr[0].item() + inp_img_iris_xyr[2].item(), inp_img_iris_xyr[1].item() + inp_img_iris_xyr[2].item()))
            inp_mask = inp_mask.crop((inp_img_iris_xyr[0].item() - inp_img_iris_xyr[2].item(), inp_img_iris_xyr[1].item() - inp_img_iris_xyr[2].item(), inp_img_iris_xyr[0].item() + inp_img_iris_xyr[2].item(), inp_img_iris_xyr[1].item() + inp_img_iris_xyr[2].item()))
            sw_crop, sh_crop = inp_img.size
            inp_img_pupil_xyr[0] = inp_img_pupil_xyr[0] - inp_img_iris_xyr[0] + sw_crop/2
            inp_img_pupil_xyr[1] = inp_img_pupil_xyr[1] - inp_img_iris_xyr[1] + sh_crop/2
            inp_img_iris_xyr[0] = sw_crop/2
            inp_img_iris_xyr[1] = sh_crop/2            
            
            tar_img = tar_img.crop((tar_img_iris_xyr[0].item() - tar_img_iris_xyr[2].item(), tar_img_iris_xyr[1].item() - tar_img_iris_xyr[2].item(), tar_img_iris_xyr[0].item() + tar_img_iris_xyr[2].item(), tar_img_iris_xyr[1].item() + tar_img_iris_xyr[2].item()))
            tar_mask = tar_mask.crop((tar_img_iris_xyr[0].item() - tar_img_iris_xyr[2].item(), tar_img_iris_xyr[1].item() - tar_img_iris_xyr[2].item(), tar_img_iris_xyr[0].item() + tar_img_iris_xyr[2].item(), tar_img_iris_xyr[1].item() + tar_img_iris_xyr[2].item()))
            bw_crop, bh_crop = tar_img.size
            tar_img_pupil_xyr[0] = tar_img_pupil_xyr[0] - tar_img_iris_xyr[0] + bw_crop/2
            tar_img_pupil_xyr[1] = tar_img_pupil_xyr[1] - tar_img_iris_xyr[1] + bh_crop/2
            tar_img_iris_xyr[0] = bw_crop/2
            tar_img_iris_xyr[1] = bh_crop/2
            
            sh_mult = self.input_size[0]/sh_crop
            sw_mult = self.input_size[1]/sw_crop
            inp_img = inp_img.resize((self.input_size[1], self.input_size[0]))
            inp_mask = inp_mask.resize((self.input_size[1], self.input_size[0]))
            inp_img_pupil_xyr[0] = self.input_size[1]/2 - sw_mult * (sw_crop/2 - inp_img_pupil_xyr[0])
            inp_img_pupil_xyr[1] = self.input_size[0]/2 - sh_mult * (sh_crop/2 - inp_img_pupil_xyr[1])
            inp_img_pupil_xyr[2] = max(sw_mult, sh_mult) * inp_img_pupil_xyr[2]
            inp_img_iris_xyr[0] = self.input_size[1]/2
            inp_img_iris_xyr[1] = self.input_size[0]/2
            inp_img_iris_xyr[2] = max(sw_mult, sh_mult) * inp_img_iris_xyr[2]
            
            bh_mult = self.input_size[0]/bh_crop
            bw_mult = self.input_size[1]/bw_crop
            tar_img = tar_img.resize((self.input_size[1], self.input_size[0]))
            tar_mask = tar_mask.resize((self.input_size[1], self.input_size[0]))
            tar_img_pupil_xyr[0] = self.input_size[1]/2 - bw_mult * (bw_crop/2 - tar_img_pupil_xyr[0])
            tar_img_pupil_xyr[1] = self.input_size[0]/2 - bh_mult * (bh_crop/2 - tar_img_pupil_xyr[1])
            tar_img_pupil_xyr[2] = max(bw_mult, bh_mult) * tar_img_pupil_xyr[2]
            tar_img_iris_xyr[0] = self.input_size[1]/2
            tar_img_iris_xyr[1] = self.input_size[0]/2
            tar_img_iris_xyr[2] = max(bw_mult, bh_mult) * tar_img_iris_xyr[2]
            
        else:
            sw, sh = inp_img.size
            sh_mult = self.input_size_nocrop[0]/sh
            sw_mult = self.input_size_nocrop[1]/sw
            
            inp_img = inp_img.resize((self.input_size_nocrop[1], self.input_size_nocrop[0]))
            inp_mask = inp_mask.resize((self.input_size_nocrop[1], self.input_size_nocrop[0]))
            inp_img_pupil_xyr[0] = self.input_size_nocrop[1]/2 - sw_mult * (sw/2 - inp_img_pupil_xyr[0])
            inp_img_pupil_xyr[1] = self.input_size_nocrop[0]/2 - sh_mult * (sh/2 - inp_img_pupil_xyr[1])
            inp_img_pupil_xyr[2] = max(sw_mult, sh_mult) * inp_img_pupil_xyr[2]
            inp_img_iris_xyr[0] = self.input_size_nocrop[1]/2 - sw_mult * (sw/2 - inp_img_iris_xyr[0])
            inp_img_iris_xyr[1] = self.input_size_nocrop[0]/2 - sh_mult * (sh/2 - inp_img_iris_xyr[1])
            inp_img_iris_xyr[2] = max(sw_mult, sh_mult) * inp_img_iris_xyr[2]
            
            bw, bh = tar_img.size
            bh_mult = self.input_size_nocrop[0]/bh
            bw_mult = self.input_size_nocrop[1]/bw
            
            tar_img = tar_img.resize((self.input_size_nocrop[1], self.input_size_nocrop[0]))
            tar_mask = tar_mask.resize((self.input_size_nocrop[1], self.input_size_nocrop[0]))
            tar_img_pupil_xyr[0] = self.input_size_nocrop[1]/2 - bw_mult * (bw/2 - tar_img_pupil_xyr[0])
            tar_img_pupil_xyr[1] = self.input_size_nocrop[0]/2 - bh_mult * (bh/2 - tar_img_pupil_xyr[1])
            tar_img_pupil_xyr[2] = max(bw_mult, bh_mult) * tar_img_pupil_xyr[2]
            tar_img_iris_xyr[0] = self.input_size_nocrop[1]/2 - bw_mult * (bw/2 - tar_img_iris_xyr[0])
            tar_img_iris_xyr[1] = self.input_size_nocrop[0]/2 - bh_mult * (bh/2 - tar_img_iris_xyr[1])
            tar_img_iris_xyr[2] = max(bw_mult, bh_mult) * tar_img_iris_xyr[2]

        if random.random() < self.heavy_augment_prob:
            if random.random() < 0.5:
                random_choice = np.random.choice([1,2,3,4,5])
                if random_choice == 1:   
                    # sharpening
                    random_degree = np.random.choice([1,2,3,4,5])
                    if random_degree == 1:
                        inp_img = inp_img.filter(ImageFilter.DETAIL)
                        tar_img = tar_img.filter(ImageFilter.DETAIL)
                    elif random_degree == 2:
                        inp_img = inp_img.filter(ImageFilter.SHARPEN)
                        tar_img = tar_img.filter(ImageFilter.SHARPEN)
                    elif random_degree == 3:
                        inp_img = inp_img.filter(ImageFilter.EDGE_ENHANCE)
                        tar_img = tar_img.filter(ImageFilter.EDGE_ENHANCE)
                    elif random_degree == 4:
                        inp_img = inp_img.filter(ImageFilter.EDGE_ENHANCE_MORE)
                        tar_img = tar_img.filter(ImageFilter.EDGE_ENHANCE_MORE)                          
                    else:
                        random_alpha = random.uniform(0.0, 0.3)
                        random_lightness = random.uniform(0.6, 1.0)
                        aug = iaa.Sharpen(alpha=random_alpha, lightness=random_lightness)
                        inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                        inp_img = Image.fromarray(inp_img_np[0])
                        tar_img_np = aug(images = np.expand_dims(np.array(tar_img), axis=0))
                        tar_img = Image.fromarray(tar_img_np[0])
                elif random_choice == 2:
                    #blurring
                    random_degree = np.random.choice([1,2,3,4,5])
                    if random_degree == 1:
                        inp_img = inp_img.filter(ImageFilter.GaussianBlur())
                        tar_img = tar_img.filter(ImageFilter.GaussianBlur())
                    elif random_degree == 2:
                        inp_img = inp_img.filter(ImageFilter.BLUR)
                        tar_img = tar_img.filter(ImageFilter.BLUR)
                    elif random_degree == 3:
                        inp_img = inp_img.filter(ImageFilter.SMOOTH)
                        tar_img = tar_img.filter(ImageFilter.SMOOTH)
                    elif random_degree == 4:
                        inp_img = inp_img.filter(ImageFilter.SMOOTH_MORE)
                        tar_img = tar_img.filter(ImageFilter.SMOOTH_MORE)
                    else:
                        random_degree = random.randint(-179, 180)
                        aug = iaa.MotionBlur(k=3, angle=random_degree)
                        inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                        inp_img = Image.fromarray(inp_img_np[0])
                        tar_img_np = aug(images = np.expand_dims(np.array(tar_img), axis=0))
                        tar_img = Image.fromarray(tar_img_np[0])
                elif random_choice == 3:
                    # Basic compression and expansion
                    divider = random.uniform(1.0, 1.5)
                    cw, ch = inp_img.size
                    new_cw = int(cw/divider)
                    new_ch = int(ch/divider)
                    
                    first_choice = np.random.choice([1,2,3,4,5,6])
                    if first_choice == 1:
                        inp_img = inp_img.resize((new_cw, new_ch), Image.NEAREST)
                        tar_img = tar_img.resize((new_cw, new_ch), Image.NEAREST)
                    elif first_choice == 2:
                        inp_img = inp_img.resize((new_cw, new_ch), Image.BILINEAR)
                        tar_img = tar_img.resize((new_cw, new_ch), Image.BILINEAR)
                    elif first_choice == 3:
                        inp_img = inp_img.resize((new_cw, new_ch), Image.BICUBIC)
                        tar_img = tar_img.resize((new_cw, new_ch), Image.BICUBIC)
                    elif first_choice == 4:
                        inp_img = inp_img.resize((new_cw, new_ch), Image.LANCZOS)
                        tar_img = tar_img.resize((new_cw, new_ch), Image.LANCZOS)
                    elif first_choice == 5:
                        inp_img = inp_img.resize((new_cw, new_ch), Image.HAMMING)
                        tar_img = tar_img.resize((new_cw, new_ch), Image.HAMMING)
                    else:
                        inp_img = inp_img.resize((new_cw, new_ch), Image.BOX)
                        tar_img = tar_img.resize((new_cw, new_ch), Image.BOX)
                    
                    second_choice = np.random.choice([1,2,3,4,5,6])
                    if second_choice == 1:
                        inp_img = inp_img.resize((cw, ch), Image.NEAREST)
                        tar_img = tar_img.resize((cw, ch), Image.NEAREST)
                    elif second_choice == 2:
                        inp_img = inp_img.resize((cw, ch), Image.BILINEAR)
                        tar_img = tar_img.resize((cw, ch), Image.BILINEAR)
                    elif second_choice == 3:
                        inp_img = inp_img.resize((cw, ch), Image.BICUBIC)
                        tar_img = tar_img.resize((cw, ch), Image.BICUBIC)
                    elif second_choice == 4:
                        inp_img = inp_img.resize((cw, ch), Image.LANCZOS)
                        tar_img = tar_img.resize((cw, ch), Image.LANCZOS)
                    elif second_choice == 5:
                        inp_img = inp_img.resize((cw, ch), Image.HAMMING)
                        tar_img = tar_img.resize((cw, ch), Image.HAMMING)
                    else:
                        inp_img = inp_img.resize((cw, ch), Image.BOX)
                        tar_img = tar_img.resize((cw, ch), Image.BOX)
                elif random_choice == 4:
                    #JPEG compression
                    random_compression = random.randint(10, 50)
                    aug = iaa.JpegCompression(compression=random_compression)
                    inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                    inp_img = Image.fromarray(inp_img_np[0])
                    tar_img_np = aug(images = np.expand_dims(np.array(tar_img), axis=0))
                    tar_img = Image.fromarray(tar_img_np[0])
                else:
                    if random.random() < 0.5:
                        aug = iaa.AdditiveGaussianNoise(scale=3)
                        inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                        inp_img = Image.fromarray(inp_img_np[0])
                        tar_img_np = aug(images = np.expand_dims(np.array(tar_img), axis=0))
                        tar_img = Image.fromarray(tar_img_np[0])
                    else:
                        aug = iaa.AdditiveLaplaceNoise(scale=3)
                        inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                        inp_img = Image.fromarray(inp_img_np[0])
                        tar_img_np = aug(images = np.expand_dims(np.array(tar_img), axis=0))
                        tar_img = Image.fromarray(tar_img_np[0])
                       
            if random.random() < 0.5: # random contrast change
                random_degree = np.random.choice([1,2,3,4,5,6,7,8])                
                if random_degree == 1:
                    aug = iaa.GammaContrast((0.5, 2.0))
                elif random_degree == 2:
                    aug = iaa.LinearContrast((0.4, 1.6))
                elif random_degree == 3:
                    aug = iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6))
                elif random_degree == 4:
                    aug = iaa.LogContrast(gain=(0.6, 1.4)) 
                else:    
                    aug = iaa.pillike.EnhanceBrightness()
                inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                inp_img = Image.fromarray(inp_img_np[0])
                random_degree = np.random.choice([1,2,3,4,5,6,7,8])                
                if random_degree == 1:
                    aug = iaa.GammaContrast((0.5, 2.0))
                elif random_degree == 2:
                    aug = iaa.LinearContrast((0.4, 1.6))
                elif random_degree == 3:
                    aug = iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6))
                elif random_degree == 4:
                    aug = iaa.LogContrast(gain=(0.6, 1.4)) 
                else:    
                    aug = iaa.pillike.EnhanceBrightness()
                tar_img_np = aug(images = np.expand_dims(np.array(tar_img), axis=0))
                tar_img = Image.fromarray(tar_img_np[0])

            if random.random() < 0.5: # random contrast change
                random_degree = np.random.choice([1,2,3,4,5,6,7,8])                
                if random_degree == 1:
                    aug = iaa.GammaContrast((0.5, 2.0))
                elif random_degree == 2:
                    aug = iaa.LinearContrast((0.4, 1.6))
                elif random_degree == 3:
                    aug = iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6))
                elif random_degree == 4:
                    aug = iaa.LogContrast(gain=(0.6, 1.4)) 
                else:    
                    aug = iaa.pillike.EnhanceBrightness()
                inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                inp_img = Image.fromarray(inp_img_np[0])
                random_degree = np.random.choice([1,2,3,4,5,6,7,8])                
                if random_degree == 1:
                    aug = iaa.GammaContrast((0.5, 2.0))
                elif random_degree == 2:
                    aug = iaa.LinearContrast((0.4, 1.6))
                elif random_degree == 3:
                    aug = iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6))
                elif random_degree == 4:
                    aug = iaa.LogContrast(gain=(0.6, 1.4)) 
                else:    
                    aug = iaa.pillike.EnhanceBrightness()
                inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
                inp_img = Image.fromarray(inp_img_np[0])
               
        if self.normalize:
            inp_img_t = torch.from_numpy(np.float32(inp_img)).float()
            inp_img_mean = torch.mean(inp_img_t)
            inp_img_std = torch.std(inp_img_t)
            xform_inp_img = torch.clamp(torch.nan_to_num((inp_img_t - inp_img_mean)/inp_img_std, nan=4, posinf=4, neginf=-4), -4, 4)
            xform_inp_img = xform_inp_img.unsqueeze(0)
            
            xform_inp_mask = self.transform(inp_mask)
            xform_inp_mask[xform_inp_mask<0] = -1.0
            xform_inp_mask[xform_inp_mask>=0] = 1.0
            
            tar_img_t = torch.from_numpy(np.float32(tar_img)).float()
            tar_img_mean = torch.mean(tar_img_t)
            tar_img_std = torch.std(tar_img_t)
            xform_tar_img = torch.clamp(torch.nan_to_num((tar_img_t - tar_img_mean)/tar_img_std, nan=4, posinf=4, neginf=-4), -4, 4)
            xform_tar_img = xform_tar_img.unsqueeze(0)
            
            xform_tar_mask = self.transform(tar_mask)
            xform_tar_mask[xform_tar_mask<0] = -1.0
            xform_tar_mask[xform_tar_mask>=0] = 1.0
            
            return {'inp_img': xform_inp_img, 'inp_img_mean': inp_img_mean, 'inp_img_std':inp_img_std, 'tar_img' : xform_tar_img, 'tar_img_mean':tar_img_mean, 'tar_img_std':tar_img_std, 'tar2_img' : xform_tar2_img, 'tar2_img_mean':tar2_img_mean, 'tar2_img_std':tar2_img_std, 'imp_img' : xform_imp_img, 'imp_img_mean': imp_img_mean, 'imp_img_std':imp_img_std, 'inp_mask': xform_inp_mask, 'tar_mask' : xform_tar_mask, 'tar2_mask' : xform_tar2_mask, 'imp_mask' : xform_imp_mask, 'inp_img_pxyr' : inp_img_pupil_xyr, 'tar_img_pxyr' : tar_img_pupil_xyr, 'tar2_img_pxyr' : tar2_img_pupil_xyr, 'imp_img_pxyr' : imp_img_pupil_xyr, 'inp_img_ixyr': inp_img_iris_xyr, 'tar_img_ixyr' :  tar_img_iris_xyr, 'tar2_img_ixyr' :  tar2_img_iris_xyr, 'imp_img_ixyr' : imp_img_iris_xyr, 'identifier' : quadruplet[8], 'imposter_id' : quadruplet[9]}

        else:
            xform_inp_img = self.transform(inp_img)
            xform_inp_mask = self.transform(inp_mask)
            xform_inp_mask[xform_inp_mask<0] = -1.0
            xform_inp_mask[xform_inp_mask>=0] = 1.0
            
            xform_tar_img = self.transform(tar_img)
            xform_tar_mask = self.transform(tar_mask)
            xform_tar_mask[xform_tar_mask<0] = -1.0
            xform_tar_mask[xform_tar_mask>=0] = 1.0

            return {'inp_img': xform_inp_img, 'tar_img' : xform_tar_img, 'tar2_img' : xform_tar2_img, 'imp_img' : xform_imp_img, 'inp_mask': xform_inp_mask, 'tar_mask' : xform_tar_mask, 'tar2_mask' : xform_tar2_mask, 'imp_mask' : xform_imp_mask, 'inp_img_pxyr' : inp_img_pupil_xyr, 'tar_img_pxyr' : tar_img_pupil_xyr, 'tar2_img_pxyr' : tar2_img_pupil_xyr, 'imp_img_pxyr' : imp_img_pupil_xyr, 'inp_img_ixyr': inp_img_iris_xyr, 'tar_img_ixyr' :  tar_img_iris_xyr,  'tar2_img_ixyr' :  tar2_img_iris_xyr, 'imp_img_ixyr' : imp_img_iris_xyr, 'identifier' : quadruplet[8], 'imposter_id' : quadruplet[9]}
    
    def __len__(self):
        return len(self.pairs)
    
    def reset(self):
        print('Shuffling pairs for WSD...')
        self.pairs = []
        self.genuine_ids = []
        self.imposter_ids = []
        wsd_identifiers = list(self.bins_wsd.keys())
        random.shuffle(wsd_identifiers)
        for identifier in tqdm(wsd_identifiers):
            wsd_identifiers_removed = wsd_identifiers.copy()
            wsd_identifiers_removed.remove(identifier)
            bin_numbers1 = list(self.bins_wsd[identifier].keys())
            bin_numbers2 = list(self.bins_wsd[identifier].keys())
            bin_numbers1.sort()
            bin_numbers2.sort()
            for bin_num_1 in bin_numbers1:
                for bin_num_2 in bin_numbers2:
                    if bin_num_1 != bin_num_2:
                        random.shuffle(self.bins_wsd[identifier][bin_num_1])
                        random.shuffle(self.bins_wsd[identifier][bin_num_2])
                        max_pairs = min(len(self.bins_wsd[identifier][bin_num_1]), len(self.bins_wsd[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            imagename1 = self.bins_wsd[identifier][bin_num_1][img_ind]
                            imagename2 = self.bins_wsd[identifier][bin_num_2][img_ind]
                            indiv_id1 = imagename1.split('_')[0]
                            indiv_id2 = imagename2.split('_')[0]
                            imagepath1 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id1), imagename1)
                            maskpath1 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id1), imagename1)
                            imagepath2 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id2), imagename2)
                            maskpath2 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id2), imagename2)
                            self.pairs.append([imagepath1, maskpath1, imagepath2, maskpath2, identifier])             
        
        print('Shuffling pairs for BXGRID...')
        bxgrid_identifiers = list(self.bins_bxgrid.keys())
        random.shuffle(bxgrid_identifiers)
        for identifier in tqdm(bxgrid_identifiers):
            bxgrid_identifiers_removed = bxgrid_identifiers.copy()
            bxgrid_identifiers_removed.remove(identifier)
            bin_numbers1 = list(self.bins_bxgrid[identifier].keys())
            bin_numbers2 = list(self.bins_bxgrid[identifier].keys())
            bin_numbers1.sort()
            bin_numbers2.sort()
            for bin_num_1 in bin_numbers1:
                for bin_num_2 in bin_numbers2:
                    if bin_num_1 != bin_num_2:
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_1])
                        random.shuffle(self.bins_bxgrid[identifier][bin_num_2])
                        max_pairs = min(len(self.bins_bxgrid[identifier][bin_num_1]), len(self.bins_bxgrid[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            imagename1 = self.bins_bxgrid[identifier][bin_num_1][img_ind]
                            imagename2 = self.bins_bxgrid[identifier][bin_num_2][img_ind]
                            imagepath1 = os.path.join(self.image_dir_bxgrid, imagename1)
                            maskpath1 = os.path.join(self.mask_dir_bxgrid, imagename1)
                            imagepath2 = os.path.join(self.image_dir_bxgrid, imagename2)
                            maskpath2 = os.path.join(self.mask_dir_bxgrid, imagename2)
                            self.pairs.append([imagepath1, maskpath1, imagepath2, maskpath2, identifier])
                            
        print('Total pairs: ', len(self.pairs))
        
class PairFromBinDatasetSB(Dataset):
    def __init__(self, bins_path_wsd, parent_dir_wsd, flip_data = True, res_mult=1):
        super().__init__()
        
        self.flip_data = flip_data   
        self.input_size=(int(240*res_mult),int(320*res_mult))
        self.res_mult = res_mult
        
        self.image_dir_wsd = os.path.join(parent_dir_wsd, 'WarsawData')
        self.mask_dir_wsd = os.path.join(parent_dir_wsd, 'WarsawMask')            
        
        with open(bins_path_wsd, 'rb') as wsdbinsFile:
            self.bins_wsd = pkl.load(wsdbinsFile)
        
        print('Initializing pairs for Warsaw Pupil Dynamics...')
        self.pairs = []
        self.pair_ids = []
        self.all_ids = list(self.bins_wsd.keys())
        
        for identifier in tqdm(self.bins_wsd.keys()):
            for bin_num_1 in self.bins_wsd[identifier].keys():
                for bin_num_2 in self.bins_wsd[identifier].keys():
                    if bin_num_1 < bin_num_2:
                        random.shuffle(self.bins_wsd[identifier][bin_num_1])
                        random.shuffle(self.bins_wsd[identifier][bin_num_2])
                        max_pairs = min(len(self.bins_wsd[identifier][bin_num_1]), len(self.bins_wsd[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            imagename1 = self.bins_wsd[identifier][bin_num_1][img_ind]
                            imagename2 = self.bins_wsd[identifier][bin_num_2][img_ind]
                            indiv_id1 = imagename1.split('_')[0]
                            indiv_id2 = imagename2.split('_')[0]
                            imagepath1 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id1), imagename1)
                            maskpath1 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id1), imagename1)
                            imagepath2 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id2), imagename2)
                            maskpath2 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id2), imagename2)
                            self.pairs.append([imagepath1, maskpath1, imagepath2, maskpath2])  
                            self.pair_ids.append(identifier)
        
        print('Total pairs: ', len(self.pairs))
        
        self.transform = transforms.Compose([
            transforms.Resize([self.input_size[0], self.input_size[1]]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ])
        
        self.pupil_iris_xyrs_path_wsd = os.path.join(parent_dir_wsd, 'pupil_iris_xyrs.pkl')
        with open(self.pupil_iris_xyrs_path_wsd, 'rb') as pupilirisxyrsFile:
            self.pupil_iris_xyrs_wsd = pkl.load(pupilirisxyrsFile)
        
        
        self.pupil_xyrs = {}
        self.iris_xyrs = {}
        
        for imagename in self.pupil_iris_xyrs_wsd.keys():
            indiv_id = imagename.split('_')[0]
            imagepath = os.path.join(os.path.join(self.image_dir_wsd, indiv_id), imagename)
            self.pupil_xyrs[imagepath] = self.pupil_iris_xyrs_wsd[imagename]['pxyr']
            self.iris_xyrs[imagepath] = self.pupil_iris_xyrs_wsd[imagename]['ixyr']
        
       
    
    def load_image(self, file):
        return Image.open(file).convert('L')
        
    def __getitem__(self, index):
    
        pair = self.pairs[index]
        
        inp_img_path = pair[0].strip()
        inp_mask_path = pair[1].strip()
        tar_img_path = pair[2].strip()
        tar_mask_path = pair[3].strip()
        
        inp_img = self.load_image(inp_img_path)
        inp_mask = self.load_image(inp_mask_path)
        
        tar_img = self.load_image(tar_img_path)
        tar_mask = self.load_image(tar_mask_path)
        
        inp_img_pupil_xyr = torch.FloatTensor(self.pupil_xyrs[inp_img_path])
        inp_img_iris_xyr = torch.FloatTensor(self.iris_xyrs[inp_img_path])
        
        tar_img_pupil_xyr = torch.FloatTensor(self.pupil_xyrs[tar_img_path])
        tar_img_iris_xyr = torch.FloatTensor(self.iris_xyrs[tar_img_path])
        
        sw, sh = inp_img.size
        sw_mult = (320/sw)
        sh_mult = (240/sh)
        
        bw, bh = tar_img.size
        bw_mult = (320/sw)
        bh_mult = (240/sh)
        
        inp_img_pupil_xyr[0] = (inp_img_pupil_xyr[0] * sw_mult)
        inp_img_pupil_xyr[1] = (inp_img_pupil_xyr[1] * sh_mult)
        inp_img_pupil_xyr[2] = (inp_img_pupil_xyr[2] * max(sw_mult, sh_mult))
        
        tar_img_pupil_xyr[0] = (tar_img_pupil_xyr[0] * bw_mult)
        tar_img_pupil_xyr[1] = (tar_img_pupil_xyr[1] * bh_mult)
        tar_img_pupil_xyr[2] = (tar_img_pupil_xyr[2] * max(bw_mult, bh_mult))
    
        inp_img_iris_xyr[0] = (inp_img_iris_xyr[0] * sw_mult)
        inp_img_iris_xyr[1] = (inp_img_iris_xyr[1] * sh_mult)
        inp_img_iris_xyr[2] = (inp_img_iris_xyr[2] * max(sw_mult, sh_mult))
        
        tar_img_iris_xyr[0] = (tar_img_iris_xyr[0] * bw_mult)
        tar_img_iris_xyr[1] = (tar_img_iris_xyr[1] * bh_mult)
        tar_img_iris_xyr[2] = (tar_img_iris_xyr[2] * max(bw_mult, bh_mult))
        
        if self.flip_data and random.random() < 0.5:
            inp_img = inp_img.transpose(Image.FLIP_LEFT_RIGHT) 
            inp_mask = inp_mask.transpose(Image.FLIP_LEFT_RIGHT)
            tar_img = tar_img.transpose(Image.FLIP_LEFT_RIGHT) 
            tar_mask = tar_mask.transpose(Image.FLIP_LEFT_RIGHT)
            inp_img_pupil_xyr[0] = self.input_size[1] - inp_img_pupil_xyr[0]
            tar_img_pupil_xyr[0] = self.input_size[1] - tar_img_pupil_xyr[0]
            inp_img_iris_xyr[0] = self.input_size[1] - inp_img_iris_xyr[0]
            tar_img_iris_xyr[0] = self.input_size[1] - tar_img_iris_xyr[0]
            #d1 = ImageDraw.Draw(inp_img)
            #d1.text((0, 0), "flipped", fill=0)
        
        xform_inp_img = self.transform(inp_img)
        xform_inp_mask = self.transform(inp_mask)
        xform_inp_mask[xform_inp_mask<0] = -1
        xform_inp_mask[xform_inp_mask>=0] = 1
        
        xform_tar_img = self.transform(tar_img)
        xform_tar_mask = self.transform(tar_mask)
        xform_tar_mask[xform_tar_mask<0] = -1
        xform_tar_mask[xform_tar_mask>=0] = 1
        
        return {'inp_img': xform_inp_img, 'inp_mask' : xform_inp_mask, 'tar_img' : xform_tar_img, 'tar_mask' : xform_tar_mask, 'inp_img_pxyr': inp_img_pupil_xyr, 'inp_img_ixyr' : inp_img_iris_xyr, 'tar_img_pxyr' : tar_img_pupil_xyr, 'tar_img_ixyr' :  tar_img_iris_xyr, 'identifier':self.pair_ids[index] }
    
    def __len__(self):
        return len(self.pairs)
    
    def reset(self):
        print('Shuffling pairs for Warsaw Pupil Dynamics...')
        self.pairs = []
        self.pair_ids = []
        for identifier in tqdm(self.bins_wsd.keys()):
            for bin_num_1 in self.bins_wsd[identifier].keys():
                for bin_num_2 in self.bins_wsd[identifier].keys():
                    if bin_num_1 < bin_num_2:
                        random.shuffle(self.bins_wsd[identifier][bin_num_1])
                        random.shuffle(self.bins_wsd[identifier][bin_num_2])
                        max_pairs = min(len(self.bins_wsd[identifier][bin_num_1]), len(self.bins_wsd[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            imagename1 = self.bins_wsd[identifier][bin_num_1][img_ind]
                            imagename2 = self.bins_wsd[identifier][bin_num_2][img_ind]
                            indiv_id1 = imagename1.split('_')[0]
                            indiv_id2 = imagename2.split('_')[0]
                            imagepath1 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id1), imagename1)
                            maskpath1 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id1), imagename1)
                            imagepath2 = os.path.join(os.path.join(self.image_dir_wsd, indiv_id2), imagename2)
                            maskpath2 = os.path.join(os.path.join(self.mask_dir_wsd, indiv_id2), imagename2)
                            self.pairs.append([imagepath1, maskpath1, imagepath2, maskpath2])  
                            self.pair_ids.append(identifier)                        
                            
        print('Total pairs: ', len(self.pairs))      
                                   
class PairFromBinDatasetBS(Dataset):
    def __init__(self, bins_path, parent_dir, flip_data = True, res_mult=1):
        super().__init__()
        
        self.image_dir = os.path.join(parent_dir, 'WarsawData')
        self.mask_dir = os.path.join(parent_dir, 'WarsawMask')
        self.pupil_xyrs_path = os.path.join(parent_dir, 'WarsawPupilCircles.pkl')
        self.iris_xyrs_path = os.path.join(parent_dir, 'WarsawIrisCircles.pkl')
        self.flip_data = flip_data 
        self.input_size=(int(240*res_mult),int(320*res_mult))
        self.res_mult = res_mult
        
        with open(bins_path, 'rb') as binsFile:
            self.bins = pkl.load(binsFile)
        
        print('Initializing pairs...')
        self.pairs = []
        self.pair_ids = []
        for identifier in tqdm(self.bins.keys()):
            for bin_num_1 in self.bins[identifier].keys():
                for bin_num_2 in self.bins[identifier].keys():
                    if bin_num_1 > bin_num_2:
                        random.shuffle(self.bins[identifier][bin_num_1])
                        random.shuffle(self.bins[identifier][bin_num_2])
                        max_pairs = min(len(self.bins[identifier][bin_num_1]), len(self.bins[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            self.pairs.append([self.bins[identifier][bin_num_1][img_ind], self.bins[identifier][bin_num_2][img_ind]]) 
                            self.pair_ids.append('_'.join(identifier.split('_')[:-1]))           
        
        self.transform = transforms.Compose([
            transforms.Resize([self.input_size[0], self.input_size[1]]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ])
        
        with open(self.pupil_xyrs_path, 'rb') as pupilFile:
            self.pupil_xyrs = pkl.load(pupilFile)
        
        with open(self.iris_xyrs_path, 'rb') as irisFile:
            self.iris_xyrs = pkl.load(irisFile)
    
    def load_image(self, file):
        return Image.open(file).convert('L')
        
    def __getitem__(self, index):
    
        pair = self.pairs[index]
        
        inp_img_name = pair[0].strip()
        tar_img_name = pair[1].strip()
        
        indiv_id1 = inp_img_name.split('_')[0]
        indiv_id2 = tar_img_name.split('_')[0]
        
        inp_img_path = os.path.join(os.path.join(self.image_dir, indiv_id1), inp_img_name)
        inp_mask_path = os.path.join(os.path.join(self.mask_dir, indiv_id1), inp_img_name)
        
        tar_img_path = os.path.join(os.path.join(self.image_dir, indiv_id2), tar_img_name)
        tar_mask_path = os.path.join(os.path.join(self.mask_dir, indiv_id2), tar_img_name)
        
        inp_img = self.load_image(inp_img_path)
        inp_mask = self.load_image(inp_mask_path)
        
        tar_img = self.load_image(tar_img_path)
        tar_mask = self.load_image(tar_mask_path)
        
        inp_img_pupil_xyr = torch.tensor(self.pupil_xyrs[inp_img_name]) * self.res_mult
        inp_img_iris_xyr = torch.tensor(self.iris_xyrs[inp_img_name]) * self.res_mult
        
        tar_img_pupil_xyr = torch.tensor(self.pupil_xyrs[tar_img_name]) * self.res_mult
        tar_img_iris_xyr = torch.tensor(self.iris_xyrs[tar_img_name]) * self.res_mult
        
        if self.flip_data and random.random() < 0.5:
            inp_img = inp_img.transpose(Image.FLIP_LEFT_RIGHT) 
            inp_mask = inp_mask.transpose(Image.FLIP_LEFT_RIGHT)
            tar_img = tar_img.transpose(Image.FLIP_LEFT_RIGHT) 
            tar_mask = tar_mask.transpose(Image.FLIP_LEFT_RIGHT)
            inp_img_pupil_xyr[0] = self.input_size[1] - inp_img_pupil_xyr[0]
            tar_img_pupil_xyr[0] = self.input_size[1] - tar_img_pupil_xyr[0]
            inp_img_iris_xyr[0] = self.input_size[1] - inp_img_iris_xyr[0]
            tar_img_iris_xyr[0] = self.input_size[1] - tar_img_iris_xyr[0]
        
        xform_inp_img = self.transform(inp_img)
        xform_inp_mask = self.transform(inp_mask)
        xform_inp_mask[xform_inp_mask<0] = -1
        xform_inp_mask[xform_inp_mask>=0] = 1
        
        xform_tar_img = self.transform(tar_img)
        xform_tar_mask = self.transform(tar_mask)
        xform_tar_mask[xform_tar_mask<0] = -1
        xform_tar_mask[xform_tar_mask>=0] = 1
        
        
        return {'inp_img': xform_inp_img, 'inp_mask' : xform_inp_mask, 'tar_img' : xform_tar_img, 'tar_mask' : xform_tar_mask, 'inp_img_pxyr': inp_img_pupil_xyr, 'inp_img_ixyr' : inp_img_iris_xyr, 'tar_img_pxyr' : tar_img_pupil_xyr, 'tar_img_ixyr' :  tar_img_iris_xyr, 'identifier': self.pair_ids[index] }
    
    def __len__(self):
        return len(self.pairs)
    
    def reset(self):
        print('Resetting Pairs....')
        self.pairs = []
        self.pair_ids = []
        for identifier in tqdm(self.bins.keys()):
            for bin_num_1 in self.bins[identifier].keys():
                for bin_num_2 in self.bins[identifier].keys():
                    if bin_num_1 > bin_num_2:
                        random.shuffle(self.bins[identifier][bin_num_1])
                        random.shuffle(self.bins[identifier][bin_num_2])
                        max_pairs = min(len(self.bins[identifier][bin_num_1]), len(self.bins[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            self.pairs.append([self.bins[identifier][bin_num_1][img_ind], self.bins[identifier][bin_num_2][img_ind]]) 
                            self.pair_ids.append('_'.join(identifier.split('_')[:-1])) 

class PairFromBinDatasetPolar(Dataset):
    def __init__(self, bins_path, parent_dir, flip_data = True, input_size=(240,320), polar_width=512, polar_height=64):
        super().__init__()
        
        self.image_dir = os.path.join(parent_dir, 'WarsawData')
        self.mask_dir = os.path.join(parent_dir, 'WarsawMask')
        self.pupil_xyrs_path = os.path.join(parent_dir, 'WarsawPupilCircles.pkl')
        self.iris_xyrs_path = os.path.join(parent_dir, 'WarsawIrisCircles.pkl')
        self.flip_data = flip_data   
        self.input_size = input_size
        self.polar_width = polar_width
        self.polar_height = polar_height
        
        with open(bins_path, 'rb') as binsFile:
            self.bins = pkl.load(binsFile)
        
        print('Initializing pairs...')
        self.pairs = []
        for identifier in tqdm(self.bins.keys()):
            for bin_num_1 in self.bins[identifier].keys():
                for bin_num_2 in self.bins[identifier].keys():
                    if bin_num_1 < bin_num_2:
                        random.shuffle(self.bins[identifier][bin_num_1])
                        random.shuffle(self.bins[identifier][bin_num_2])
                        max_pairs = min(len(self.bins[identifier][bin_num_1]), len(self.bins[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            self.pairs.append([self.bins[identifier][bin_num_1][img_ind], self.bins[identifier][bin_num_2][img_ind]])            
        
        self.transform = transforms.Compose([
            transforms.Resize([self.input_size[0], self.input_size[1]]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ])
        
        with open(self.pupil_xyrs_path, 'rb') as pupilFile:
            self.pupil_xyrs = pkl.load(pupilFile)
        
        with open(self.iris_xyrs_path, 'rb') as irisFile:
            self.iris_xyrs = pkl.load(irisFile)
            
    def grid_sample(self, input, grid, interp_mode='bilinear'):

        # grid: [-1, 1]
        N, C, H, W = input.shape
        gridx = grid[:, :, :, 0]
        gridy = grid[:, :, :, 1]
        gridx = ((gridx + 1) / 2 * W - 0.5) / (W - 1) * 2 - 1
        gridy = ((gridy + 1) / 2 * H - 0.5) / (H - 1) * 2 - 1
        newgrid = torch.stack([gridx, gridy], dim=-1)
        return torch.nn.functional.grid_sample(input, newgrid, mode=interp_mode)
        
    # Rubbersheet model-based Cartesian-to-polar transformation using bilinear interpolation from torch grid sample
    def cartToPol(self, image, mask, pupil_xyr, iris_xyr):
        with torch.no_grad():
            if pupil_xyr is None or iris_xyr is None:
                return None, None
            
            image = image.unsqueeze(0)
            mask = mask.unsqueeze(0)
            width = image.shape[3]
            height = image.shape[2]

            polar_height = self.polar_height
            polar_width = self.polar_width

            pupil_xyr = torch.tensor(pupil_xyr).unsqueeze(0).float()
            iris_xyr = torch.tensor(iris_xyr).unsqueeze(0).float()
            
            theta = (2*pi*torch.linspace(1,polar_width,polar_width)/polar_width)
            pxCirclePoints = pupil_xyr[:, 0].reshape(-1, 1) + pupil_xyr[:, 2].reshape(-1, 1) @ torch.cos(theta).reshape(1, polar_width) #b x 512
            pyCirclePoints = pupil_xyr[:, 1].reshape(-1, 1) + pupil_xyr[:, 2].reshape(-1, 1) @ torch.sin(theta).reshape(1, polar_width)  #b x 512
            
            ixCirclePoints = iris_xyr[:, 0].reshape(-1, 1) + iris_xyr[:, 2].reshape(-1, 1) @ torch.cos(theta).reshape(1, polar_width)  #b x 512
            iyCirclePoints = iris_xyr[:, 1].reshape(-1, 1) + iris_xyr[:, 2].reshape(-1, 1) @ torch.sin(theta).reshape(1, polar_width)  #b x 512

            radius = (torch.linspace(0,polar_height,polar_height)/polar_height).reshape(-1, 1)  #64 x 1
            
            pxCoords = torch.matmul((1-radius), pxCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
            pyCoords = torch.matmul((1-radius), pyCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
            
            ixCoords = torch.matmul(radius, ixCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
            iyCoords = torch.matmul(radius, iyCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512

            x = torch.clamp(pxCoords + ixCoords, 0, width-1).float()
            x_norm = (x/(width-1))*2 - 1 #b x 64 x 512

            y = torch.clamp(pyCoords + iyCoords, 0, height-1).float()
            y_norm = (y/(height-1))*2 - 1  #b x 64 x 512

            grid_sample_mat = torch.cat([x_norm.unsqueeze(-1), y_norm.unsqueeze(-1)], dim=-1)

            image_polar = self.grid_sample(image, grid_sample_mat, interp_mode='bilinear')
            mask_polar = self.grid_sample(mask, grid_sample_mat, interp_mode='nearest')

            return image_polar[0], mask_polar[0]
    
    def load_image(self, file):
        return Image.open(file).convert('L')
        
    def __getitem__(self, index):
    
        pair = self.pairs[index]
        
        inp_img_name = pair[0].strip()
        tar_img_name = pair[1].strip()
        
        indiv_id1 = inp_img_name.split('_')[0]
        indiv_id2 = tar_img_name.split('_')[0]
        
        inp_img_path = os.path.join(os.path.join(self.image_dir, indiv_id1), inp_img_name)
        inp_mask_path = os.path.join(os.path.join(self.mask_dir, indiv_id1), inp_img_name)
        
        tar_img_path = os.path.join(os.path.join(self.image_dir, indiv_id2), tar_img_name)
        tar_mask_path = os.path.join(os.path.join(self.mask_dir, indiv_id2), tar_img_name)
        
        inp_img = self.load_image(inp_img_path)
        inp_mask = self.load_image(inp_mask_path)
        
        tar_img = self.load_image(tar_img_path)
        tar_mask = self.load_image(tar_mask_path)
        
        inp_img_pupil_xyr = torch.tensor(self.pupil_xyrs[inp_img_name])
        inp_img_iris_xyr = torch.tensor(self.iris_xyrs[inp_img_name])
        
        tar_img_pupil_xyr = torch.tensor(self.pupil_xyrs[tar_img_name])
        tar_img_iris_xyr = torch.tensor(self.iris_xyrs[tar_img_name])
        
        if self.flip_data and random.random() < 0.5:
            inp_img = inp_img.transpose(Image.FLIP_LEFT_RIGHT) 
            inp_mask = inp_mask.transpose(Image.FLIP_LEFT_RIGHT)
            tar_img = tar_img.transpose(Image.FLIP_LEFT_RIGHT) 
            tar_mask = tar_mask.transpose(Image.FLIP_LEFT_RIGHT)
            inp_img_pupil_xyr[0] = self.input_size[1] - inp_img_pupil_xyr[0]
            tar_img_pupil_xyr[0] = self.input_size[1] - tar_img_pupil_xyr[0]
            inp_img_iris_xyr[0] = self.input_size[1] - inp_img_iris_xyr[0]
            tar_img_iris_xyr[0] = self.input_size[1] - tar_img_iris_xyr[0]
        
        xform_inp_img = self.transform(inp_img)
        xform_inp_mask = self.transform(inp_mask)
        xform_inp_mask[xform_inp_mask<0] = -1
        xform_inp_mask[xform_inp_mask>=0] = 1
        
        xform_inp_img_polar, xform_inp_mask_polar = self.cartToPol(xform_inp_img, xform_inp_mask, inp_img_pupil_xyr, inp_img_iris_xyr)
        
        xform_tar_img = self.transform(tar_img)
        xform_tar_mask = self.transform(tar_mask)
        xform_tar_mask[xform_tar_mask<0] = -1
        xform_tar_mask[xform_tar_mask>=0] = 1
        
        xform_tar_img_polar, xform_tar_mask_polar = self.cartToPol(xform_tar_img, xform_tar_mask, tar_img_pupil_xyr, tar_img_iris_xyr)        
        
        return {'inp_img': xform_inp_img_polar, 'inp_mask' : xform_inp_mask_polar, 'tar_img' : xform_tar_img_polar, 'tar_mask' : xform_tar_mask_polar, 'inp_img_pxyr': inp_img_pupil_xyr, 'inp_img_ixyr' : inp_img_iris_xyr, 'tar_img_pxyr' : tar_img_pupil_xyr, 'tar_img_ixyr' :  tar_img_iris_xyr }
    
    def __len__(self):
        return len(self.pairs)
    
    def reset(self):
        print('Resetting Pairs....')
        self.pairs = []
        for identifier in tqdm(self.bins.keys()):
            for bin_num_1 in self.bins[identifier].keys():
                for bin_num_2 in self.bins[identifier].keys():
                    if bin_num_1 < bin_num_2:
                        random.shuffle(self.bins[identifier][bin_num_1])
                        random.shuffle(self.bins[identifier][bin_num_2])
                        max_pairs = min(len(self.bins[identifier][bin_num_1]), len(self.bins[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            self.pairs.append([self.bins[identifier][bin_num_1][img_ind], self.bins[identifier][bin_num_2][img_ind]])