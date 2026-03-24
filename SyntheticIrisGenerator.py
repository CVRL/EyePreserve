import math
import torch
from torch.nn.functional import grid_sample
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchvision import models
from torchvision.transforms import Compose, ToTensor, Normalize
from network import NestedSharedAtrousResUNet, NestedSharedAtrousAttentionResUNetIN
from PIL import Image
from torchvision import models, transforms
from math import pi
import numpy as np
import random
import dnnlib
import legacy
from typing import List, Optional, Tuple, Union
#ninja, legacy

class SyntheticIrisGenerator(object):
    def __init__(self, net_path='./models/network-snapshot-025000.pkl', device=torch.device('cpu')):
        self.net_path = net_path
        self.device = device
        with dnnlib.util.open_url(self.net_path) as f:
            self.G = legacy.load_network_pkl(f)['G_ema'].to(device)

    def generate_image(self):
        with torch.inference_mode():
            z = torch.randn(1, self.G.z_dim).to(self.device)
            img = self.G(z, None, truncation_psi=0.5, noise_mode='random')
            img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            return Image.fromarray(img[0][0].cpu().numpy(), 'L')
        
        

