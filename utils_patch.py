import torch
import torch.nn as nn
import torchvision.transforms as TF
from torchvision.utils import save_image
import cv2
import numpy as np
import random
import kornia.geometry.transform as KT
import matplotlib.pyplot as plt
import random


class PatchTransformer(nn.Module):
    def __init__(self):
        super(PatchTransformer, self).__init__()
        self.noise_factor = 0.1
        self.temp_factor = 0.1
        self.ratio_h = 0.2
    
    
    def forward(self, patch, targets, imgs):
        patch = patch.unsqueeze(0)
        patch = patch.expand(-1, 3, -1, -1)
        patch_mask = torch.ones_like(patch).cuda()
        image_size = imgs.size()         # height, width

        patch_tmp = torch.zeros_like(imgs).cuda()
        patch_mask_tmp = torch.zeros_like(imgs).cuda()
        
        # -------------------------------------------
        for i in range(targets.size(0)):
            img_idx = targets[i][0]
            bbox_w = targets[i][-2] * image_size[-1]
            bbox_h = targets[i][-1] * image_size[-2]
            
            # resize
            patch_size = int(bbox_h * self.ratio_h)
            #print(patch_size)
            patch_resize = KT.resize(patch, (patch_size, patch_size), align_corners=True)
            patch_mask_resize = KT.resize(patch_mask, (patch_size, patch_size), align_corners=True)
            
            # temperature
            random_seed = round(random.uniform(-1, 1), 2)
            temp_tensor = torch.ones_like(patch_resize) * self.temp_factor * random_seed
            patch_with_temp = patch_resize + temp_tensor
            patch_with_temp.data.clamp_(0, 1)

            # rotation
            angle = random.uniform(-10, 10)
            patch_rotation = TF.functional.rotate(patch_with_temp, angle, expand=True)
            patch_mask_rotation = TF.functional.rotate(patch_mask_resize, angle, expand=True)
            
            patch_size_h = patch_rotation.size()[-1]
            patch_size_w = patch_rotation.size()[-2]
            
            # padding
            x_center = int(targets[i][2] * image_size[-1])
            y_center = int(targets[i][3] * image_size[-2])
            
            padding_h = image_size[-2] - patch_size_h
            padding_w = image_size[-1] - patch_size_w
            
            padding_left = x_center - int(0.5 * patch_size_w)
            padding_right = padding_w - padding_left
            
            padding_top = y_center - int(0.7 * patch_size_h)
            padding_bottom = padding_h - padding_top

            padding = nn.ZeroPad2d((int(padding_left), int(padding_right), int(padding_top), int(padding_bottom)))
            patch_padding = padding(patch_rotation)
            patch_mask_padding = padding(patch_mask_rotation)
            
            patch_tmp[int(img_idx.item())] += patch_padding.squeeze()
            patch_mask_tmp[int(img_idx.item())] += patch_mask_padding.squeeze()        
        
        patch_tf = patch_tmp
        patch_mask_tf = patch_mask_tmp

        return patch_tf, patch_mask_tf
        
        
        
class PatchApplier(nn.Module):
    def __init__(self):
        super(PatchApplier, self).__init__()

    def forward(self, img_batch, patch, patch_mask_tf):
        patch_mask = patch_mask_tf - 1
        patch_mask = - patch_mask

        img_batch = torch.mul(img_batch, patch_mask) + patch

        imgWithPatch = img_batch
        return imgWithPatch

