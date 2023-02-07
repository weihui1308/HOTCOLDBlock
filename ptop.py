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


class ParticleToPatch():
    def __init__(self, patch_size):
        self.ratio_h = patch_size
        self.noise_factor = 0.1
        self.temp_factor = 0.1


    def __call__(self, x, targets, imgs):
        bs, c, h, w = imgs.size()
        
        patch_size_h = 90
        patch_size_w = 90
        patch = torch.ones((3, patch_size_w, patch_size_h), dtype=torch.float32).cuda() * 0.3
        patch_mask = torch.ones_like(patch)
        
        for m in range(3):
            for n in range(3):
                if int(x[0][m][n].item()) == 0:
                    patch_mask[:, m*30:m*30+30, n*30:n*30+30] = 0
        
        patch_tmp = torch.zeros_like(imgs).cuda()
        patch_mask_tmp = torch.zeros_like(imgs).cuda()
        
        
        for i in range(targets.size(0)):
            img_idx = targets[i][0]
            bbox_w = targets[i][-2] * w
            bbox_h = targets[i][-1] * h
            bbox_x = targets[i][-4] * w
            bbox_y = targets[i][-3] * h
            bbox_tl_x = bbox_x - bbox_w / 2
            bbox_tl_y = bbox_y - bbox_h / 2
            
            patch_vertex_1_x = bbox_tl_x + x[1][0][0] * bbox_w
            patch_vertex_1_y = bbox_tl_y + x[1][0][1] * bbox_h

            # second point
            patch_vertex_2_x = bbox_tl_x + x[1][1][0] * bbox_w
            patch_vertex_2_y = bbox_tl_y + x[1][1][1] * bbox_h

            # third point
            patch_vertex_3_x = bbox_tl_x + x[1][2][0] * bbox_w
            patch_vertex_3_y = bbox_tl_y + x[1][2][1] * bbox_h

            # fourth point
            patch_vertex_4_x = bbox_tl_x + x[1][3][0] * bbox_w
            patch_vertex_4_y = bbox_tl_y + x[1][3][1] * bbox_h
            
            # resize
            patch_size = int(bbox_h * self.ratio_h)
            if patch_size == 0:
                patch_size = 1
            patch_resize = KT.resize(patch, (patch_size, patch_size), align_corners=True)
            patch_mask_resize = KT.resize(patch_mask, (patch_size, patch_size), align_corners=True)
            
            # temperature
            random_seed = round(random.uniform(-1, 1), 2)
            temp_tensor = torch.ones_like(patch_resize) * self.temp_factor * random_seed
            patch_with_temp = patch_resize + temp_tensor
            patch_with_temp.data.clamp_(0, 1)
            
            # padding
            x1_center = int(patch_vertex_1_x + patch_size/2)
            y1_center = int(patch_vertex_1_y + patch_size/2)
            
            x2_center = int(patch_vertex_2_x + patch_size/2)
            y2_center = int(patch_vertex_2_y + patch_size/2)
            
            x3_center = int(patch_vertex_3_x + patch_size/2)
            y3_center = int(patch_vertex_3_y + patch_size/2)
            
            x4_center = int(patch_vertex_4_x + patch_size/2)
            y4_center = int(patch_vertex_4_y + patch_size/2)
            
            patch_size_h = patch_with_temp.size()[-1]
            patch_size_w = patch_with_temp.size()[-2]
            padding_h = h - patch_size_h
            padding_w = w - patch_size_w
            
            padding_left = x1_center - int(0.5 * patch_size_w)
            padding_right = padding_w - padding_left
            padding_top = y1_center - int(0.5 * patch_size_h)
            padding_bottom = padding_h - padding_top
            padding = nn.ZeroPad2d((int(padding_left), int(padding_right), int(padding_top), int(padding_bottom)))

            patch1_padding = padding(patch_with_temp.clone())
            patch1_mask_padding = padding(patch_mask_resize.clone())
            
            padding_left = x2_center - int(0.5 * patch_size_w)
            padding_right = padding_w - padding_left
            padding_top = y2_center - int(0.5 * patch_size_h)
            padding_bottom = padding_h - padding_top
            padding = nn.ZeroPad2d((int(padding_left), int(padding_right), int(padding_top), int(padding_bottom)))

            patch2_padding = padding(patch_with_temp.clone())
            patch2_mask_padding = padding(patch_mask_resize.clone())
            
            padding_left = x3_center - int(0.5 * patch_size_w)
            padding_right = padding_w - padding_left
            padding_top = y3_center - int(0.5 * patch_size_h)
            padding_bottom = padding_h - padding_top
            padding = nn.ZeroPad2d((int(padding_left), int(padding_right), int(padding_top), int(padding_bottom)))

            patch3_padding = padding(patch_with_temp.clone())
            patch3_mask_padding = padding(patch_mask_resize.clone())
            
            padding_left = x4_center - int(0.5 * patch_size_w)
            padding_right = padding_w - padding_left
            padding_top = y4_center - int(0.5 * patch_size_h)
            padding_bottom = padding_h - padding_top
            padding = nn.ZeroPad2d((int(padding_left), int(padding_right), int(padding_top), int(padding_bottom)))

            patch4_padding = padding(patch_with_temp.clone())
            patch4_mask_padding = padding(patch_mask_resize.clone())         
            
            patch_tmp[int(img_idx.item())] += patch1_padding
            patch_tmp[int(img_idx.item())] += patch2_padding
            patch_tmp[int(img_idx.item())] += patch3_padding
            patch_tmp[int(img_idx.item())] += patch4_padding
            patch_mask_tmp[int(img_idx.item())] += patch1_mask_padding
            patch_mask_tmp[int(img_idx.item())] += patch2_mask_padding
            patch_mask_tmp[int(img_idx.item())] += patch3_mask_padding
            patch_mask_tmp[int(img_idx.item())] += patch4_mask_padding

        patch_tf = patch_tmp
        patch_mask_tf = patch_mask_tmp
        
        patch_tf.data.clamp_(0, 1)

        return patch_tf, patch_mask_tf