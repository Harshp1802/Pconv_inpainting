import math

import torch
import torch.nn as nn
from my_pconv import PConv

# Important formulae
# 1) Padding should be n for kernel size of 2 * n + 1 (This is for dimension conservation)

class PConvUNet(nn.Module):
    def __init__(self, num_layers = 7, input_channels = 3):
        super().__init__()
    
        self.upsampling_mode = 'bilinear'
        
        # Creating a list of all encoders
        self.enc_prim = nn.ModuleList()
        self.enc_inter = nn.ModuleList()
        
        # Creating a list of all encoders
        self.num_layers = num_layers
        self.enc_prim.append(PConv(3, 64, 7, 1, 3, bn=False))
        self.enc_inter.append(PConv(3, 64, 7, 1, 3, bn=False))
        self.enc_prim.append(PConv(3, 64, 7, 1, 3, bn=False))
        self.enc_inter.append(PConv(64, 64, 7, 1, 3, bn=False))
        self.enc__prim.append(PConv(64, 128, 5, 1, 2, bn=False))
        self.enc_inter.append(PConv(128, 128, 5, 1, 2, bn=False))
        self.enc_prim.append(PConv(128, 256, 5, 1, 2, bn=False))
        self.enc_inter.append(PConv(256, 256, 5, 1, 2, bn=False))
        self.enc_prim.append(PConv(256, 512, 3, 1, 1, bn=False))
        self.enc_inter.append(PConv(512, 512, 3, 1, 1, bn=False))


        for i in range(5, num_layers + 1):
            self.enc_prim.append(PConv(512, 512, 3, 1, 1, bn=False))
            self.enc_inter.append(PConv(512, 512, 3, 1, 1, bn=False))

        # Creating a list of all decoders
        self.dec_prim = nn.ModuleList()
        self.dec_inter = nn.ModuleList()
        
        # Creating a list of all encoders
        self.enc_prim.append(PConv(3, 64, 7, 1, 3, bn=False))
        self.enc_inter.append(PConv(3, 64, 7, 1, 3, bn=False))
        self.enc_prim.append(PConv(3, 64, 7, 1, 3, bn=False))
        self.enc_inter.append(PConv(64, 64, 7, 1, 3, bn=False))
        self.enc__prim.append(PConv(64, 128, 5, 1, 2, bn=False))
        self.enc_inter.append(PConv(128, 128, 5, 1, 2, bn=False))
        self.enc_prim.append(PConv(128, 256, 5, 1, 2, bn=False))
        self.enc_inter.append(PConv(256, 256, 5, 1, 2, bn=False))
        self.enc_prim.append(PConv(256, 512, 3, 1, 1, bn=False))
        self.enc_inter.append(PConv(512, 512, 3, 1, 1, bn=False))


        for i in range(5, num_layers + 1):
            self.enc_prim.append(PConv(512, 512, 3, 1, 1, bn=False))
            self.enc_inter.append(PConv(512, 512, 3, 1, 1, bn=False))


        