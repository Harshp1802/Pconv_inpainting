import math

import torch
import torch.nn as nn
from my_pconv import PConv
import torch.nn.functional as F

# Important formulae
# 1) Padding should be n for kernel size of 2 * n + 1 (This is for dimension conservation)

class PConvUNet(nn.Module):
    def __init__(self, num_layers = 7, input_channels = 3):
        super().__init__()
    
        self.upsampling_mode = 'bilinear'
        
        # Creating a list of all encoders
        self.enc_prim = nn.ModuleList()
        self.enc_inter = nn.ModuleList()
        
        self.num_layers = num_layers

        # Creating a list of all encoders
        self.enc_prim.append(PConv(input_channels, 64, 7, 1, 3, bn=False))
        self.enc_inter.append(PConv(64, 64, 7, 1, 3, bn=False))

        self.enc_prim.append(PConv(64, 128, 5, 1, 2, bn=True))
        self.enc_inter.append(PConv(128, 128, 5, 1, 2, bn=True))

        self.enc_prim.append(PConv(128, 256, 5, 1, 2, bn=True))
        self.enc_inter.append(PConv(256, 256, 5, 1, 2, bn=True))

        self.enc_prim.append(PConv(256, 512, 3, 1, 1, bn=True))
        self.enc_inter.append(PConv(512, 512, 3, 1, 1, bn=True))


        for i in range(5, num_layers + 1):
            self.enc_prim.append(PConv(512, 512, 3, 1, 1, bn=True))
            self.enc_inter.append(PConv(512, 512, 3, 1, 1, bn=True))

        # Creating a list of all decoders
        self.dec_prim = nn.ModuleList()
        self.dec_inter = nn.ModuleList()
        
        # Creating a list of all decoders
        self.dec_prim.append(PConv(64 + input_channels, input_channels, 3, 1, 1, bn=False, activ = None))
        self.dec_inter.append(PConv(input_channels, input_channels, 3, 1, 1, bn=False, activ = None))

        self.dec_prim.append(PConv(128 + 64, 64, 3, 1, 1, bn=True, activ = 'leaky'))
        self.dec_inter.append(PConv(64, 64, 3, 1, 1, bn=True, activ = 'leaky'))

        self.dec_prim.append(PConv(256 + 128, 128, 3, 1, 1, bn=True, activ = 'leaky'))
        self.dec_inter.append(PConv(128, 128, 3, 1, 1, bn=True, activ = 'leaky'))

        self.dec_prim.append(PConv(512 + 256, 256, 3, 1, 1, bn=True, activ = 'leaky'))
        self.dec_inter.append(PConv(256, 256, 3, 1, 1, bn=True, activ = 'leaky'))


        for i in range(5, num_layers + 1):
            self.dec_prim.append(PConv(512 + 512, 512, 3, 1, 1, bn=True, activ = 'leaky'))
            self.dec_inter.append(PConv(512, 512, 3, 1, 1, bn=True, activ = 'leaky'))

    
    def forward(self, image, mask):
        h_dict = {}  # for the output of enc_N
        h_mask_dict = {}  # for the output of enc_N

        h_dict[0], h_mask_dict[0] = image, mask


        h_key_prev = 0
        for i in range(1, self.num_layers + 1):
            #operator = nn.Sequential(
                #self.enc_prim[i-1],
                #self.enc_inter[i-1]
            #)
            pool = nn.MaxPool2d(2)
            x, y = self.enc_prim[i-1](h_dict[h_key_prev], h_mask_dict[h_key_prev])
            h_dict[i], h_mask_dict[i] = x, y
            # h_dict[i], h_mask_dict[i] = self.enc_inter[i-1](x, y)
            # h_dict[i], h_mask_dict[i] = operator(h_dict[h_key_prev], h_mask_dict[h_key_prev])
            h_dict[i] = pool(h_dict[i])
            h_mask_dict[i] = pool(h_mask_dict[i])
            h_key_prev = i

        h_key = self.num_layers
        h, h_mask = h_dict[h_key], h_mask_dict[h_key]

        # concat upsampled output of h_enc_N-1 and dec_N+1, then do dec_N
        # (exception)
        #                            input         dec_2            dec_1
        #                            h_enc_7       h_enc_8          dec_8

        for i in range(self.num_layers, 0, -1):
            enc_h_key = i - 1
            dec_l_key = i

            # We upsample using the bilinear mode
            h = F.interpolate(h, scale_factor=2, mode=self.upsampling_mode, align_corners=False)
            h_mask = F.interpolate(h_mask, scale_factor=2, mode='bilinear', align_corners=False)
            #print ("-------", i, h.shape, h_dict[enc_h_key].shape)

            h = torch.cat([h, h_dict[enc_h_key]], dim=1)
            h_mask = torch.cat([h_mask, h_mask_dict[enc_h_key]], dim=1)

            #operator = nn.Sequential(
                #self.dec_prim[i],
                #self.dec_inter[i],
                #nn.Upsample(scale_factor=2, mode = self.upsampling_mode, align_corners = True)
            #)
            x, y = self.dec_prim[i - 1](h, h_mask)
            h, h_mask = x, y
            # h, h_mask = self.dec_inter[i - 1](x, y)
            #h = nn.Upsample(scale_factor=2, mode = self.upsampling_mode, align_corners = True)(h)
            #h_mask = nn.Upsample(scale_factor=2, mode = self.upsampling_mode, align_corners = True)(h_mask)

            #h, h_mask = operator(h, h_mask)


        return h, h_mask

    def train(self, mode=True):
        # call the default class train
        super().train(mode)