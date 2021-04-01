import math

import torch
import torch.nn as nn


class PConv(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size = (3, 3), stride = (1, 1), padding = 0, dilation = 1, bias = True, activ = 'relu', bn = True):
        super().__init__()
        
        # Create a 2D convolutional layer for the images of the partial convolution.
        self.image_convolution = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilation, bias = True)

        # Initialise Gaussian Weights for Image_Convolution Kernel with mean 0, standard deviation 0.02
        torch.nn.init.normal(self.image_convolution.weight, 0.0, 0.02) 

        # Create a 2D convolutional layer for the masks of the partial convolution.
        self.mask_convolution = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilation, bias = False)
        # Initialise Constant Weights of value for Mask_Convolution Kernel
        torch.nn.init.constant_(self.mask_convolution, 1.0)

        # We don't want the masks to get updated through gradients since that is what we will do manually
        for param in self.mask_convolution.parameters():
            param.requires_grad = False

        # We use Maxpool 2x2 for getting a low rank image 
        #self.pool = nn.MaxPool2d(kernel_size = pooling_dim)
        
        # Defining activation functions
        if activ == 'relu':
            self.activ = nn.ReLU()
        elif activ == 'leaky':
            self.activ = nn.LeakyReLU(negative_slope=0.2)

        # Defining batch normalization
        if bn == True:
            self.bn = nn.BatchNorm2d(output_channels)
        else:
            self.bn = None

    def forward(self, image, mask):

        # Convolve only at places where the mask is set
        output = self.image_convolution(image * mask)

        # We include a bias term
        bias = self.image_convolution.bias.view(1,-1,1,1).expand_as(output)
        
        # We will use the mask convolution layer defined earlier to obtain the 2D mask for the output.
        # However, we do not want any gradients to be computed, since we will be doing that manually in the next few lines
        with torch.no_grad():
            output_mask = self.mask_convolution(mask)

        # Those places become holes where the output mask is not set
        no_update_holes = (output_mask == 0)
        # Temporarily filling up hole regions with ones so that we do not face any errors during element by element division done in the next step
        mask_normaliser = output_mask.masked_fill_(no_update_holes, 1.0)

        # Since we are using Partial Convolutions, normalising the mask only over the no-update region is necessary
        output = ((output - bias) / mask_normaliser + bias).masked_fill_(no_update_holes, 0.0)

        # Now the new mask will be generated using the no_update_holes
        new_mask = (torch.ones_like(output)).masked_fill_(no_update_holes, 0.0)

        # # We apply max pooling, since that is a part of UNet
        # output = self.pool(output)
        
        # We apply batch normalisation, 
        if self.bn:
            output = self.bn(output)

        # We apply activation function (ReLu or Leaky)
        output = self.activ(output)
        
        # The output, along with the updated mask, goes on to the next layer
        return output, new_mask
