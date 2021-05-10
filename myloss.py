import torch
import torch.nn as nn


def gram_matrix(tensor):
    """ Calculate the Gram Matrix of a given tensor """
    # get the batch_size, depth, height, and width of the Tensor
    b, d, h, w = tensor.size()
    # reshape so we're multiplying the features for each channel
    tensor = tensor.view(b, d, h * w)
    # calculate the gram matrix
    gram = torch.bmm(tensor, tensor.transpose(1, 2))/(d*h*w)
    return gram

def total_variation_loss(img):
    """ Function to calculate the Total Variation Loss of an image """
    # Shifting image by 1 pixel in both x and y directions
    tv_loss = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) + torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))
    return tv_loss

class inPaintingLoss(nn.Module):
    def __init__(self, extractor):
        """ The extractor is the function that will help 
            to extract features of an image from the loss network """
        super().__init__()
        self.l1 = nn.L1Loss()
        self.extractor = extractor
    
    def forward(self, input, mask, output, gt):
        losses = {}
        
        """ To keep the unmasked regions of the original image,
            and only fill in the empty pixels """
        computed_output = mask * input + (1-mask) * output

        losses['holes'] = self.l1((1 - mask) * output, (1 - mask) * gt)
        losses['valid'] = self.l1(mask * output, mask * gt)

        """ Now we extract the features of the input, output, computed image, and ground truth """
        
        # This code also works for grayscale images
        if output.shape[1] == 3:
            computed_output_features = self.extractor(computed_output)
            output_features = self.extractor(output)
            gt_features = self.extractor(gt)

        # if grayscale image, then convert into three channel by concatenating the same content in all channels
        elif output.shape[1] == 1:
            computed_output_features = self.extractor(torch.cat([computed_output]*3, 1))
            output_features = self.extractor(torch.cat([output]*3, 1))
            gt_features = self.extractor(torch.cat([gt]*3, 1))
        
        else:
            raise ValueError("Input should be 3 channel or grayscales")

        
        # Now we have the features, we find the Perceptual Loss
        losses['perceptual'] = 0.0
        for i in range(3):
            losses['perceptual'] += self.l1(output_features[i],gt_features[i]) + self.l1(computed_output_features[i],gt_features[i])

        # Finding the style loss
        losses['style_out'] = 0.0
        losses['style_comp'] = 0.0

        for i in range(3):
            losses['style_out'] += self.l1(gram_matrix(output_features[i]),gram_matrix(gt_features[i]))
            losses['style_comp'] += self.l1(gram_matrix(computed_output_features[i]),gram_matrix(gt_features[i]))
        

        # Finding the Total variation Loss
        losses['total_variation'] = total_variation_loss(computed_output)

        # Return all the losses
        return losses