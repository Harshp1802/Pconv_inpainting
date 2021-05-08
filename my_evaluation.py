import torch
from torchvision.utils import make_grid
from torchvision.utils import save_image
from torchvision import transforms
from util.image import unnormalize
import opt
from PIL import Image
from my_unet import PConvUNet
from net import PConvUNet
from util.io import load_ckpt

device = torch.device('cuda')
def evaluate_train(model, dataset, device, filename):
    image, mask, gt = zip(*[dataset[i] for i in range(8)])
    image = torch.stack(image)
    mask = torch.stack(mask)
    gt = torch.stack(gt)
    with torch.no_grad():
        output, _ = model(image.to(device), mask.to(device))
    output = output.to(torch.device('cpu'))
    output_comp = mask * image + (1 - mask) * output

    grid = make_grid(
        torch.cat((unnormalize(image), mask, unnormalize(output),
                   unnormalize(output_comp), unnormalize(gt)), dim=0))
    save_image(grid, filename)

def evaluate_test(model, image, mask, gt):
    # image.resize_(1, 3, 256, 256)
    image = torch.stack((image, ))
    # mask.resize_(1, 3, 256, 256)
    mask = torch.stack((mask, ))
    # mask = torch.stack(mask)
    gt = torch.stack((gt, ))
    with torch.no_grad():
        output, _ = model(image.to(device), mask.to(device))
    output = output.to(torch.device('cpu'))
    output_comp = mask * image + (1 - mask) * output

    grid = make_grid(
        torch.cat((unnormalize(image), mask, unnormalize(output),
                   unnormalize(output_comp), unnormalize(gt)), dim=0))
    #print("Hello")
    save_image(grid, "test123.jpg")

def load_image(path = "places2_img/2.jpg",
 mask_path = "output.png"):
    size = (256, 256)
    gt_img = Image.open(path)
    img_tf = transforms.Compose(
        [transforms.Resize(size=size), transforms.ToTensor(),
        transforms.Normalize(mean=opt.MEAN, std=opt.STD)])
    gt_img = img_tf(gt_img.convert('RGB'))
    
    
    mask_tf = transforms.Compose(
        [transforms.Resize(size=size), transforms.ToTensor()])
    mask = Image.open(mask_path)
    mask = mask_tf(mask.convert('RGB'))
    return gt_img * mask, mask, gt_img

#if __name__ == '__main__':
    #inp_image, inp_mask, gt = load_image()

    #model = PConvUNet().to(device)
    #load_ckpt("snapshots/default/ckpt/1000000.pth", [('model', model)])

    #model.eval()
    #evaluate_my(model, inp_image, inp_mask, gt)
