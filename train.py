import argparse
import numpy as np
import os
import torch
from tensorboardX import SummaryWriter
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

import opt
from my_evaluation import evaluate_train
from myloss import inPaintingLoss
from my_unet import PConvUNet
from my_unet import VGG16FeatureExtractor
from data_loader import Load_Data
from util.io import load_ckpt
from util.io import save_ckpt

os.environ["CUDA_VISIBLE_DEVICES"]="0"

class InfiniteSampler(data.sampler.Sampler):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        return iter(self.loop())

    def __len__(self):
        return 2 ** 31

    def loop(self):
        i = 0
        order = np.random.permutation(self.num_samples)
        while True:
            yield order[i]
            i += 1
            if i >= self.num_samples:
                np.random.seed()
                order = np.random.permutation(self.num_samples)
                i = 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # training options
    parser.add_argument('--root', type=str, default='./Data')
    parser.add_argument('--mask_root', type=str, default='./masks')
    parser.add_argument('--save_dir', type=str, default='./snapshots/default')
    parser.add_argument('--log_dir', type=str, default='./logs/default')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--max_iter', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--n_threads', type=int, default=2)
    parser.add_argument('--save_model_interval', type=int, default=5000)
    parser.add_argument('--vis_interval', type=int, default=5)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--resume', type=str)
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')


    # Creating the folders to save images and checkpoints
    if not os.path.exists(args.save_dir):
        os.makedirs('{:s}/images'.format(args.save_dir))
        os.makedirs('{:s}/ckpt'.format(args.save_dir))

    # Creating the folders to save logs
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    writer = SummaryWriter(log_dir=args.log_dir)

    size = (args.image_size, args.image_size)
    
    # Resize the image, convert to Tensor and finally normalise
    img_tf = transforms.Compose([
        transforms.Resize(size=size), 
        transforms.ToTensor(),
        transforms.Normalize(mean=opt.MEAN, std=opt.STD)])
    
    # Resize the image, convert to Tensor
    mask_tf = transforms.Compose([
        transforms.Resize(size=size),
        transforms.ToTensor()])

    # Returns list of file names corresponding to the training and validation data respectively
    dataset_train = Load_Data(args.root, args.mask_root, img_tf, mask_tf, 'train')
    dataset_val = Load_Data(args.root, args.mask_root, img_tf, mask_tf, 'val')

    # torch.utils.data.DataLoader class is PyTorch data loading utility.
    # It represents a Python iterable over a dataset
    iterator_train = iter(data.DataLoader(
        dataset_train, batch_size=args.batch_size,
        sampler=InfiniteSampler(len(dataset_train)),
        num_workers=args.n_threads))
    print("Loaded Dataset: ", len(dataset_train))
    
    # Loading the model
    model = PConvUNet().to(device)
    lr = args.lr

    start_iter = 0
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = inPaintingLoss(VGG16FeatureExtractor()).to(device)

    if args.resume:
        start_iter = load_ckpt(
            args.resume, [('model', model)], [('optimizer', optimizer)])
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Starting from iter ', start_iter)

    for i in tqdm(range(start_iter, args.max_iter)):
        model.train()

        image, mask, gt = [x.to(device) for x in next(iterator_train)]
        output, _ = model(image, mask)
        loss_dict = criterion(image, mask, output, gt)

        loss = 0.0
        for key, coef in opt.LAMBDA_DICT.items():
            value = coef * loss_dict[key]
            loss += value
            if (i + 1) % args.log_interval == 0:
                writer.add_scalar('loss_{:s}'.format(key), value.item(), i + 1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
            save_ckpt('{:s}/ckpt/{:d}.pth'.format(args.save_dir, i + 1),
                    [('model', model)], [('optimizer', optimizer)], i + 1)

        if (i + 1) % args.vis_interval == 0:
            model.eval()
            evaluate_train(model, dataset_val, device,
                    '{:s}/images/test_{:d}.jpg'.format(args.save_dir, i + 1))

    writer.close()
