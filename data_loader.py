import random
import torch
from PIL import Image
from glob import glob


class Load_Data(torch.utils.data.Dataset):
    def __init__(self, img_root, mask_root, img_transform, mask_transform,
                 split='train'):
        super(Load_Data, self).__init__()
        self.img_transform = img_transform
        self.mask_transform = mask_transform

        '''
        we use glob(returns list of files) to recursively find all jpg files rooted at the specified directory
        
        The image source directories are:
        if split is "train" -> img_root/data_large 
        if split is "val" -> img_root/val_large 
        if split is "test" -> img_root/test_large 
        '''
        if split == 'train':
            self.paths = glob('{:s}/data_large/**/*.jpg'.format(img_root),
                              recursive=True)
        else:
            self.paths = glob('{:s}/{:s}_large/*'.format(img_root, split))

        # Similarly, source directory for the masks
        self.mask_paths = glob('{:s}/*.jpg'.format(mask_root))
        self.N_mask = len(self.mask_paths)

    def __getitem__(self, index):
        gt_img = Image.open(self.paths[index])
        gt_img = self.img_transform(gt_img.convert('RGB'))

        mask = Image.open(self.mask_paths[random.randint(0, self.N_mask - 1)])
        mask = self.mask_transform(mask.convert('RGB'))
        return gt_img * mask, mask, gt_img

    def __len__(self):
        return len(self.paths)
