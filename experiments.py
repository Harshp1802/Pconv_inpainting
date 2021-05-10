from metrics import metrics
import os
from my_evaluation import *
import random
import numpy as np
mask_dir = ""
mask_paths = os.listdir(mask_dir)
model = PConvUNet().to(device)
load_ckpt("snapshots/default/ckpt/1000000.pth", [('model', model)])
test_dir = ""
test_paths = os.listdir(test_dir)
A = np.zeros([1, 4])

for img_path in test_paths:
    inp_image, inp_mask, gt = load_image(path = img_path, mask_path = mask_paths[random.randint(0, len(mask_paths) - 1)])
    model.eval()
    out = evaluate_test(model, inp_image, inp_mask, gt)
    results = metrics(gt,out)
    A = np.vstack((A, results))
ans = np.sum(axis = 0)
print(mask_dir)
print("l1, mse, psnr, ssim")
print(ans/len(test_paths))