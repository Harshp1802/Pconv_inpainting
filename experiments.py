from metrics import metrics
from tqdm import tqdm
import os
from my_evaluation import *
import random
import numpy as np
import cv2


mask_dir = "masks/ratio10.0"
mask_paths = os.listdir(mask_dir)
model = PConvUNet().to(device)
load_ckpt("snapshot/25000.pth", [('model', model)])
test_dir = "Data/test_small"
test_paths = os.listdir(test_dir)
A = np.zeros([1, 4])
NS = np.zeros([1, 4])

for img_path in tqdm(test_paths):
    inp_image, inp_mask, gt = load_image(path = test_dir + "/" + img_path, mask_path = mask_dir + "/" + mask_paths[random.randint(0, len(mask_paths) - 1)])
    model.eval()
    out = evaluate_test(model, inp_image, inp_mask, gt)
    results = metrics(np.array(gt).T, np.array(out.squeeze()).T)
    A = np.vstack((A, results))

    inv_mask = 255 - inp_mask
    inp_image = np.array(inp_image, np.uint8).T
    inv_mask = np.array(inv_mask, np.uint8).T
    gt = np.array(gt).T
    out = cv2.inpaint(src=inp_image, inpaintMask=inv_mask[:, :, 0], inpaintRadius=3, flags=cv2.INPAINT_NS)
    results = metrics(np.array(gt), np.array(out.squeeze()))
    NS = np.vstack((NS, results))

ans = np.sum(A, axis = 0)
ans_NS = np.sum(NS, axis=0)
print(mask_dir)
print("l1, mse, psnr, ssim")
print(ans/len(test_paths))
print(ans_NS/len(test_paths))
