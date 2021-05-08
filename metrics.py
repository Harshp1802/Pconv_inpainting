import numpy as np
from skimage.measure import compare_ssim as ssim

def metrics(original, generated):
    mse = np.mean((original - generated) ** 2)

    # To avoid divide by zero
    if(mse == 0):  
        return 100
    
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))

    ssim = ssim(original, generated)

    return mse, psnr, ssim
