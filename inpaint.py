import numpy as np
import sys
from my_evaluation import *
import cv2 as cv

# OpenCV Utility Class for Mouse Handling
class Sketcher:
    def __init__(self, windowname, dests, colors_func):
        self.prev_pt = None
        self.windowname = windowname
        self.dests = dests
        self.colors_func = colors_func
        self.dirty = False
        self.show()
        cv.setMouseCallback(self.windowname, self.on_mouse)

    def show(self):
        cv.imshow(self.windowname, self.dests[0])
        cv.imshow(self.windowname + ": mask", self.dests[1])

    # onMouse function for Mouse Handling
    def on_mouse(self, event, x, y, flags, param):
        pt = (x, y)
        if event == cv.EVENT_LBUTTONDOWN:
            self.prev_pt = pt
        elif event == cv.EVENT_LBUTTONUP:
            self.prev_pt = None

        if self.prev_pt and flags & cv.EVENT_FLAG_LBUTTON:
            for dst, color in zip(self.dests, self.colors_func()):
                cv.line(dst, self.prev_pt, pt, color, 20)
            self.dirty = True
            self.prev_pt = pt
            self.show()


def main():

    print("Usage: python inpaint <image_path>")
    print("Keys: ")
    print("t - inpaint using FMM")
    print("n - inpaint using NS technique")
    print("m - inpaint using Partial Convolution technique")
    print("r - reset the inpainting mask")
    print("ESC - exit")

    # Read image in color mode
    sys.argv.append("places2_img/2.jpg")
    gt_img = sys.argv[1]
    img = cv.imread(sys.argv[1], cv.IMREAD_COLOR)
    #img = cv.resize(img, (512, 512))
    # If image is not read properly, return error

    # Create a copy of original image
    img_mask = img.copy()
    # Create a black copy of original image
    # Acts as a mask
    inpaintMask = np.zeros(img.shape[:2], np.uint8)
    # Create sketch using OpenCV Utility Class: Sketcher
    sketch = Sketcher('image', [img_mask, inpaintMask], lambda : ((255, 255, 255), 255))

    while True:
        ch = cv.waitKey()
        if ch == 27:
            break
        if ch == ord('t'):
            # Use Algorithm proposed by Alexendra Telea: Fast Marching Method (2004)
            # Reference: https://pdfs.semanticscholar.org/622d/5f432e515da69f8f220fb92b17c8426d0427.pdf
            res = cv.inpaint(src=img_mask, inpaintMask=inpaintMask, inpaintRadius=3, flags=cv.INPAINT_TELEA)
            cv.imshow('Inpaint Output using FMM', res)

        if ch == ord('n'):
            # Use Algorithm proposed by Bertalmio, Marcelo, Andrea L. Bertozzi, and Guillermo Sapiro: Navier-Stokes, Fluid Dynamics, and Image and Video Inpainting (2001)
            res = cv.inpaint(src=img_mask, inpaintMask=inpaintMask, inpaintRadius=3, flags=cv.INPAINT_NS)
            cv.imshow('Inpaint Output using NS Technique', res)

        if ch == ord('m'):
            inpaintMask_t = np.bitwise_not(inpaintMask)
            cv.imwrite("img_mask.jpg", img_mask)
            cv.imwrite("inpaint_mask.jpg", inpaintMask_t)
            #sketch.show()
            inp_image, inp_mask, gt = load_image(path=gt_img, mask_path="inpaint_mask.jpg")

            model = PConvUNet().to(device)
            load_ckpt("snapshot/25000.pth", [('model', model)])

            model.eval()
            evaluate_test(model, inp_image, inp_mask, gt)
            out_test = cv.imread("test123.jpg", cv.IMREAD_COLOR)
            cv.imshow('Inpaint Output', out_test)

        if ch == ord('r'):
            img_mask[:] = img
            inpaintMask[:] = 0
            sketch.show()


    print('Completed')


if __name__ == '__main__':
    main()
    cv.destroyAllWindows()
