import torch
model = torch.hub.load('pytorch/vision:v0.9.0', 'fcn_resnet101', pretrained=True)
#model.eval()
import numpy as np
from torchvision import transforms
from my_evaluation import *
import cv2

input_image_path = "places2_img/4.jpg"
input_image = cv2.imread(input_image_path)
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)['out'][0]
output_predictions = output.argmax(0)


r = output_predictions.byte().cpu().numpy()
ret, out = cv2.threshold(r, 5, 255, cv2.THRESH_BINARY)
output = cv2.dilate(out, np.ones((7,7)))
output = np.bitwise_not(output)
cv2.imwrite("segment_mask.jpg", output)

inp_image, inp_mask, gt = load_image(path=input_image_path, mask_path="segment_mask.jpg")

model = PConvUNet().to(device)
load_ckpt("snapshot/25000.pth", [('model', model)])

model.eval()
evaluate_test(model, inp_image, inp_mask, gt)
out_test = cv2.imread("test123.jpg", cv2.IMREAD_COLOR)