from torchvision.transforms import transforms
from PIL import Image


import yaml
import os
import argparse
import logging
import cv2
import numpy as np
import torch

from RUSH_CV.Network.SRCNN import SRCNN
from RUSH_CV.utils import load_checkpoint



def main():
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


    logging.debug("Detecting device ...")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    logging.debug("Processing input image ...")
    cv2_img = cv2.cvtColor(cv2.imread(args.image_input_path), cv2.COLOR_RGB2BGR)
    cv2_img_scale = cv2.resize(cv2_img, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_CUBIC)
    img_transpose = np.ascontiguousarray(cv2_img_scale.transpose((2, 0, 1)))
    img_tensor = torch.from_numpy(img_transpose).float()
    img_tensor.mul_(1.0 / 255)

    logging.debug("Loading model ...")
    network = SRCNN()
    load_checkpoint(os.path.join(args.ckp_dir, "best.pth"), network)

    logging.debug("Predicting ...")
    with torch.no_grad():
        img_tensor = img_tensor.unsqueeze(0).to(device)
        network.to(device)
        network.eval()
        result_np = network(img_tensor).cpu().detach().numpy().squeeze()

    result_img = cv2.cvtColor(result_np.transpose((1, 2, 0)), cv2.COLOR_RGB2BGR) * 255
    cv2.imwrite(args.image_output_path, result_img)
    logging.info(f"Output image shape of {result_img.shape} stored at {args.image_output_path}")



device = 
image_name = "/kaggle/input/pexels-flowers-dataset-official/valid_LR/x4/0051x4.png"
image = Image.open(image_name)
image = Variable(transforms.ToTensor()(image)).unsqueeze(0).to(device)
out = netG(image)
out_img = transforms.ToPILImage()(out[0].data.cpu())
out_img.save('output.png')