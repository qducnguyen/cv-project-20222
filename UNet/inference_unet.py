import sys
sys.path.append(".")

import numpy as np
import cv2
import os
import argparse
import logging
import torch
from RUSH_CV.Network.UNet import UNet2, UNet4, UNet3
from RUSH_CV.utils import load_checkpoint


pp = argparse.ArgumentParser(description="Inference")

pp.add_argument("--ckp_dir", type=str, default="../ckp/UNet/")
pp.add_argument("--image_input_path", type=str, default="examples/sample_inference_01.jpg")
pp.add_argument("--image_output_path", type=str, default="examples/sample_inference_01_test.png")
pp.add_argument("-s", "--scale", type=int, default=4)

args = pp.parse_args()


def main():

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    logging.debug("Detecting device ...")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    logging.debug("Processing input image ...")
    cv2_img = cv2.cvtColor(cv2.imread(args.image_input_path), cv2.COLOR_RGB2BGR)
    img_transpose = np.ascontiguousarray(cv2_img.transpose((2, 0, 1)))
    img_tensor = torch.from_numpy(img_transpose).float()
    img_tensor.mul_(1.0 / 255)

    logging.debug("Loading model ...")
    if args.scale == 2:
        network = UNet2(3, 3)
    elif args.scale == 3:
        network = UNet3(3, 3) #
    else:
        network = UNet4(3, 3)


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



if __name__ == '__main__':
    main()