## Give one images -> return new images based on models (current is CNN), -> return HR imagew
import sys
sys.path.append(".")

import yaml
import os
import argparse
import logging
import cv2
import numpy as np
import torch

from RUSH_CV.Network.SRCNN import SRCNN
from RUSH_CV.utils import load_checkpoint


pp = argparse.ArgumentParser(description="Inference")

pp.add_argument("--ckp_dir", type=str, default="./ckp/SRCNN/")
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


if __name__ == "__main__":
    main()