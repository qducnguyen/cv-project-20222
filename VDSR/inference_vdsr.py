## Give one images -> return new images based on models (current is CNN), -> return HR imagew
import sys
sys.path.append(".")

import os
import argparse
import logging
import cv2
import numpy as np
import torch
import time

from utils import str2bool
from RUSH_CV.Evaluation.PSNR import PSNR
from RUSH_CV.Evaluation.SSIM import SSIM
from RUSH_CV.Network.VDSR import VDSR, VDSRAttention
from RUSH_CV.utils import load_checkpoint



def main(args):

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    ckp_dir = os.path.join(args.ckp_dir, "att" if args.attention else "no-att", "VDSR", "x" + str(args.scale))


    logging.debug("Detecting device ...")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    logging.debug("Processing input image ...")
    cv2_img = cv2.imread(args.image_input_path)
    start_time_1 = time.time()
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
    cv2_img_scale = cv2.resize(cv2_img, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_CUBIC)
    img_transpose = np.ascontiguousarray(cv2_img_scale.transpose((2, 0, 1)))
    img_tensor = torch.from_numpy(img_transpose).float()
    img_tensor.mul_(1.0 / 255)

    end_time_1 = time.time()
    logging.debug("Loading model ...")
    
    if args.attention:
        network = VDSRAttention(num_channels=3, base_channels=64, num_residuals=18)
    else:
        network = VDSR(num_channels=3, base_channels=64, num_residuals=18)

    load_checkpoint(os.path.join(ckp_dir, "best.pth"), network)
    network.to(device)
    network.eval()

    logging.debug("Predicting ...")
    
    start_time_2 = time.time()
    with torch.no_grad():
        img_tensor = img_tensor.unsqueeze(0).to(device)
        result_tensor = network(img_tensor).clamp(0.0, 1.0)
        result_np = result_tensor.cpu().detach().numpy().squeeze()

    result_img = cv2.cvtColor(result_np.transpose((1, 2, 0)), cv2.COLOR_RGB2BGR) * 255
    
    time_interval = time.time() - start_time_2 + end_time_1 - start_time_1
    cv2.imwrite(args.image_output_path, result_img)
    logging.info(f"Output image shape of {result_img.shape} stored at {args.image_output_path} in {time_interval:.3f}")

    if args.metric:
        if args.image_hr_input_path is None:
            raise Exception("no HR path")
        else:
            cv2_img_hr = cv2.imread(args.image_hr_input_path)
            cv2_img_hr = cv2.cvtColor(cv2_img_hr, cv2.COLOR_RGB2BGR)
            img_transpose_hr = np.ascontiguousarray(cv2_img_hr.transpose((2, 0, 1)))
            img_tensor_hr = torch.from_numpy(img_transpose_hr).float()
            img_tensor_hr.mul_(1.0 / 255)
            img_tensor_hr = img_tensor_hr.to(device).unsqueeze(0)

            psnr = PSNR()
            psnr.update(img_tensor_hr, result_tensor)
            ssim = SSIM()
            ssim.update(img_tensor_hr, result_tensor)
            logging.info(f"PSNR: {psnr():.3f}, SSIM: {ssim():.3f}")
            return psnr, ssim

if __name__ == "__main__":
        
    pp = argparse.ArgumentParser(description="Inference")

    pp.add_argument("--ckp_dir", type=str, default="./ckp/")
    pp.add_argument("--image_input_path", type=str, default="examples/sample_inference_01.jpg")
    pp.add_argument("--image_output_path", type=str, default="examples/sample_inference_01_test.png")
    pp.add_argument("-s", "--scale", type=int, default=4)
    pp.add_argument("-a", "--attention", type=str2bool, default=False)
    pp.add_argument("--image_hr_input_path", type=str, default=None)
    pp.add_argument("--metric", type=str2bool, default=False)


    args = pp.parse_args()


    main(args)