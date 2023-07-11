import sys
sys.path.append(".")

import cv2
import argparse
import logging
import time 
from utils import str2bool
from RUSH_CV.Evaluation.PSNR import PSNR
from RUSH_CV.Evaluation.SSIM import SSIM




def main(args):
    
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    img_path = args.image_input_path
    img = cv2.imread(img_path)
    start_time = time.time()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).astype(float)
    img_norm = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)
    img_scaled = cv2.resize(img_norm, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_CUBIC)[:,:,::-1]
    result_img = img_scaled * 255
    time_interval = time.time() - start_time
    cv2.imwrite(args.image_output_path, result_img)
    logging.info(f"Output image shape of {result_img.shape} stored at {args.image_output_path} in {time_interval:.3f}")
    
    if args.metric:
        if args.image_hr_input_path is None:
            raise Exception("no HR path")
        else:
            img_metric = cv2.resize(img_norm, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_CUBIC).transpose(2, 0, 1)
            Y = cv2.cvtColor(cv2.imread(args.image_hr_input_path), cv2.COLOR_RGB2BGR).transpose(2, 0, 1).astype(float)
            Y_norm = cv2.normalize(Y, None, 0, 1, cv2.NORM_MINMAX)
            psnr = PSNR()
            psnr.update(Y_norm, img_metric)
            ssim = SSIM()
            ssim.update(Y_norm, img_metric)
            logging.info(f"PSNR: {psnr():.3f}, SSIM: {ssim():.3f}")
            return psnr, ssim


if __name__ == '__main__':

    pp = argparse.ArgumentParser(description="Bicubic inference mode")

    pp.add_argument("--image_input_path" ,type=str, default="examples/sample_inference_01.jpg")
    pp.add_argument("--image_output_path", type=str, default="examples/sample_inference_01_test.png")
    pp.add_argument("--scale", type=int, default=4)
    pp.add_argument("--image_hr_input_path", type=str, default=None)
    pp.add_argument("--metric", type=str2bool, default=False)

    args = pp.parse_args()

    main(args)