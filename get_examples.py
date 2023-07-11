import matplotlib.pyplot as plt
import glob
import cv2
from utils import str2bool
import argparse
import os

from Bicubic.solver import BicubicInferencer
from SRCNN.solver import SRCNNInferencer
from VDSR.solver import VDSRInferencer
from EDSR.solver import EDSRInferencer
from SRGAN.solver import SRGANInferencer
from UNet.solver import UNetInferencer



def get_image(lr_path, scale_factor=2):

    parser = argparse.ArgumentParser(description='RUSH20222 Super-resolution Inferencer')

    # model configuration
    parser.add_argument("--ckp_dir", type=str, default="./ckp/")
    parser.add_argument('--scale', '-s',  type=int, default=4, help="Super Resolution upscale factor")
    parser.add_argument("--image_input_path", "-in", type=str)
    parser.add_argument("--image_output_path", "-out", type=str)
    parser.add_argument('--attention', '-a', type=str2bool, default=True, help="Attention or Not, skip for bicubic")
    parser.add_argument("--metric", type=str2bool, default=False)

    # hyper-parameters
    args, unknown = parser.parse_known_args()

    output_name = lr_path.split('/')[-1].split('.')[0]

    args.image_input_path = lr_path
    args.scale = scale_factor
    os.makedirs(f"image/sr/{output_name}", exist_ok=True)

    for m in ["Bicubic", "SRCNN", "VDSR", "EDSR", "SRGAN", "UNet"]:

        args.image_output_path = f'image/sr/{output_name}/{m}.png'
        print( args.image_output_path)
        if m.lower() == "bicubic":
            inferencer  = BicubicInferencer(args)
        elif m.lower() == "srcnn":
            inferencer = SRCNNInferencer(args)
        elif m.lower() == "vdsr":
            inferencer = VDSRInferencer(args)
        elif m.lower() == "edsr":
            inferencer = EDSRInferencer(args)
        elif m.lower() == "srgan":
            inferencer = SRGANInferencer(args)
        elif m.lower() == "unet":
            inferencer = UNetInferencer(args)

        inferencer.run()
if __name__ == "__main__":
    for i in ['0184', '0239', '0360', '0459', '0774']:
        for j in [2,3,4]:
            get_image(f'image/examples/{i}x{j}.png', scale_factor=j)
            print(f"Finish {i} {j}")
