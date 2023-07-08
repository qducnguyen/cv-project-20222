from utils import str2bool
import argparse

from Bicubic.solver import BicubicInferencer
from SRCNN.solver import SRCNNInferencer
from VDSR.solver import VDSRInferencer
from EDSR.solver import EDSRInferencer
from SRGAN.solver import SRGANInferencer
from UNet.solver import UNetInferencer


# ===========================================================
# Evaluatoin settings
# ===========================================================
parser = argparse.ArgumentParser(description='RUSH20222 Super-resolution Inferencer')

# model configuration
parser.add_argument('--model', '-m',  type=str, default="bicubic", help="model")
parser.add_argument("--ckp_dir", type=str, default="./ckp/")
parser.add_argument('--scale', '-s',  type=int, default=4, help="Super Resolution upscale factor")
parser.add_argument("--image_input_path", "-in", type=str, default="examples/sample_inference_01.jpg")
parser.add_argument("--image_output_path", "-out", type=str, default="examples/sample_inference_01_test.png")
parser.add_argument('--attention', '-a', type=str2bool, default=False, help="Attention or Not, skip for bicubic")

# hyper-parameters
args = parser.parse_args()


def main():
    if args.model == "bicubic":
        inferencer  = BicubicInferencer(args)
    elif args.model == "srcnn":
        inferencer = SRCNNInferencer(args)
    elif args.model == "vdsr":
        inferencer = VDSRInferencer(args)
    elif args.model == "edsr":
        inferencer = EDSRInferencer(args)
    elif args.model == "srgan":
        inferencer = SRGANInferencer(args)
    elif args.model == "unet":
        inferencer = UNetInferencer(args)
    else:
        raise Exception("The model does not exist")    
    
    inferencer.run()


if __name__ == '__main__':
    main()
