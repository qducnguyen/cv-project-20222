import sys
sys.path.append(".")

from torchvision.transforms import transforms
from PIL import Image


import os
import argparse
import logging
import torch
from torch.autograd import Variable

from RUSH_CV.Network.SRGAN import Generator
from RUSH_CV.utils import load_checkpoint



pp = argparse.ArgumentParser(description="Inference")

pp.add_argument("--ckp_dir", type=str, default="../ckp/SRGAN/")
pp.add_argument("--image_input_path", type=str, default="Examples/sample_inference_01.jpg")
pp.add_argument("--image_output_path", type=str, default="Examples/sample_inference_01_test.png")
pp.add_argument("-s", "--scale", type=int, default=4)

args = pp.parse_args()


def main():
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    logging.debug("Detecting device ...")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")


    logging.debug("Loading model ...")
    networkG = Generator(scale_factor=args.scale)
    load_checkpoint(os.path.join(args.ckp_dir, "best.pth"), networkG)

    networkG.to(device)
    networkG.eval()

    logging.debug("Predicting ...")

    with torch.no_grad():
        image = Image.open(args.image_input_path)
        image = Variable(transforms.ToTensor()(image)).unsqueeze(0).to(device)
        out = networkG(image)
        out_img = transforms.ToPILImage()(out[0].data.cpu())
        out_img.save(args.image_output_path)

        logging.info(f"Output image shape of {out_img.shape} stored at {args.image_output_path}")


if __name__ == '__main__':
    main()