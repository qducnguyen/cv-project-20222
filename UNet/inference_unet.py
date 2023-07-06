import sys
sys.path.append(".")

from torchvision.transforms import transforms
from PIL import Image


import os
import argparse
import logging
import torch
from torch.autograd import Variable

from RUSH_CV.Network.UNet import UNet2, UNet4, UNet8
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


    logging.debug("Loading model ...")

    if args.scale == 2:
        network = UNet2(3, 3)
    elif args.scale == 3:
        network = UNet8(3, 3) #
    else:
        network = UNet4(3, 3)
    
    load_checkpoint(os.path.join(args.ckp_dir, "best.pth"), network)

    network.to(device)
    network.eval()

    logging.debug("Predicting ...")

    with torch.no_grad():
        image = Image.open(args.image_input_path)
        image = Variable(transforms.ToTensor()(image)).unsqueeze(0).to(device)
        out = network(image)
        out_img = transforms.ToPILImage()(out[0].data.cpu())
        out_img.save(args.image_output_path)

        logging.info(f"Output image shape of {out_img.size} stored at {args.image_output_path}")


if __name__ == '__main__':
    main()