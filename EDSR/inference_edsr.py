## Give one images -> return new images based on models (current is CNN), -> return HR imagew
import sys
sys.path.append(".")

from torchvision.transforms import transforms
from PIL import Image


import os
import argparse
import logging
import torch
from torch.autograd import Variable

from RUSH_CV.Network.EDSR import EDSR
from RUSH_CV.utils import load_checkpoint


pp = argparse.ArgumentParser(description="Inference")

pp.add_argument("--ckp_dir", type=str, default="./ckp/EDSR/")
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
    network = EDSR(num_channels=3, base_channel=64, num_residuals=4 , upscale_factor=args.scale)
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


if __name__ == "__main__":
    main()