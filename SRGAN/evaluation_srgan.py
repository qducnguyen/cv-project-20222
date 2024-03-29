### For replicate results on evaluation set, test set (just for CNN for now) on one particular checkpoint
import sys
sys.path.append(".")

import os
import argparse
import logging
import torch

from utils import str2bool
from tqdm.auto import tqdm
from RUSH_CV.utils import seed_everything, count_parameters
from RUSH_CV.Dataset.PexelsFlowers import PexelsFlowers
from RUSH_CV.DataLoader.DataLoader import DataLoader
from RUSH_CV.Network.SRGAN import Generator, GeneratorAttention
from RUSH_CV.utils import load_checkpoint
from RUSH_CV.Evaluation.PSNR import PSNR
from RUSH_CV.Evaluation.SSIM import SSIM


def main(args):

    seed_everything(73)

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    ckp_dir = os.path.join(args.ckp_dir, "att" if args.attention else "no-att"  ,"SRGAN", "x" + str(args.scale))

    
    valid_dataset = PexelsFlowers(data_np_path=f'data/preprocess/pexels_flowers_valid_x{args.scale}.npy',
                                    patch_size=None,
                                    is_train=False,
                                    is_pre_scale=False,
                                    scale=args.scale)
    
    test_dataset = PexelsFlowers(data_np_path=f'data/preprocess/pexels_flowers_test_x{args.scale}.npy',
                                   patch_size=None,
                                   is_train=False,
                                   is_pre_scale=False,
                                   scale=args.scale)

    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=1,
                                  shuffle=False,
                                  drop_last=False)
    
    test_dataloader = DataLoader(test_dataset,
                                  batch_size=1,
                                  shuffle=False,
                                  drop_last=False)

    # Network
    if args.attention:
        networkG = GeneratorAttention(scale_factor=args.scale)
    else:
        networkG = Generator(scale_factor=args.scale)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    networkG.to(device)

    load_checkpoint(os.path.join(ckp_dir, "best.pth"), networkG)

    logging.info("There are total of " + str(count_parameters(networkG)) + " parameters.")


    test_evaluation = {"PSNR": PSNR(), "SSIM":SSIM()} 

    networkG.eval()

    logging.info("Evaluation on " + str(device))

    with torch.no_grad():
        with tqdm(total=len(valid_dataloader)) as t:
            for idx, data, target in valid_dataloader:

                lr = data.to(device)
                hr = target.to(device)
                sr = networkG(lr)

                idx = idx.detach()
                lr = lr.detach()
                hr = hr.detach()
                sr = sr.detach()

                for _ , val in test_evaluation.items():
                    val.update(hr, sr)
                
                t.set_postfix(**{u:f"{v():.3f}" for u, v in test_evaluation.items()})

                t.update()


        performance = {}
        for key , val in test_evaluation.items():
            performance[key] = val()
        
        logging.info(performance)


    logging.info("-" * 20)
    logging.info(f"Valid performance: {performance}")


    test_evaluation = {"PSNR": PSNR(), "SSIM":SSIM()} 

    with torch.no_grad():
        with tqdm(total=len(test_dataloader)) as t:
            for idx, data, target in test_dataloader:

                lr = data.to(device)
                hr = target.to(device)
                sr = networkG(lr)

                idx = idx.detach()
                lr = lr.detach()
                hr = hr.detach()
                sr = sr.detach()

                for _ , val in test_evaluation.items():
                    val.update(hr, sr)

                t.set_postfix(**{u:f"{v():.3f}" for u, v in test_evaluation.items()})
                t.update()

        performance = {}
        for key , val in test_evaluation.items():
            performance[key] = val()
        
        logging.info(performance)


    logging.info("-" * 20)
    logging.info(f"Test performance: {performance}")

if __name__ == '__main__':
        
    pp = argparse.ArgumentParser(description="Evaluation mode")

    pp.add_argument("--ckp_dir", type=str, default="./ckp/")
    pp.add_argument("--scale", type=int, default=4)
    pp.add_argument("-a", "--attention", type=str2bool, default=False)


    args = pp.parse_args()


    main(args)

