### For replicate results on evaluation set, test set (just for CNN for now) on one particular checkpoint
import sys
sys.path.append(".")

import os
import argparse
import logging
import torch

from RUSH_CV.utils import seed_everything
from RUSH_CV.Dataset.PexelsFlowers import PexelsFlowers
from RUSH_CV.DataLoader.DataLoader import DataLoader
from RUSH_CV.Network.EDSR import EDSR
from RUSH_CV.Loss.L1Loss import L1Loss
from RUSH_CV.Evaluation.PSNR import PSNR
from RUSH_CV.Evaluation.SSIM import SSIM

from RUSH_CV.Trainer.CNNTrainer import CNNTrainer


pp = argparse.ArgumentParser(description="Evaluation mode")

pp.add_argument("--ckp_dir", type=str, default="./ckp/EDSR/")
pp.add_argument("--scale", type=int, default=4)

args = pp.parse_args()

def main():

    seed_everything(73)

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    valid_dataset = PexelsFlowers(f'data/preprocess/pexels_flowers_valid_x{args.scale}.npy',
                                   patch_size=None,
                                    is_train=False,
                                    is_pre_scale=False,
                                    scale=args.scale)
    
    test_dataset = PexelsFlowers(f'data/preprocess/pexels_flowers_test_x{args.scale}.npy',
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
    network = EDSR(num_channels=3, base_channel=64, num_residuals=4 , upscale_factor=args.scale)
    criterion = L1Loss()

    # Loss

    # Evaluation
    device = 0
    evaluation = {"PSNR": PSNR(), "SSIM":SSIM()} # Dictionary  must be

    trainer = CNNTrainer(train_dataloader=None,
                         valid_dataloader=valid_dataloader,
                         test_dataloader=test_dataloader,
                         network=network,
                         criterion=criterion,
                         optimizer=None,
                         device=device,
                         evaluation=evaluation,
                         num_epoch=None,
                         eval_epoch=None,
                         key_metric=None,
                         ckp_dir=args.ckp_dir)
    

    trainer.load_checkpoint(os.path.join(args.ckp_dir, "best.pth"))

    if trainer.device is  None:
        trainer.device = 'cpu'
    if trainer.device != 'cpu':
        torch.cuda.set_device(trainer.device)
        
    logging.info("Evaluation on " + str(trainer.device))
    trainer.network = trainer.network.to(trainer.device)

    valid_results = trainer.predict(valid=True)
    logging.info(f"Valid results: {valid_results}")

    test_results = trainer.predict(valid=False)
    logging.info(f"Test results: {test_results}")


if __name__ == '__main__':
    main()
