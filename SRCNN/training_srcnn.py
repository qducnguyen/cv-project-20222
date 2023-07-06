import sys
sys.path.append(".")

import os
import argparse
import logging

from utils import str2bool

from RUSH_CV.utils import seed_everything
from RUSH_CV.Dataset.PexelsFlowers import PexelsFlowers
from RUSH_CV.DataLoader.DataLoader import DataLoader
from RUSH_CV.Network.SRCNN import SRCNN, SRCNNAttention
from RUSH_CV.Loss.MSELoss import MSELoss
from RUSH_CV.Optimizer.Adam import Adam
from RUSH_CV.Evaluation.PSNR import PSNR
from RUSH_CV.Trainer.CNNTrainer import CNNTrainer


def main(args):

    # Seed everything
    seed_everything(73)

    # DEBUG: set logging
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    ckp_dir = os.path.join(args.ckp_dir,  "SRCNN", args.scale + "-a")



    # Data
    train_dataset = PexelsFlowers(data_np_path=f'data/preprocess/pexels_flowers_train_x{args.scale}.npy',
                                    patch_size=args.patch_size,
                                    is_train=True,
                                    is_pre_scale=True,
                                    scale=args.scale,
                                    is_debug=args.debug)
    
    valid_dataset = PexelsFlowers(data_np_path=f'data/preprocess/pexels_flowers_valid_x{args.scale}.npy',
                                   patch_size=None,
                                    is_train=False,
                                    is_pre_scale=True,
                                    scale=args.scale,
                                    is_debug=args.debug)
    
    test_dataset = PexelsFlowers(data_np_path=f'data/preprocess/pexels_flowers_test_x{args.scale}.npy',
                                   patch_size=None,
                                   is_train=False,
                                   is_pre_scale=True,
                                   scale=args.scale)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size_train,
                                  shuffle=True,
                                  num_workers=args.num_worker,
                                  pin_memory=True,
                                  drop_last=True)
    
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
        network = SRCNNAttention()
    else:
        network = SRCNN()
    # Loss
    criterion = MSELoss()

    # Optimizer 
    optimizer = Adam([
        {'params': network.conv1.parameters()},
        {'params': network.conv2.parameters()},
        {'params': network.conv3.parameters(), 'lr': args.lr * 0.1}
    ], lr=args.lr)

    # Evaluation
    evaluation = {"PSNR": PSNR()} # Dictionary  must be

    device = args.device
    eval_epoch = 1
    num_epoch = args.num_epoch


    trainer = CNNTrainer(train_dataloader=train_dataloader,
                         valid_dataloader=valid_dataloader,
                         test_dataloader=test_dataloader,
                         network=network,
                         criterion=criterion,
                         optimizer=optimizer,
                         device=device,
                         evaluation=evaluation,
                         num_epoch=num_epoch,
                         eval_epoch=eval_epoch,
                         key_metric=args.key_metric,
                         ckp_dir=ckp_dir)


    trainer.fit()

    logging.info("-" * 20)
    trainer.load_checkpoint(os.path.join(ckp_dir, "best.pth"))    
    logging.info("Evaluation on test set ...")
    test_performance = trainer.evaluate(valid=False)
    logging.info("-" * 20)

    logging.info(f"Test performance: {test_performance}")


if __name__ == '__main__':

    pp = argparse.ArgumentParser(description="Training SRCNN")

    pp.add_argument("--debug", type=str2bool, default=False)
    pp.add_argument("--key_metric", type=str, default="PSNR")
    pp.add_argument("--ckp_dir", type=str, default="./ckp/")
    pp.add_argument("-s", "--scale", type=int, default=4)
    pp.add_argument("--batch_size_train", type=int, default=4)
    pp.add_argument("--num_worker",type=int,default=os.cpu_count() // 2)
    pp.add_argument("--patch_size",type=int,default=64)
    pp.add_argument("-a", "--attention", type=str2bool, default=False)

    pp.add_argument("--lr", type=float, default=1e-4)
    pp.add_argument("--num_epoch", type=int, default=30)
    pp.add_argument("-d", "--device", type=int, default=0)


    args = pp.parse_args()


    main(args)




