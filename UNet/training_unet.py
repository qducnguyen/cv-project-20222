import sys
sys.path.append(".")

import os
import argparse
import logging
import torch

from utils import str2bool
from tqdm.auto import tqdm

from RUSH_CV.utils import seed_everything
from RUSH_CV.Dataset.PexelsFlowers import PexelsFlowers
from RUSH_CV.DataLoader.DataLoader import DataLoader
from RUSH_CV.Network.UNet import UNet2, UNet4, UNet8
from RUSH_CV.utils import RunningAverage, save_checkpoint, load_checkpoint
from RUSH_CV.Optimizer.Adam import Adam
from RUSH_CV.Loss.MSELoss import MSELoss
from RUSH_CV.Loss.GradientLoss import GradientLoss
from RUSH_CV.Evaluation.PSNR import PSNR
from RUSH_CV.Evaluation.SSIM import SSIM

pp = argparse.ArgumentParser(description="UNet")

pp.add_argument("--debug", type=str2bool, default=False)
pp.add_argument("--key_metric", type=str, default="PSNR")
pp.add_argument("--ckp_dir", type=str, default="../ckp/UNet/")
pp.add_argument("-s", "--scale", type=int, default=4)
pp.add_argument("--batch_size_train", type=int, default=4)
pp.add_argument("--num_worker",type=int,default=os.cpu_count() // 2)
pp.add_argument("--patch_size",type=int,default=512)

pp.add_argument("--lr", type=float, default=1e-3)
pp.add_argument("--num_epoch", type=int, default=30)
pp.add_argument("-d", "--device", type=int, default=0)


args = pp.parse_args()


def main():

    # Seed everything
    seed_everything(73)

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')


    # DEBUG: set logging
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    # Data
    train_dataset = PexelsFlowers(data_np_path=f'data/preprocess/pexels_flowers_train_x{args.scale}.npy',
                                    patch_size=args.patch_size,
                                    is_train=True,
                                    is_pre_scale=False,
                                    scale=args.scale,
                                    is_debug=args.debug)
    
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
    if args.scale == 2:
        network = UNet2(3, 3)
    elif args.scale == 3:
        network = UNet8(3, 3) #
    else:
        network = UNet4(3, 3)

    network.weight_init(mean=0.0, std=0.01)
    
    # Loss

    criterion_mse = MSELoss()
    criterion_gra = GradientLoss()


    criterion_mse = criterion_mse.to(device)
    criterion_gra = criterion_gra.to(device)
    network = network.to(device)

    # Optimizer 
    optimizer = Adam(network.parameters(), lr=args.lr, weight_decay=1e-6)
     
    # Evaluation metrics
    evaluation = {"PSNR": PSNR()} # Dictionary  must be
    key_metric = "PSNR"

    # Training
    is_best_model = False
    best_value_key_metric = 0

    loss_tracking = RunningAverage()

    for epoch in range(1, args.num_epoch+1):
        network.train()
        is_best_model = False

        with tqdm(total=len(train_dataloader), desc= f"Epoch {epoch}/{args.num_epoch}: ") as t:

            
            for idx, data, target in train_dataloader:

                idx = idx.to(device)
                target = target.to(device)
                data = data.to(device)

                optimizer.zero_grad()
                prediction = network(data)

                loss_mse = criterion_mse(prediction, target)
                loss_gra = criterion_gra(prediction, target)
                loss = loss_mse + 0.1 * loss_gra 
                loss.backward()
                optimizer.step()

                loss_tracking.update(loss.detach().cpu().item())

                t.set_postfix(loss=loss_tracking())
                t.update()

        # Eval
        for _ , val in evaluation.items():
            val.reset()

        network.eval()

        with torch.no_grad():
            with tqdm(total=len(valid_dataloader)) as t:
                for idx, data, target in valid_dataloader:

                    lr = data.to(device)
                    hr = target.to(device)
                    sr = network(lr)

                    idx = idx.detach()
                    lr = lr.detach()
                    hr = hr.detach()
                    sr = sr.detach()

                    for _ , val in evaluation.items():
                        val.update(hr, sr)

                    t.set_postfix(**{u:f"{v():.3f}" for u, v in evaluation.items()})

                    t.update()

            performance = {}
            for key , val in evaluation.items():
                performance[key] = val()
            
            logging.info(performance)

            # key metric
            key_metric_value = performance[key_metric]
            if key_metric_value > best_value_key_metric:
                best_value_key_metric = key_metric_value
                is_best_model = True

                logging.info(f"New best performance on {key_metric} : {best_value_key_metric}")


            if args.ckp_dir:
                save_checkpoint(
                    epoch=epoch,
                    current_it_epoch=None,
                    current_it_total=None,
                    total_epoch=args.num_epoch,
                    model=network,
                    optimizer= optimizer,
                    is_best=is_best_model,
                    checkpoint_path=args.ckp_dir)


    logging.info(f"Best valid performance on {key_metric} : {best_value_key_metric}")

    logging.info("-" * 20)
    logging.info("Evaluation on test set ...")

    test_evaluation = {"PSNR": PSNR(), "SSIM": SSIM()} # Dictionary  must be

    load_checkpoint(os.path.join(args.ckp_dir, "best.pth"), network)
    network.eval()

    with torch.no_grad():
        with tqdm(total=len(test_dataloader)) as t:
            for idx, data, target in test_dataloader:

                lr = data.to(device)
                hr = target.to(device)
                sr = network(lr)

                idx = idx.detach()
                lr = lr.detach()
                hr = hr.detach()
                sr = sr.detach()

                for _ , val in test_evaluation.items():
                    val.update(hr, sr)
                
                t.set_postfix(**{u:f"{v():.3f}" for u, v in evaluation.items()})
                t.update()



        performance = {}
        for key , val in test_evaluation.items():
            performance[key] = val()
        
        logging.info(performance)


    logging.info("-" * 20)
    logging.info(f"Test performance: {performance}")

if __name__ == '__main__':
    main()




