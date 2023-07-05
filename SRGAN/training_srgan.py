import os
import argparse
import logging
import torch

from ..utils import str2bool
from tqdm.auto import tqdm

from RUSH_CV.utils import seed_everything
from RUSH_CV.Dataset.PexelsFlowers import PexelsFlowers
from RUSH_CV.DataLoader.DataLoader import DataLoader
from RUSH_CV.Network.SRGAN import Generator, Discriminator
from RUSH_CV.Loss.GANLoss import GeneratorLoss
from RUSH_CV.utils import RunningAverage, save_checkpoint, load_checkpoint
from RUSH_CV.Optimizer.Adam import Adam
from RUSH_CV.Evaluation.PSNR import PSNR
from RUSH_CV.Evaluation.SSIM import SSIM

pp = argparse.ArgumentParser(description="Testing")

pp.add_argument("--debug", type=str2bool, default=False)
pp.add_argument("--key_metric", type=str, default="PSNR")
pp.add_argument("--ckp_dir", type=str, default="../ckp/SRGAN/")
pp.add_argument("-s", "--scale", type=int, default=4)
pp.add_argument("--batch_size_train", type=int, default=64)
pp.add_argument("--num_worker",type=int,default=os.cpu_count() // 2)
pp.add_argument("--patch_size",type=int,default=96)

pp.add_argument("--lr", type=float, default=2e-5)
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
                                    scale=args.scale,
                                    is_debug=args.debug)
    
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
    networkG = Generator(scale_factor=args.scale)
    networkD = Discriminator()

    # Loss
    generator_criterion = GeneratorLoss()

    generator_criterion = generator_criterion.to(device)
    networkG = networkG.to(device)
    networkD = networkD.to(device)

    # Optimizer 
    optimizerG = Adam(networkG.parameters(), lr=args.lr)
    optimizerD = Adam(networkD.parameters(), lr=args.lr)
     
    # Evaluation metrics
    evaluation = {"PSNR": PSNR()} # Dictionary  must be
    key_metric = "PSNR"

    # Training

    is_best_model = False
    best_value_key_metric = 0

    loss_d_tracking = RunningAverage()
    loss_g_tracking = RunningAverage()
    score_d_tracking = RunningAverage()
    score_g_tracking = RunningAverage()

    for epoch in range(1, args.num_epoch+1):

        networkG.train()
        networkD.train()


        with tqdm(total=len(train_dataloader), desc= f"Epoch {epoch}/{args.num_epoch}: ") as t:

            for idx, data, target in train_dataloader:
                ############################
                # (1) Update D network: maximize D(x)-1-D(G(z))
                ###########################
                idx = idx.to(device)
                real_img = target.to(device)
                z = data.to(device)

                fake_img = networkG(z)
        
                networkG.zero_grad()
                real_out = networkD(real_img).mean()
                fake_out = networkD(fake_img).mean()

                d_loss = 1 - real_out + fake_out
                d_loss.backward(retain_graph=True)
                optimizerD.step()



                ############################
                # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
                ###########################
                networkG.zero_grad()
                ## The two lines below are added to prevent runetime error in Google Colab ##
                fake_img = networkG(z)
                fake_out = networkD(fake_img).mean()
                ##
                g_loss = generator_criterion(fake_out, fake_img, real_img)
                g_loss.backward()
                
                fake_img = networkG(z)
                fake_out = networkD(fake_img).mean()
                
                optimizerG.step()

                
                loss_d_tracking.update(d_loss.detach().cpu().item())
                score_d_tracking.update(real_out.detach().cpu().item())
                loss_g_tracking.update(g_loss.detach().cpu().item())
                score_g_tracking.update(fake_out.detach().cpu().item())

                t.set_postfix(loss_d=loss_d_tracking(), 
                              loss_g=loss_g_tracking(), 
                              score_d=score_d_tracking(), 
                              score_g=score_g_tracking())
                t.update()

            # Eval

            for _ , val in evaluation.items():
                val.reset()

            networkG.eval()

            with torch.no_grad():
                for idx, data, target in tqdm(valid_dataloader):

                    lr = data.to(device)
                    hr = target.to(device)
                    sr = networkG(lr)

                    idx = idx.detach()
                    lr = lr.detach()
                    hr = hr.detach()
                    sr = sr.detach()

                    for _ , val in evaluation.items():
                        val.update(hr, sr)
                    
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
                        model=networkG,
                        optimizer= optimizerG,
                        is_best=is_best_model,
                        checkpoint_path=args.ckp_dir)




    logging.info("-" * 20)
    logging.info("Evaluation on test set ...")

    test_evaluation = {"PSNR": PSNR(), "SSIM": SSIM()} # Dictionary  must be


    load_checkpoint(os.path.join(args.ckp_dir, "best.pth"), networkG)
    networkG.eval()

    with torch.no_grad():
        for idx, data, target in tqdm(test_dataloader):

            lr = data.to(device)
            hr = target.to(device)
            sr = networkG(lr)

            idx = idx.detach()
            lr = lr.detach()
            hr = hr.detach()
            sr = sr.detach()

            for _ , val in test_evaluation.items():
                val.update(hr, sr)


        performance = {}
        for key , val in test_evaluation.items():
            performance[key] = val()
        
        logging.info(performance)


    logging.info("-" * 20)
    logging.info(f"Test performance: {performance}")

if __name__ == '__main__':
    main()




