import yaml
import argparse
import logging

from RUSH_CV.utils import seed_everything
from RUSH_CV.Dataset.PexelsFlowers import PexelsFlowers
from RUSH_CV.DataLoader.DataLoader import DataLoader
from RUSH_CV.Network.SRCNN import SRCNN

from RUSH_CV.Loss.MSELoss import MSELoss
from RUSH_CV.Optimizer.Adam import Adam

from RUSH_CV.Evaluation.PSNR import PSNR

from RUSH_CV.Trainer.CNNTrainer import CNNTrainer

# DEBUG
# pp = argparse.ArgumentParser(description="Testing")

# pp.add_argument('--data-npy-path',  type=str, default="pexels_flowers_train")

def main():

    # Seed everything
    seed_everything(73)

    # DEBUG: set logging
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    # Data

    train_dataset = PexelsFlowers(data_np_path='data/preprocess/pexels_flowers_train_x4.npy',
                                    patch_size=64,
                                    is_train=True,
                                    is_pre_scale=True,
                                    scale=4)
    
    valid_dataset = PexelsFlowers(data_np_path='data/preprocess/pexels_flowers_valid_x4.npy',
                                   patch_size=64,
                                    is_train=False,
                                    is_pre_scale=True,
                                    scale=4)
    
    test_dataset = PexelsFlowers('data/preprocess/pexels_flowers_test_x4.npy',
                                   patch_size=64,
                                    is_train=False,
                                    is_pre_scale=True,
                                    scale=4)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=4,
                                  shuffle=True,
                                  num_workers=4,
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
    network = SRCNN()

    # Loss
    criterion = MSELoss()

    # Optimizer 
    optimizer = Adam([
        {'params': network.conv1.parameters()},
        {'params': network.conv2.parameters()},
        {'params': network.conv3.parameters(), 'lr': 1e-4 * 0.1}
    ], lr=1e-4)

    # Evaluation
    evaluation = PSNR()

    device = 0
    num_epoch = 2
    eval_epoch = 1


    trainer = CNNTrainer(train_dataloader=train_dataloader,
                         valid_dataloader=valid_dataloader,
                         test_dataloader=test_dataloader,
                         network=network,
                         criterion=criterion,
                         optimizer=optimizer,
                         scheduler=None,
                         device=device,
                         evaluation=evaluation,
                         num_epoch=num_epoch,
                         eval_epoch=eval_epoch)


    trainer.fit()

    test_performance = trainer.evaluate(valid=False)

    logging.info(test_performance)


if __name__ == '__main__':
    main()




