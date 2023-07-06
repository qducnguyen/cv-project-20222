from utils import str2bool
import argparse
import os

from SRCNN.solver import SRCNNTrainer

# ===========================================================
# Evaluatoin settings
# ===========================================================
parser = argparse.ArgumentParser(description='RUSH20222 Super-resolution Trainer')

parser.add_argument("--debug", type=str2bool, default=False)
parser.add_argument("--key_metric", type=str, default="PSNR")
parser.add_argument("--batch_size_train", type=int, default=4)
parser.add_argument("--num_worker",type=int,default=os.cpu_count() // 2)
parser.add_argument("--patch_size",type=int,default=64)
parser.add_argument("-a", "--attention", type=str2bool, default=False)

parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--num_epoch", type=int, default=30)
parser.add_argument("-d", "--device", type=int, default=0)

# model configuration
parser.add_argument('--model', '-m', type=str, default='bicubic', help='Choose which model is going to use')

# hyper-parameters
args = parser.parse_args()


def main():
    if args.model == "bicubic":
        print("Bicubic model do not need to be trained.")
    elif args.model == "srcnn":
        trainer  = SRCNNTrainer(args)
    else:
        raise Exception("The model does not exist")
    
    trainer.run()


if __name__ == '__main__':
    main()
