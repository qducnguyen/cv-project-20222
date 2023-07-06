from utils import str2bool
import argparse

from Bicubic.solver import BicubicEvaluator

# ===========================================================
# Evaluatoin settings
# ===========================================================
parser = argparse.ArgumentParser(description='RUSH20222 Super-resolution Evaluator')

# model configuration
parser.add_argument('--scale', '-s',  type=int, default=4, help="Super Resolution upscale factor")
parser.add_argument('--model', '-m', type=str, default='bicubic', help='Choose which model is going to use')
parser.add_argument('--ckp_dir', '-ck', type=str, default="./ckp/", help="Parent checkpoint directory")
parser.add_argument('--attention', '-a', type=str2bool, default=False, help="Attention or Not")

# hyper-parameters
args = parser.parse_args()


def main():
    if args.model == "bicubic":
        evaluator  = BicubicEvaluator(args)
    else:
        raise Exception("The model does not exist")
    
    evaluator.run()



if __name__ == '__main__':
    main()
