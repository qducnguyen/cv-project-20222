from utils import str2bool
import argparse

from Bicubic.solver import BicubicInferencer

# ===========================================================
# Evaluatoin settings
# ===========================================================
parser = argparse.ArgumentParser(description='RUSH20222 Super-resolution Inferencer')

# model configuration
parser.add_argument('--scale', '-s',  type=int, default=4, help="Super Resolution upscale factor")
parser.add_argument("--image_input_path", "-in", type=str, default="examples/sample_inference_01.jpg")
parser.add_argument("--image_output_path", "-out", type=str, default="examples/sample_inference_01_test.png")
parser.add_argument('--attention', '-a', type=str2bool, default=False, help="Attention or Not, skip for bicubic")

# hyper-parameters
args = parser.parse_args()


def main():
    if args.model == "bicubic":
        evaluator  = BicubicInferencer(args)
    else:
        raise Exception("The model does not exist")
    
    evaluator.run()



if __name__ == '__main__':
    main()
