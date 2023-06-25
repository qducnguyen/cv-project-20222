import numpy as np
import cv2
import argparse
import logging
from tqdm.auto import tqdm

from RUSH_CV.Evaluation.PSNR import PSNR
from RUSH_CV.utils import seed_everything

pp = argparse.ArgumentParser(description="Bicubic evaluation mode")

pp.add_argument("--scale", type=int, default=4)


args = pp.parse_args()

def main():
    
    seed_everything(73)

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    for stage in ["valid", "test"]:
        evaluation = {"PSNR": PSNR()} # Dictionary  must be
        data_npy = np.load(f"data/preprocess/pexels_flowers_{stage}_x{args.scale}.npy")
        for idx in tqdm(range(len(data_npy)), desc=f"Bicubic {stage}: "):
            Y_path, X_path = data_npy[idx]

            X = cv2.cvtColor(cv2.imread(X_path), cv2.COLOR_BGR2RGB).astype(float)
            X_norm = cv2.normalize(X, None, 0, 1, cv2.NORM_MINMAX)


            Y = cv2.cvtColor(cv2.imread(Y_path), cv2.COLOR_RGB2BGR).transpose(2, 0, 1).astype(float)
            Y_norm = cv2.normalize(Y, None, 0, 1, cv2.NORM_MINMAX)

            X_scaled = cv2.resize(X_norm, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_CUBIC).transpose(2, 0, 1)

            for _, val in evaluation.items():
                val.update(Y_norm, X_scaled)


        logging.info("-" * 20)
        for key, val in evaluation.items():
            logging.info(f"{key} {stage}: {val()}")


if __name__ == '__main__':
    main()