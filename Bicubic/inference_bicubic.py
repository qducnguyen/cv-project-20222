import sys
sys.path.append(".")

import cv2
import argparse
import logging




def main(args):
    
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    img_path = args.image_input_path

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_RGB2BGR).astype(float)
    img_norm = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)

    img_scaled = cv2.resize(img_norm, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_CUBIC)[:,:,::-1]

    result_img = img_scaled * 255

    cv2.imwrite(args.image_output_path, result_img)
    logging.info(f"Output image shape of {result_img.shape} stored at {args.image_output_path}")


if __name__ == '__main__':

    pp = argparse.ArgumentParser(description="Bicubic inference mode")

    pp.add_argument("--image_input_path", type=str, default="examples/sample_inference_01.jpg")
    pp.add_argument("--image_output_path", type=str, default="examples/sample_inference_01_test.png")
    pp.add_argument("--scale", type=int, default=4)


    args = pp.parse_args()

    main(args)