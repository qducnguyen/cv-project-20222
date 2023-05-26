"""
Generate .np metadata for dataloader with the format 
    
    [[/path/to/hr_img_1.png  /path/to/low_img_1.png], 
      ...
      ...
      /path/to/hr_img_n.png   /path/to/low_img_n.png]]

"""



import os 
import glob
import argparse
import numpy as np
import logging

from utils import str2bool


# IMG_EXTENSIONS = [
#     '.jpg', '.JPG', '.jpeg', '.JPEG',
#     '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP'
# ]
# Assume that all image are pngs

# Create parser
pp = argparse.ArgumentParser(description="Generate .npy file for corresponding dataset.")


pp.add_argument('--output-name',  type=str, default="pexels_flowers_train")

pp.add_argument('--save-folder-path', type=str, default='data/preprocess/',
                help='Path to output directory.')

pp.add_argument('--hr-folder-path', type=str, default='data/pexels-flowers/train_HR/',
                help='Path to LR directory including scale subfolders, note that LR, HR should have similar name ')

pp.add_argument('--lr-folder-path', type=str, default='data/pexels-flowers/train_LR/',
                help='Path to LR directory including scale subfolders.')

pp.add_argument('--scale', type=int, default=3,
                help='Downsampling scale of LR images')

pp.add_argument('--all-scale', type=str2bool, default=False,
                help="Ignore scale argument and convert all scale in LR folder")


def main():

    args = pp.parse_args()
    # logging.debug(str("scale" in args))
    
    # Create directories and check dirrectory 
    os.makedirs(args.save_folder_path, exist_ok=True)
    assert os.path.exists(args.lr_folder_path), f"Not exists {args.lr_folder_path}"
    assert os.path.exists(args.hr_folder_path), f"Not exists {args.hrs_folder_path}"

    if not args.all_scale:
        convert_one_scale(args)
    else:
        lr_folder_paths = sorted(os.listdir(args.lr_folder_path))
        all_scales_lr = [int(name_folder[1:]) for name_folder in lr_folder_paths]

        for scale in all_scales_lr:
            args.scale = scale
            convert_one_scale(args)


def convert_one_scale(args):

    list_hr_sorted_imgs = sorted(glob.glob(args.hr_folder_path + '*.png'))

    lr_folder_paths = sorted(os.listdir(args.lr_folder_path))

    all_scales_lr = [int(name_folder[1:]) for name_folder in lr_folder_paths]
    assert args.scale in all_scales_lr, f"Not exists scale {args.scale} in LR folder {args.lr_folder_path}"

    # 2 cases x* X*
    folder_scale = f'x{args.scale}' if f'x{args.scale}' in lr_folder_paths else f'X{args.scale}'
    
    list_lr_sorted_imgs = sorted(glob.glob(args.lr_folder_path + folder_scale  + '/*.png'))

    # Create numpy array
    np_results = np.array(list(zip(list_hr_sorted_imgs, list_lr_sorted_imgs)))

    file_output_path = os.path.join(args.save_folder_path, args.output_name + '_' + folder_scale + '.npy')

    np.save(file_output_path,
            np_results)
    
    logging.info(f"File {file_output_path} is saved.")
    
if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    main()






