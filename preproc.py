import os 
import argparse
from utils import str2bool


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP'
]

# Create parser
pp = argparse.ArgumentParser(description="Generate .npy file for corresponding dataset.")


pp.add_argument('--name',  type=str, default="pexels_flowers_train")

pp.add_argument('--save-folder-path', type=str, default='data/',
                help='Path to output directory.')

pp.add_argument('--lr-folder-path', type=str, default='data/pexels-flowers/train_LR/',
                help='Path to LR directory including scale subfolders.')


pp.add_argument('--scale', type=int, default=3,
                help='Downsampling scale of LR images')

pp.add_argument('--all-scale', type=str2bool, default=False,
                help="Ignore scale argument and convert all scale in LR folder")


pp.add_argument('--hr-folder-path', type=str, default='data/pexels-flowers/train_HR/',
                help='Path to LR directory including scale subfolders, note that LR, HR should have similar name ')


args = pp.parse_args()


# Create directories and check dirrectory 
os.makedirs(args.save_folder_path, exist_ok=True)
assert os.path.exists(args.lr_folder_path), f"Not exists {args.lr_folder_path}"
assert os.path.exists(args.hr_folder_path), f"Not exists {args.hrs_folder_path}"






