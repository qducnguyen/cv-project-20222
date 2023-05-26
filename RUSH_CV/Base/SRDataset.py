import os
import numpy as np

from PIL import Image
from torch.utils.data import Dataset

class SRDataset(Dataset):
    """
    Base class of super-resolution dataset.

    """

    def __init__(self, source_transform, target_transform, args):

        self.source_transform = source_transform  # transformation for source images
        self.target_transform = target_transform  # transformation for target images

        self.args = args  # argument 

        self.data_npy_path = self.args.data_npy_path
        # Reserved varibles        
        self.data_npy = None


    def __getitem__(self, idx):
        X_path, Y_path = self.data_npy[idx]
        X = Image.open(X_path)
        Y = Image.open(Y_path) 

        if self.source_transform is not None:
            X = self.source_transform(X)
        if self.target_transform is not None:
            Y = self.target_transform(Y)

        return idx, X, Y 

    def __len__(self):
        return self.data_npy.shape[0]

    def init_dataset(self):
        """
        Running when starting fitting the model
        """
        self.data_npy = np.load(self.data_npy_path)