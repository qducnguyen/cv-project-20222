import os
import numpy as np

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
        assert os.path.exists(self.data_npy_path), f"Not exist {self.data_npy_path}"

        # Reserve varibles        
        self.data_npy = None

    def init_dataset(self):
        """
        Running when starting fitting the model
        """

        self.data_npy = np.load(self.data_npy_path)



    def __getitem__(self, i):
        pass
    
    def __len__(self):
        pass