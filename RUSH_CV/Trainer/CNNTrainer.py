from tqdm.auto import tqdm

from RUSH_CV.Base.Trainer import Trainer


class CNNTrainer(Trainer):
    def __init__(self, 
                 network, 
                 train_dataset, 
                 valid_dataset, 
                 test_dataset,
                 num_epoch):
        
        Trainer.__init__(self,
                         network=network,
                         train_dataset=train_dataset,
                         valid_dataset=valid_dataset,
                         test_dataset=test_dataset,
                         num_epoch=num_epoch)


    def fit(self):
        pass

    def eval(self, valid=False):
        pass

    