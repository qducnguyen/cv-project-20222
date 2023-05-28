class Trainer():

    def __init__(self, 
                 train_dataset,
                 valid_dataset,
                 test_dataset,
                 num_epoch):
        
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.num_epoch = num_epoch

    def fit(self):
        raise NotImplementedError

    def eval(self, valid=False):
        raise NotImplementedError

