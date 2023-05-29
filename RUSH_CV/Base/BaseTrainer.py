import logging
from tqdm.auto import tqdm

import torch

from RUSH_CV.utils import to_device


class BaseTrainer():

    def __init__(self, 
                 train_dataloader,
                 valid_dataloader,
                 test_dataloader,
                 network,
                 optimizer,
                 scheduler=None,
                 device=None,
                 evaluation=None,
                 num_epoch=10,
                 eval_epoch=1,
                 ):
        
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader

        self.network = network
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.device = device

        self.evaluation = evaluation
        self.num_epoch = num_epoch
        self.eval_epoch = eval_epoch


        self.valid_performance = None
        self.it_epoch = 0
        self.it_total = 0

        self.loss = None
        self.performance = None
        self.estimate_dataloader = None

    def fit(self, *args, **kwargs):
        if self.network is not None:
            self.init_model()
        self.start_fit()
        self.fit_epoch_loop()
        self.end_fit()
        return self

    def init_model(self, *args, **kwargs):
        if self.device is None:
            self.device = 'cpu'
        if self.device != 'cpu':
            torch.cuda.set_device(self.device)
        
        logging.info("Training on " + str(self.device))

        self.network = self.network.to(self.device)


    def start_fit(self, *args, **kwargs):
        self.network.zero_grad()
        self.network.train()

    def fit_epoch_loop(self, *args, **kwargs):
        self.it_total = 0
        for self._epoch in range(1, self.num_epoch+1):
            self.it_epoch=0
            self.start_fit_epoch()
            self.fit_batch_loop()
            self.end_fit_epoch()

            if self.eval_epoch is not None and self._epoch % self.eval_epoch==0:
                self.evaluate(valid=True)

        if self.eval_epoch is None or self.num_epoch % self.eval_epoch !=0 :
            self.evaluate(valid=True)
    
    def end_fit(self, *args, **kwargs):
        pass

    def start_fit_epoch(self, *args, **kwargs):
        pass

    def fit_batch_loop(self, *args, **kwargs):
        for idx, X, Y in tqdm(self.train_dataloader):
            self.start_fit_batch()
            idx = to_device(idx, self.device)
            X = to_device(X, self.device)
            Y = to_device(Y, self.device)

            train_result = self.train(idx=idx, X=X, Y=Y)

            self.end_fit_batch(train_result)

            self.it_total += 1
            self.it_epoch += 1

    def end_fit_epoch(self, *args, **kwargs):
        pass

    def start_fit_batch(self, *args, **kwargs):
        pass

    def train(self, idx, X, Y, *args,**kwargs):
        raise NotImplementedError
    
    def end_fit_batch(self, train_result,*args, **kwargs):
        self.loss = self.get_loss(train_result)
        self.optimize(self.loss)

    def get_loss(self,train_result,*args,**kwargs):
        raise NotImplementedError

    def optimize(self,loss,*args,**kwargs):
        self.network.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

    @torch.no_grad()
    def evaluate(self, valid=False):


        self.performance =self.predict(valid=valid)

        return self.performance

    @torch.no_grad()
    def predict(self,valid=False):
        self.start_predict(valid=valid)
        self.predict_batch_loop(valid=valid)
        self.end_predict(valid=valid)
        return self.performance

    @torch.no_grad()
    def start_predict(self, valid, *args, **kwargs):

        self.network.eval()

        if self.evaluation is not None:
            if isinstance(self.evaluation,(list,tuple)):
                for eval in self.evaluation:
                    eval.reset()
            elif isinstance(self.evaluation,dict):
                for _ , val in self.evaluation.items():
                    val.reset()
            else:
                self.evaluation.reset()


    @torch.no_grad()
    def predict_batch_loop(self, valid):
        self.estimate_dataloader = self.valid_dataloader if valid else self.test_dataloader
        with torch.no_grad():
            for idx, X, Y in tqdm(self.estimate_dataloader):
                self.start_predict_batch()

                idx = to_device(idx, self.device)
                X = to_device(X, self.device)
                Y = to_device(Y, self.device)

                Y_pred = self.predict_batch(idx=idx, X=X, Y=Y)
                self.get_evaluation(idx=idx, X=X, Y=Y, Y_pred=Y_pred)

                self.end_predict_batch()

    @torch.no_grad()
    def end_predict(self, valid, *args, **kwargs):

        if self.evaluation is None:
            performance = None
        else:
            if isinstance(self.evaluation,(list,tuple)):
                performance = []
                for eval in self.evaluation:
                    performance.append(eval())  
            elif isinstance(self.evaluation,dict):
                performance = {}
                for key , val in self.evaluation.items():
                    performance[key] = val()
            else:
                performance = self.evaluation()   

        self.network.train()


        logging.info(performance)

        self.performance = performance
        return self.performance

    @torch.no_grad()
    def start_predict_batch(self, *args, **kwargs):
        pass

    @torch.no_grad()
    def predict_batch(self, idx, X, Y, *args, **kwargs):
        raise NotImplementedError
    
    def get_evaluation(self, idx, X, Y, Y_pred):
        idx = idx.detach().cpu().numpy()
        X = X.detach().cpu().numpy()
        Y = Y.detach().cpu().numpy()
        Y_pred = Y_pred.detach().cpu().numpy()

        if self.evaluation is not None:
            if isinstance(self.evaluation,(list,tuple)):
                for eval in self.evaluation:
                    eval.update(Y, Y_pred)  
            elif isinstance(self.evaluation,dict):
                for _ , val in self.evaluation.items():
                    val.update(Y, Y_pred)
            else:
                self.evaluation.update(Y, Y_pred)       

    @torch.no_grad()
    def end_predict_batch(self, *args, **kwargs):
        pass
    

