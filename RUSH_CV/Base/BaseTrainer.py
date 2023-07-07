import logging
from tqdm.auto import tqdm

import torch
from RUSH_CV.utils import to_device, RunningAverage, save_checkpoint, load_checkpoint

class BaseTrainer():

    def __init__(self, 
                 train_dataloader,
                 valid_dataloader,
                 test_dataloader,
                 network,
                 optimizer,
                 device=None,
                 evaluation=None,
                 num_epoch=10,
                 eval_epoch=1,
                 key_metric=None,
                 ckp_dir=None,
                 ):
        
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader

        self.network = network
        self.optimizer = optimizer
        
        self.device = device

        self.evaluation = evaluation

        assert self.evaluation is None or isinstance(self.evaluation, dict), "evaluation variable should be None or a dict instance"

        self.num_epoch = num_epoch
        self.eval_epoch = eval_epoch


        self.valid_performance = None
        self.it_epoch = 0
        self.it_total = 0

        self.loss = None
        self.performance = None
        self.estimate_dataloader = None

        self.ckp_dir = ckp_dir

        self.loss_tracking = RunningAverage()

        self.best_value_key_metric = 0
        self.key_metric = key_metric
        self.is_best_model = False

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
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logging.info("Training on " + str(self.device))

        self.network = self.network.to(self.device)


    def start_fit(self, *args, **kwargs):
        # Settings key_metrics
        if self.evaluation is not None:
            if self.key_metric is None or self.key_metric not in self.evaluation:
                self.key_metric = self.evaluation.keys()[-1]
            
            logging.info(f"Choose {self.key_metric} as the key metric.")

        self.network.zero_grad()
        self.network.train()

    def fit_epoch_loop(self, *args, **kwargs):
        self.best_value_key_metric = 0

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
        self.is_best_model = False        


    def fit_batch_loop(self, *args, **kwargs):
        with tqdm(total=len(self.train_dataloader), desc= f"Epoch {self._epoch}/{self.num_epoch}: ") as t:
            for idx, X, Y in self.train_dataloader:
                self.start_fit_batch()
                idx = to_device(idx, self.device)
                X = to_device(X, self.device)
                Y = to_device(Y, self.device)

                train_result = self.train(idx=idx, X=X, Y=Y)

                self.end_fit_batch(train_result)

                self.it_total += 1
                self.it_epoch += 1

                t.set_postfix(loss=self.loss_tracking())
                t.update()

    def end_fit_epoch(self, *args, **kwargs):
        pass
        
           
    def start_fit_batch(self, *args, **kwargs):
        pass

    def train(self, idx, X, Y, *args,**kwargs):
        raise NotImplementedError
    
    def end_fit_batch(self, train_result,*args, **kwargs):
        self.loss = self.get_loss(train_result)
        self.optimize(self.loss)

        self.loss_tracking.update(self.loss.detach().cpu().item())

    def get_loss(self,train_result,*args,**kwargs):
        raise NotImplementedError

    def optimize(self,loss,*args,**kwargs):
        self.network.zero_grad()
        loss.backward()
        self.optimizer.step()

    @torch.no_grad()
    def evaluate(self, valid=False):

        self.performance =self.predict(valid=valid)

        if valid:
            # key metric
            key_metric_value = self.performance[self.key_metric]
            if key_metric_value > self.best_value_key_metric:
                self.best_value_key_metric = key_metric_value
                self.is_best_model = True

                logging.info(f"New best performance on {self.key_metric} : {self.best_value_key_metric}")


            if self.ckp_dir:
                save_checkpoint(
                    epoch=self._epoch,
                    current_it_epoch=self.it_epoch,
                    current_it_total=self.it_total,
                    total_epoch=self.num_epoch,
                    model=self.network,
                    optimizer=self.optimizer,
                    is_best=self.is_best_model,
                    checkpoint_path=self.ckp_dir)


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
            for _ , val in self.evaluation.items():
                val.reset()

    @torch.no_grad()
    def predict_batch_loop(self, valid):
        self.estimate_dataloader = self.valid_dataloader if valid else self.test_dataloader
        with torch.no_grad():
            with tqdm(total=len(self.estimate_dataloader)) as t:
                for idx, X, Y in self.estimate_dataloader:
                    self.start_predict_batch()

                    idx = to_device(idx, self.device)
                    X = to_device(X, self.device)
                    Y = to_device(Y, self.device)

                    Y_pred = self.predict_batch(idx=idx, X=X, Y=Y)
                    self.get_evaluation(idx=idx, X=X, Y=Y, Y_pred=Y_pred)

                    self.end_predict_batch()

                    t.set_postfix(**{u:v() for u, v in self.evaluation.items()})
                    t.update()

    @torch.no_grad()
    def end_predict(self, valid, *args, **kwargs):

        if self.evaluation is None:
            performance = None
        else:
            performance = {}
            for key , val in self.evaluation.items():
                performance[key] = val()

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
        idx = idx.detach()
        X = X.detach()
        Y = Y.detach()
        Y_pred = Y_pred.detach()

        if self.evaluation is not None:
            for _ , val in self.evaluation.items():
                val.update(Y, Y_pred)

    @torch.no_grad()
    def end_predict_batch(self, *args, **kwargs):
        pass

    @torch.no_grad()
    def load_checkpoint(self, checkpoint_path):
        load_checkpoint(checkpoint_path, self.network)
    

