import torch

from RUSH_CV.Base.BaseTrainer import BaseTrainer


class CNNTrainer(BaseTrainer):
    def __init__(self, 
                 train_dataloader,
                 valid_dataloader,
                 test_dataloader,
                 network,
                 criterion,
                 optimizer,
                 scheduler=None,
                 device=None,
                 evaluation=None,
                 num_epoch=10,
                 eval_epoch=1,
                 key_metric=None,
                 ckp_dir=None):
        
        BaseTrainer.__init__(self,
                            train_dataloader=train_dataloader,
                            valid_dataloader=valid_dataloader,
                            test_dataloader=test_dataloader,
                            network=network,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            device=device,
                            evaluation=evaluation,
                            num_epoch=num_epoch,
                            eval_epoch=eval_epoch,
                            key_metric=key_metric,
                            ckp_dir=ckp_dir)

        self.criterion = criterion

    def train(self, idx, X, Y, *args, **kwargs):
        
        logits = self.network(X)
        preds = logits

        return logits, preds, Y


    def get_loss(self, train_result, *args, **kwargs):
        logits, preds, Y = train_result
        loss = self.criterion(preds, Y)
        return loss

    @torch.no_grad()
    def predict_batch(self, idx, X, Y, *args, **kwargs):
        preds = self.network(X).clamp(0.0, 1.0)
        return preds
    