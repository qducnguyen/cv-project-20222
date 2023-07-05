import random, os
import numpy as np
import shutil
import logging
import torch
from torch.nn.utils.rnn import PackedSequence

def seed_everything(seed):
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def to_device(X, device):

    if device is None:
        return X

    if isinstance(X, dict):
        return {key: to_device(val, device) for key, val in X.items()}

    if isinstance(X, (tuple, list)) and (type(X) != PackedSequence):
        return type(X)(to_device(x, device) for x in X)

    if isinstance(X, torch.distributions.distribution.Distribution):
        return X

    return X.to(device, non_blocking=True)



class RunningAverage():

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps) 
    
def save_checkpoint(epoch,
                    current_it_epoch,
                    current_it_total,
                    total_epoch,
                    model,
                    optimizer,
                    is_best,
                    checkpoint_path,
                    scheduler=None):
    """Saves model and training parameters at checkpoint + 'last.pth'. If is_best==True, also saves
    checkpoint + 'best.pth'
    """

    state = {'epoch': epoch,
             'current_it_epoch': current_it_epoch,
             'current_it_total': current_it_total,
             'total_epoch': total_epoch,
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'scheduler': scheduler.state_dict() if scheduler else None,
             }

    logging.info(f"Saving checkpoint...")

    file_path = os.path.join(checkpoint_path, 'last.pth')

    if not os.path.exists(checkpoint_path):
        logging.info("Checkpoint does not exist! Making directory {}".format(checkpoint_path))
        os.makedirs(checkpoint_path)
    elif os.path.exists(checkpoint_path) and not is_best:
        logging.info("Checkpoint {} does exist!. Override the checkpoint ...".format(file_path))

    torch.save(state, file_path)

    if is_best:
        file_path_best = os.path.join(checkpoint_path, 'best.pth')
        if os.path.isfile(file_path_best):
            logging.info("Checkpoint {} does exist!. Override the checkpoint ...".format(
                file_path_best))

        shutil.copyfile(file_path, os.path.join(checkpoint_path, 'best.pth'))


def load_checkpoint(checkpoint, model, optimizer=None, scheduler=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """

    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))

    logging.info(f"Load checkpoint from {checkpoint}")

    checkpoint = torch.load(checkpoint)

    # Load model
    model.load_state_dict(checkpoint['state_dict'])  # maybe epoch as well

    # Load optimizer and scheduler
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    if scheduler and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])

    start_epoch = 0
    total_epoch = 0
    current_it_epoch = 0
    current_it_total = 0

    if 'epoch' in checkpoint:
        start_epoch = checkpoint['epoch'] + 1
        total_epoch = checkpoint['total_epoch']
        current_it_epoch = checkpoint['current_it_epoch']
        current_it_total = checkpoint['current_it_total']

    return start_epoch, total_epoch, current_it_epoch, current_it_total
