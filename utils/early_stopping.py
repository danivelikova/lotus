'''Code: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py '''
import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, ckpt_save_path_model1=None, ckpt_save_path_model2=None, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.ckpt_save_path_model1 = ckpt_save_path_model1
        self.ckpt_save_path_model2 = ckpt_save_path_model2
        self.trace_func = trace_func
        self.best_epoch = None
        
    def __call__(self, val_loss, epoch, model1, model2):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, epoch, model1, model2)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, epoch, model1, model2)
            self.counter = 0

    def save_checkpoint(self, val_loss, epoch, model1, model2):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model1.state_dict(), self.ckpt_save_path_model1 + '_epoch=' + str(epoch) + '.pt')
        torch.save(model2.state_dict(), self.ckpt_save_path_model2 + '_epoch=' + str(epoch) + '.pt')
        self.val_loss_min = val_loss
        self.best_epoch = epoch
        print(f'SAVE CKPT best val_loss: {self.val_loss_min} at best_epoch: {self.best_epoch}')
