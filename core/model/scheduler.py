import torch
import numpy as np
import math


class CosineAnnealingWarmRestarts(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, epochs, restarts=0, multiplier=1,
                 restart_decay=1, lr_min=0, warmup_epochs=0,
                 weight_decay_rate=0, restart_callback=None, verbose=False):
        self._epochs = epochs
        self._restarts = restarts
        self._multiplier = multiplier
        self._restart_decay = restart_decay
        self._lr_min = lr_min
        self._warmup_epochs = warmup_epochs
        self._weight_decay_rate = weight_decay_rate
        
        self._current_index = 0
        self._last_index = warmup_epochs if warmup_epochs > 0 else epochs
        self._current_restarts = 0
        self._last_reset = 0
        self._next_reset = self._last_index
        
        self.restart_callback = restart_callback
        
        super().__init__(optimizer, -1, verbose)
        self._base_lrs = self.base_lrs.copy()
        self._base_wds = [group['weight_decay']
                          for group in optimizer.param_groups]

    @property
    def n_epochs(self):
        return (self._epochs*self._multiplier**np.arange(
            self._restarts+1)).sum()

    def state_dict(self):
        d = super().state_dict()
        del d['restart_callback']
        return d
    
    def get_lr(self):
        if self._warmup_epochs > 0:
            return [self._current_index / self._last_index * base_lr
                    for base_lr in self._base_lrs]
        return [self._lr_min + (base_lr - self._lr_min) * (
            1 + math.cos(math.pi * self._current_index / self._last_index)) / 2
                for base_lr in self._base_lrs]
    
    def get_weight_decay(self):
        if self._warmup_epochs > 0:
            return [self._current_index / self._last_index * base_wd
                    for base_wd in self._base_wds]
        return [base_wd*(self._current_index+1)**self._weight_decay_rate
                for base_wd in self._base_wds]
        return [base_wd*(1-self._weight_decay_rate)**self._current_index
                for base_wd in self._base_wds]
    
    def get_last_weight_decay(self):
        return self._last_wd

    def step(self, epoch=0):
        if epoch >= self._next_reset:
            if self._warmup_epochs > 0:
                self._warmup_epochs = 0
                self._last_index = self._epochs
            else:
                self._current_restarts += 1
                if self.restart_callback is not None:
                    self.restart_callback()
                
                self._last_index = self._epochs * (
                    self._multiplier ** self._current_restarts)
                for i in range(len(self._base_lrs)):
                    self._base_lrs[i] *= self._restart_decay
            self._last_reset = epoch
            self._next_reset += self._last_index
        self._current_index = epoch - self._last_reset
        
        class _enable_get_lr_call:
            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            for i, (param_group, lr, weight_decay) in enumerate(zip(
                    self.optimizer.param_groups, self.get_lr(),
                    self.get_weight_decay())):
                param_group['lr'] = lr
                param_group['weight_decay'] = weight_decay
                self.print_lr(self.verbose, i, lr, weight_decay, epoch)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
        self._last_wd = [group['weight_decay']
                         for group in self.optimizer.param_groups]