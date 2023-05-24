import sys
import numpy as np
import pandas as pd
from pathlib import Path
from contextlib import nullcontext

import torch
from torch.profiler import profile, schedule

from core.model.UNet import network as _unet

from core.util import misc


_DP = 12 # decimal print precision
_PRINT_STEP = 100 # number of batches before printing the loss


# =============================================================================
# models
# =============================================================================


class _Models:
    def __getitem__(self, x):
        return getattr(self, x)
    
    class _Model:
        def __init__(self, model, extract_fn, *args, out_modifier=None,
                     **kwargs):
            self.net = model(*args, **kwargs)
            self.__extract = extract_fn
            if out_modifier is not None:
                self.extract = lambda y: out_modifier(self.__extract(y))
            else:
                self.extract = self.__extract
                   
    class unet(_Model):
        def __init__(self, *args, **kwargs):
            super().__init__(
                _unet.UNet, lambda y: y.squeeze(1),
                *args, **kwargs)  
models = _Models()


# =============================================================================
# training and evaluation
# =============================================================================


class _Model:
    def __init__(self, device, net, checkpoint_path, mixed_precision,
                 send_input_to_loss=False):
        self._device = device
        self._net = net.net
        self._extract = net.extract
        self._checkpoint_path = Path(checkpoint_path)
        self._mp = mixed_precision
        self._send_input_to_loss = send_input_to_loss
        self.epoch = 0
        
        self._model_to_device()
    
    @property
    def loss_out_path(self):
        return self._checkpoint_path / 'epoch_loss.txt'
    
    @property
    def log_path(self):
        return self._checkpoint_path / 'log.txt'
    
    @staticmethod
    def _path_to_epoch(path):
        return int(path.stem)
    
    def _model_to_device(self):
        self._net.to(self._device)
        print('Model sent to {}'.format(self._device))
    
    def send_to_device(self, X, y, args):
        X = X.to(self._device)
        y = y.to(self._device)
        args = [arg.to(self._device) for arg in args]
        
        return X, y, args
    
    def best_epoch(self, criteria):
        if type(criteria) is not list:
            criteria = [criteria]
            
        losses = pd.read_csv(self.loss_out_path, index_col='epoch')
        selected = losses[criteria].values
        selected[-1, np.isnan(selected).all(0)] = 0
        
        best_idx = np.nanargmin(selected, 0)
        best = losses.iloc[best_idx].index.values
        
        return best
    
    def predict(self, X):
        return self._extract(self._net(X))
    
    def get_checkpoint_path(self, epoch):
        return (self._checkpoint_path / str(epoch)).with_suffix('.pt')
    
    def load_checkpoint(self, epoch, weights_only=False):
        path = self.get_checkpoint_path(epoch)
        checkpoint = torch.load(path, map_location=torch.device(self._device))
        self._net.load_state_dict(checkpoint['model_state_dict'])
        
        if not weights_only:
            self.epoch = epoch
            if hasattr(self, '_optim'):
                self._optim.load_state_dict(checkpoint['optim_state_dict'])
            if hasattr(self, '_scaler') and 'scaler' in checkpoint:
                self._scaler.load_state_dict(checkpoint['scaler'])
            if hasattr(self, '_scheduler') and self._scheduler is not None:
                self._scheduler.load_state_dict(checkpoint['scheduler'])
                
        print("Loaded epoch {} from checkpoint".format(epoch))

    def load_last_checkpoint(self):
        try:
            path = max(self._checkpoint_path.glob('*.pt'),
                       key=self._path_to_epoch)
            epoch = self._path_to_epoch(path)
            self.load_checkpoint(epoch)
        except ValueError:
            print('No checkpoint found at {}'.format(self._checkpoint_path))


class Trainer(_Model):
    def __init__(self, device, seed, epochs, train_dataloader, val_dataloaders,
                 net, loss_fn, optim, checkpoint_path, metric_fns=None,
                 send_input_to_loss=False, scheduler=None,
                 mixed_precision=True, remove_old_checkpoints=True,
                 best_epoch_criteria=None, val_step=1, profile=False,
                 profile_kwargs=None):
        super().__init__(
            device, net, checkpoint_path, mixed_precision, send_input_to_loss)
        self._seed = seed
        self._epochs = epochs
        self._train_dataloader = train_dataloader
        self._val_dataloaders = val_dataloaders
        self._loss_fn = loss_fn
        self._optim = optim
        self._metric_fns = metric_fns if metric_fns is not None else dict()
        self._scheduler = scheduler
        self._scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)
        self._remove_old_checkpoints = remove_old_checkpoints
        self._best_epoch_criteria = best_epoch_criteria
        
        self._val_step = val_step
        self._profile = profile
        self._profile_kwargs = {
            'skip_first': 50, 'wait': 0, 'warmup': 5, 'active': 50, 'repeat': 1}
        if profile_kwargs is not None:
            self._profile_kwargs.update(profile_kwargs)

    def _set_seed(self):
        torch.manual_seed(self._seed)
        np.random.seed(self._seed)
        
    def _get_max_profile_step(self):
        k = self._profile_kwargs
        return k[
            'skip_first'] + (k['wait'] + k['warmup'] + k['active'])*k['repeat']

    def _trace_handler(self, profiler):
        profile_path = self._checkpoint_path / 'profile'
        profile_path.mkdir(exist_ok=True, parents=True)
        
        output = profiler.key_averages().table(sort_by='cpu_time_total')
        with open(profile_path / '{}.txt'.format(profiler.step_num), 'w') as f:
            f.write(output)

    def _write_log(self, epoch, iteration, mean_dist, loss, lr, write_header):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, 'a') as file:
            if write_header:
                file.write('epoch,iteration,mean_dist,loss,lr\n')
            file.write(
                '{},{},{},{},{}\n'.format(
                    epoch, iteration, mean_dist, loss, lr))

    def _train_loop(self, epoch, single_iter, print_step):
        self._net.train()

        iters = len(self._train_dataloader)
        size = iters * self._train_dataloader.batch_size
        
        running_loss = 0
        print_loss = 0
        
        if self._profile:
            prof_scheduler = schedule(**self._profile_kwargs)
            stop = self._get_max_profile_step()
            context_manager = profile(
                schedule=prof_scheduler, on_trace_ready=self._trace_handler)
        else:
            context_manager = nullcontext()
        with context_manager as profiler:
            for i, (X, y, *args) in enumerate(self._train_dataloader, start=1):
                X, y, args = self.send_to_device(X, y, args)
                if self._send_input_to_loss:
                    args = [X] + args
                    
                # compute loss
                with torch.cuda.amp.autocast(enabled=self._mp):
                    loss = self.get_loss(X, y, *args)
                        
                # backprop
                self._optim.zero_grad(set_to_none=True)
                self._scaler.scale(loss).backward()
                
                # update params and learning rate
                self._scaler.step(self._optim)
                self._scaler.update()
                self._scheduler.step((epoch-1) + i / iters)
    
                loss_ = loss.item()
                
                running_loss += loss_ * len(X)
                print_loss += loss_
                
                # print average loss since the last print
                if i % print_step == 0:
                    n_trained = self._train_dataloader.batch_size * i
                    mean_loss = print_loss / print_step
                    print_loss = 0
                    
                    print('{}/{}; Loss: {:.{}f}; LR: {:.3}; WD: {:.3}'.format(
                        n_trained, size, mean_loss, _DP,
                        self._scheduler.get_last_lr()[0],
                        self._scheduler.get_last_weight_decay()[0]))
                    
                    if self._profile:
                        print(prof_scheduler(profiler.step_num).name)
                        profiler.step()
                        if profiler.step_num == stop:
                            print('Profiling complete')
                            sys.exit()
                if single_iter:
                    break
        
        return running_loss / size
    
    def _eval_loop(self, single_iter):
        self._net.eval()

        loss = torch.zeros(len(self._val_dataloaders))
        metrics = torch.zeros(
            len(self._val_dataloaders), len(self._metric_fns))
        with torch.no_grad():
            for i, dataloader in enumerate(self._val_dataloaders.values()):
                for X, y, *args in dataloader:
                    X, y, args = self.send_to_device(X, y, args)
                    if self._send_input_to_loss:
                        args = [X] + args
                    # compute loss for batch
                    with torch.cuda.amp.autocast(enabled=self._mp):
                        loss_, pred = self.get_loss(
                            X, y, *args, return_prediction=True)
                    loss[i] += loss_.item() * len(X)
                    
                    for j, metric in enumerate(self._metric_fns.values()):
                        metrics[i, j] += metric(pred, y, *args).item() * len(X)
                    
                    if single_iter:
                        break
                    
                # compute loss for dataset
                iters = len(dataloader)
                size = iters * dataloader.batch_size
                loss[i] /= size
                metrics[i] /= size
                
        return loss, metrics

    def _save_checkpoint(self, epoch):
        path = self.get_checkpoint_path(epoch)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self._net.state_dict(),
            'optim_state_dict': self._optim.state_dict(),
            'scaler': self._scaler.state_dict()
            }
        if self._scheduler is not None:
            checkpoint['scheduler'] = self._scheduler.state_dict()
            
        torch.save(checkpoint, path)
        print('Checkpoint saved at {}'.format(path))
        
    def _cleanup_checkpoints(self, epoch): 
        for path in self._checkpoint_path.glob('*.pt'):
            if int(path.stem) not in (epoch, *self.best_epoch(
                    self._best_epoch_criteria)):
                path.unlink()
        
    def _output_loss(self, epoch, train_loss, val_loss, val_metrics,
                     quiet=False):
        n_dataloaders = len(self._val_dataloaders)
        n_metrics = n_dataloaders * len(self._metric_fns)
        if val_metrics is None:
            val_metrics_flat = np.full(n_metrics, np.nan)
        else:
            val_metrics_flat = val_metrics.flatten()
        
        first_write = not self.loss_out_path.exists()
        with open(self.loss_out_path, 'a') as file:
            if first_write:
                val_headers = ['val_loss_{}'.format(x)
                               for x in self._val_dataloaders.keys()]
                val_headers += ['{}_{}'.format(x, y)
                                for x in self._val_dataloaders.keys()
                                for y in self._metric_fns.keys()]
                file.write(
                    'epoch,train_loss,{}\n'.format(','.join(val_headers)))
            
            s_template = ','.join(['{}']*(n_dataloaders+n_metrics))
            s_val = '{}'.format(s_template).format(
                *val_loss, *val_metrics_flat)
            file.write('{},{},{}\n'.format(epoch, train_loss, s_val))
        
        if not quiet:
            s = 'Train loss: {:.{}f}'.format(train_loss, _DP)
            if not np.isnan(val_loss).all():
                for i, val_name in enumerate(self._val_dataloaders.keys()):
                    s += '\n{}: {:.{}f}'.format(val_name, val_loss[i], _DP)
                    for j, metric_name in enumerate(self._metric_fns.keys()):
                        s += '\n\t{}: {:.{}f}'.format(
                            metric_name, val_metrics[i,j], _DP)
            print(s)

    def get_loss(self, X, y, *args, return_prediction=False):
        # compute prediction and loss
        pred = self.predict(X)
        loss = self._loss_fn(pred, y, *args)

        if return_prediction:
            return loss, pred
        return loss

    def train(self, single_iter=False):
        if single_iter:
            print("/!\ Warning /!\ -- set to single iteration mode\n")
        
        self.load_last_checkpoint()
        starting_epoch = self.epoch + 1
        for epoch in range(starting_epoch, self._epochs+1):
            print('Epoch {}/{} {}'.format(epoch, self._epochs, '-'*30))
            print_step = _PRINT_STEP if not single_iter else 1
            
            train_loss = self._train_loop(epoch, single_iter, print_step)
            if (epoch % self._val_step) == 0:
                val_loss, val_metrics = self._eval_loop(single_iter)
            else:
                val_loss = np.full(len(self._val_dataloaders), np.nan)
                val_metrics = None
            
            #self._scheduler.step(val_loss)
            self._save_checkpoint(epoch)
            self._output_loss(epoch, train_loss, val_loss, val_metrics)
            self._cleanup_checkpoints(epoch)


class Evaluator(_Model):
    def __init__(self, device, net, dataloader, checkpoint_path,
                 mixed_precision=True, send_input_to_loss=True):
        super().__init__(device, net, checkpoint_path, mixed_precision)
        self._dataloader = dataloader
        self._send_input_to_loss = send_input_to_loss
        
    def _get_sample_loss_path(self, prefix):
        fname = 'loss.txt'
        if prefix is not None:
            fname = '{}_'.format(prefix) + fname
        return self._checkpoint_path / fname
    
    def _write_sample_losses(self, indices, eval_metrics, metric_names,
                             write_header, prefix):
        path = self._get_sample_loss_path(prefix)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'a') as file:
            if write_header:
                file.write('index,{}\n'.format(','.join(metric_names)))
            for index, line in zip(indices, zip(*eval_metrics)):
                line = ','.join(map(lambda x: str(x.item()), line))
                file.write('{},{}\n'.format(index, line))
                
    def evaluate(self, metric_fns, prefix=None):
        self._net.eval()

        indices = iter(self._dataloader.dataset.indices)
        size = len(self._dataloader.dataset)
        with torch.no_grad():
            for i, (X, y, *args) in enumerate(self._dataloader, start=1):
                X, y, args = self.send_to_device(X, y, args)
                if self._send_input_to_loss:
                    args = [X] + args
                with torch.cuda.amp.autocast(enabled=self._mp):
                    pred = self.predict(X)
            
                values = [func(pred, y, *args) for func in metric_fns.values()]
                batch_indices = misc.yield_n(indices, len(X))
                self._write_sample_losses(
                    batch_indices, values, metric_fns.keys(), i==1, prefix)

                if i % _PRINT_STEP == 0:
                    n = self._dataloader.batch_size * i
                    print(f'{n}/{size}')
    
