import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

from core import model
from core.model import loss, metrics, scheduler
from core import util

from scripts import helper


def parse_args():
    parser = argparse.ArgumentParser(
        description='Trains a model to predict the next density map snapshot.',
        formatter_class=helper.ArgParseFormatter)
    
    # =========================================================================
    # I/O required
    # =========================================================================
    parser.add_argument('directory', type=str,
                        help='root directory')
    parser.add_argument('runid', type=str,
                        help='run id of the model')
    parser.add_argument('trainset', type=str,
                        help='dataset name for training')
    parser.add_argument('valsets', type=str, nargs='+',
                        help='dataset name(s) for evaluation')
    parser.add_argument('simulation', type=str,
                        help='simulation name')
    
    # =========================================================================
    # I/O optional
    # =========================================================================
    parser.add_argument('--mesh', type=str, default='glazur64',
                        help='name of ocean mesh')
    parser.add_argument('--field-name', type=str, default=['velocity'],
                        nargs='+', help='name of the field to use as input')
    parser.add_argument('--field-variant', type=str, default='train_raw',
                        help='name of the field variant to use')
    parser.add_argument('--channels', type=int, default=5,
                        help='total number of input channels')
    parser.add_argument('--t-only', action='store_false',
                        help='only input the field at time t')
    parser.add_argument('--train-subset', type=str, default='train',
                        help='subset name for training')
    
    # =========================================================================
    # training parameters
    # =========================================================================
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for model initialisation and ' 
                        'dataset shuffling')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--batchsize', type=int, default=24,
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=12,
                        help='number of epochs')
    
    parser.add_argument('--loss', type=str, default='MAE',
                        help='name of loss function to use')
    parser.add_argument('--residual-map', action='store_true',
                        help='predict the residual map')
    parser.add_argument('--input-skip', action='store_true',
                        help='add the input to the NN output')

    parser.add_argument('--weight-decay', type=float, nargs=2,
                        default=[1, 0],
                        help='[1] weight decay value and [2] decay rate')
    
    # =========================================================================
    # additional parameters
    # =========================================================================
    parser.add_argument('--nw', type=int, default=8,
                        help='number of dataloader workers')
    parser.add_argument('--nmp', action='store_false',
                        help='do not use mixed precision GPU operations')
    parser.add_argument('--profile', action='store_true',
                        help='profile the training code')
    parser.add_argument('--debug', action='store_true',
                        help='run the script in debug mode')
    parser.add_argument('--val-step', type=int, default=12,
                        help='number of steps for model evaluation')

    args = parser.parse_args()
    return args


def extract_dataset_kwargs(args):
    common = {
        'field_names': args.field_name,
        'variant': args.field_variant,
        'next_field': args.t_only
        }
    
    train = common.copy()
    train.update(
        {'subset': args.train_subset})
    
    val = common.copy()
    val.update({
        'subset': args.train_subset.replace('train', 'val')})
    
    return train, val


def fetch_datasets(train_loader, val_loaders, train_kwargs, val_kwargs):
    train_set = train_loader.snapshot_dataset(**train_kwargs)
    
    val_sets = [val_loader.snapshot_dataset(**val_kwargs)
                for val_loader in val_loaders]
    
    return train_set, val_sets


def create_dataloaders(train_set, val_sets, batch_size, **kwargs):
    train_dataloader = DataLoader(
        train_set, batch_size, shuffle=True, **kwargs)
    val_dataloaders = []
    for val_set in val_sets:
        val_dataloaders.append(DataLoader(val_set, batch_size, **kwargs))

    return train_dataloader, val_dataloaders


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # fetch loaders
    paths = helper.PathIndex(args.directory, args.mesh, args.trainset,
                             args.simulation)
    train_loader = helper.Loader(paths)
    checkpoint_path = paths.model_dir / args.runid
    
    val_loaders = []
    for val_dataset in args.valsets:
        paths = helper.PathIndex(args.directory, args.mesh, val_dataset,
                                 args.simulation)
        val_loaders.append(helper.Loader(paths))
        
    # fetch datasets
    train_kwargs, val_kwargs = extract_dataset_kwargs(args)
    train_set, val_sets = fetch_datasets(
        train_loader, val_loaders, train_kwargs, val_kwargs)

    # create dataloaders
    train_dataloader, val_dataloaders = create_dataloaders(
        train_set, val_sets, batch_size=args.batchsize, pin_memory=True,
        drop_last=True, num_workers=args.nw)
    val_dataloaders = dict(zip(args.valsets, val_dataloaders))
    
    # select GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # initialisations
    land_mask = torch.tensor(~train_loader.mesh.mask, device=device)
    net = model.models.unet(
        n_channels=args.channels, n_classes=1, add_input=args.input_skip)
    optim = torch.optim.AdamW(
        net.net.parameters(), lr=args.lr, weight_decay=args.weight_decay[0])
    warmup_epochs = (2/(1-optim.param_groups[0]['betas'][1])) / len(
        train_dataloader)
    scheduler_fn = scheduler.CosineAnnealingWarmRestarts(
        optim, args.epochs-warmup_epochs, warmup_epochs=warmup_epochs,
        weight_decay_rate=args.weight_decay[1])

    # loss function
    loss_fn = getattr(loss, args.loss)(land_mask)
    if args.residual_map:
        loss_fn = loss.ResidualLoss(loss_fn)

    # evaluation metrics
    metric_fns = [
        metrics.MSE(land_mask, residual_map=args.residual_map, batch_mean=True),
        metrics.MAE(land_mask, residual_map=args.residual_map, batch_mean=True),
        metrics.MassConservation(
            land_mask, residual_map=args.residual_map, batch_mean=True)]
    metric_fns = {x.__class__.__name__: x for x in metric_fns}
    metric_fns.update({f'IOU_50': metrics.IOU(torch.tensor(
        0.5, device=device), land_mask, args.residual_map, batch_mean=True)})

    best_epoch_criteria = ['val_loss_{}'.format(x)
                           for x in val_dataloaders.keys()]
    
    # training
    trainer = model.Trainer(
        device, args.seed, args.epochs, train_dataloader,
        val_dataloaders, net, loss_fn, optim, checkpoint_path, metric_fns,
        send_input_to_loss=True, scheduler=scheduler_fn,
        mixed_precision=args.nmp, best_epoch_criteria=best_epoch_criteria,
        val_step=args.val_step, profile=args.profile)
    trainer.train(single_iter=args.debug)


if __name__ == '__main__':
    main()
