import argparse

import torch
from torch.utils.data import DataLoader

from core import model
from core.model import metrics

from scripts import helper


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluates a model that predicts density map snapshots.',
        formatter_class=helper.ArgParseFormatter)
    
    parser.add_argument('directory', type=str,
                        help='root directory')
    parser.add_argument('runid', type=str,
                        help='run id of the model')
    parser.add_argument('dataset', type=str, nargs='+',
                        help='dataset name(s)')
    parser.add_argument('simulation', type=str,
                        help='simulation name')
    parser.add_argument('--subset', type=str, default='val',
                        help='evaluation subset')
    
    parser.add_argument('--mesh', type=str, default='glazur64',
                        help='name of ocean mesh')
    parser.add_argument('--field-name', type=str, default=['velocity'],
                        nargs='+', help='name of the field to use as input')
    parser.add_argument('--field-variant', type=str, default='train_raw',
                        help='name of the field variant to use')
    parser.add_argument('--channels', type=int, default=5,
                        help='total number of input channels')
    parser.add_argument('--t-only', action='store_false',
                        help='only input the field at t')

    parser.add_argument('--batchsize', type=int, default=24,
                        help='batch size')
    parser.add_argument('--residual-map', action='store_true',
                        help='predict the residual map')
    parser.add_argument('--input-skip', action='store_true',
                        help='add the input to the NN output')

    parser.add_argument('--nw', type=int, default=8,
                        help='number of dataloader workers')
    parser.add_argument('--nmp', action='store_false',
                        help='do not use mixed precision GPU operations')
    
    args = parser.parse_args()
    return args


def evaluate_model(args, loader, ds, model_dir):
    # fetch evaluation set and create dataloader
    dataset = loader.snapshot_dataset(
        field_names=args.field_name, variant=args.field_variant,
        next_field=args.t_only, subset=args.subset)
    dataloader = DataLoader(
        dataset, args.batchsize, pin_memory=True, num_workers=args.nw)
    
    # select GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # initialisations
    land_mask = torch.tensor(~loader.mesh.mask, device=device)
    net = model.models.unet(n_channels=args.channels, n_classes=1,
                            add_input=args.input_skip)
    
    # evaluation metrics
    metric_fns = [
        metrics.MSE(land_mask, residual_map=args.residual_map),
        metrics.MassConservation(land_mask, residual_map=args.residual_map)]
    metric_fns = {x.__class__.__name__: x for x in metric_fns}

    metric_fns.update(
        {f'IOU_50': metrics.IOU(torch.tensor(
            0.5, device=device), land_mask, args.residual_map)}
    )

    # evaluate
    evaluator = model.Evaluator(device, net, dataloader, model_dir, args.nmp)
    evaluator.load_last_checkpoint()
    prefix = '{}_{}_{}'.format(ds, args.simulation, args.subset.split('/')[-1])
    evaluator.evaluate(metric_fns, prefix=prefix)


def main():
    args = parse_args()
    
    for ds in args.dataset:
        paths = helper.PathIndex(args.directory, args.mesh, ds, args.simulation)
        loader = helper.Loader(paths)

        root_model_dir = paths.model_dir / args.runid
        for model_dir in set(x.parent for x in root_model_dir.rglob('*.pt')):
            print(f'Evaluating {model_dir}')
            evaluate_model(args, loader, ds, model_dir)


if __name__ == '__main__':
    main()
