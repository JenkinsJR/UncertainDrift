import numpy as np
import xarray as xr
import argparse

from core import util

from scripts import helper


def parse_args():
    parser = argparse.ArgumentParser(
        description='Processes glazur data.',
        formatter_class=helper.ArgParseFormatter)
    
    parser.add_argument('directory', type=str,
                        help='root directory')
    parser.add_argument('variable', type=str,
                        help='variable name')
    parser.add_argument('source', type=str,
                        help='source dir name')
    parser.add_argument('destination', type=str,
                        help='destination dir name')
    parser.add_argument('dataset', type=str, nargs='+',
                        help='dataset name(s)')
    
    parser.add_argument('--save-train', action='store_true',
                        help='process velocity data for training')
    parser.add_argument('--start-date', type=str,
                        help='date in format yyyymmdd. Only valid with the '
                        'flag --save_train')
    parser.add_argument('--noalign', action='store_false',
                        help='do not align C grid velocities. Only valid with '
                             'the flag --save-train')

    args = parser.parse_args()
    return args


def save_path_iterator(save_dir, start_date_string, load_paths):
    if start_date_string is None:
        start = 0
    else:
        load_stems = [path.stem for path in load_paths]
        start = load_stems.index(start_date_string)
    
    iterator = util.path.gen_numerical(save_dir, '.npy', -start)
    
    return iterator

def save_velocity(source_dir, save_dir, save_train, start_date_string=None,
                  align=True):
    load_paths = sorted(list(source_dir.glob('*')))
    if save_train:
        save_paths = save_path_iterator(
            save_dir, start_date_string, load_paths)
    else:
        save_paths = (save_dir / path.name for path in load_paths)
    
    for load_path, save_path in zip(load_paths, save_paths):
        data = xr.load_dataset(load_path)
        
        # remove the extra dimension for time and convert to npy
        if save_train:
            data = data.isel(time_counter=0).to_array().data
            
            data[np.isnan(data)] = 0
            
            if align:
                util.grid.arakawa_c_to_b(*data, inplace=True)
        
            np.save(save_path, data)
        else:
            data.to_netcdf(save_path)
           

def save_var(source_dir, save_dir, *args, start_date_string=None, **kwargs):
    load_paths = sorted(list(source_dir.glob('*')))
    save_paths = save_path_iterator(save_dir, start_date_string, load_paths)
    
    for load_path, save_path in zip(load_paths, save_paths):
        data = xr.load_dataset(load_path).data
        
        data = data.data
        data[data==0] = np.nan
            
        data[np.isnan(data)] = 0
        np.save(save_path, data)


def main():
    args = parse_args()
    
    save_func = save_velocity if args.variable == 'velocity' else save_var
    for dataset in args.dataset:
        paths = helper.PathIndex(args.directory, dataset_name=dataset)
        
        source_dir = paths.extracted_dir(
            variable=args.variable, variant=args.source)
        save_dir = paths.extracted_dir(
            variable=args.variable, variant=args.destination)
        save_dir.mkdir(exist_ok=True, parents=True)
        save_func(source_dir, save_dir, args.save_train,
                  start_date_string=args.start_date, align=args.noalign)


if __name__ == '__main__':
    main()