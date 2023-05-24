import argparse

from core.data import datasets

from scripts.gendata.densitymaps.density_maps import save_density_maps
from scripts.gendata.densitymaps.loading_indices import (
    save_index_offsets, save_dataset_groups)
from scripts import helper


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generates density maps from trajectories.',
        formatter_class=helper.ArgParseFormatter)
    
    # =========================================================================
    # I/O
    # =========================================================================
    parser.add_argument('directory', type=str,
                        help='root directory')
    parser.add_argument('dataset', type=str, nargs='+',
                        help='dataset name(s)')
    parser.add_argument('simulation', type=str,
                        help='simulation name')
    parser.add_argument('--save-offsets', action='store_true',
                        help='whether or not to save index offsets for data ' 
                             'loading.')
    parser.add_argument('--save-subsets', action='store_true',
                        help='whether or not to save subset splits')

    # =========================================================================
    # Density map generation
    # =========================================================================
    parser.add_argument('--sigma', type=float, default=1,
                        help='sigma of gaussian filter')

    # =========================================================================
    # Dataset splitting
    # =========================================================================
    parser.add_argument('--split', type=float, nargs=3, default=[.7, .15, .15],
                        help='three values summing to one for splitting data '
                        'into training, validation, and test sets')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed used for dataset splitting')
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    for ds in args.dataset:
        paths = helper.PathIndex(
            args.directory, dataset_name=ds, simulation_name=args.simulation)  
        loader = helper.Loader(paths)
        
        # =====================================================================
        # Generate density maps
        # =====================================================================
        if not paths.density_dir.exists():
            save_density_maps(
                paths.density_dir, loader.mesh, loader.iter_trajectories(),
                args.sigma)
            print('Saved density maps at {}'.format(paths.density_dir))
                    
        # =====================================================================
        # Generate index offsets for density maps
        # =====================================================================
        if args.save_offsets and not paths.index_offsets.exists():
            save_index_offsets(paths.density_dir, paths.index_offsets)
            print('Saved index offsets at {}'.format(paths.index_offsets))
        
        # =====================================================================
        # Generate train/val/test groups of density maps
        # =====================================================================
        if args.save_subsets and not paths.subsets_dir.exists():
            dataset = loader.snapshot_dataset(['velocity'])
            save_dataset_groups(
                dataset, args.split, args.seed, paths.subsets_dir)
            print('Saved subsets at {}'.format(paths.subsets_dir))


if __name__ == '__main__':
    main()