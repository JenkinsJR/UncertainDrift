import numpy as np
from datetime import timedelta as dt
import argparse

from .seeds import save_seeds
from .trajectories import save_trajectories
from scripts import helper


SECONDS_IN_DAY = 86400


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate ensembles of trajectories.',
        formatter_class=helper.ArgParseFormatter)
    
    # =========================================================================
    # I/O
    # =========================================================================
    parser.add_argument('directory', type=str,
                        help='root directory')
    parser.add_argument('dataset', type=str,
                        help='dataset name')
    parser.add_argument('field', type=str,
                        help='field name')
    parser.add_argument('simulation', type=str,
                        help='simulation name')
    parser.add_argument('--from-seeds', type=str,
                        help='use the seeds from another simulation')
    parser.add_argument('--start-index', type=int, default=0,
                        help='starting seed index to generate trajectories. '
                             'If greater than 0, existing trajectories may be '
                             'overwritten')
    parser.add_argument('--end-index', type=int,
                        help='end seed index to generate trajectories. ')
    # =========================================================================
    # Particle deployment - seed generation
    # =========================================================================
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for generating the deployment '
                        'positions')
    parser.add_argument('--samples', type=int, default=20000,
                        help='number of trajectory ensembles to deploy')
    parser.add_argument('--start-date', type=str,
                        help='start date in format yyyy-mm-dd.')
    parser.add_argument('--end-date', type=str,
                        help='end date (exclusive) in format yyyy-mm-dd.')
    # =========================================================================
    # Particle deployment - ensemble generation
    # =========================================================================
    parser.add_argument('--particles', type=int, default=10000,
                        help='number of particles per ensemble')
    parser.add_argument('--radius', type=float, default=5,
                        help='size of the uncertainty radius in km')
    # =========================================================================
    # Particle advection
    # =========================================================================
    parser.add_argument('--runtime', type=int, default=15,
                        help='runtime length in days of the trajectories')
    parser.add_argument('--ostep', type=int, default=1,
                        help='timestep in days of snapshot outputs')
    parser.add_argument('--pstep', type=int, default=6,
                        help='timestep in hours of particle advection')
    parser.add_argument('--advection-kernel', type=str, default='AdvectionRK4',
                        help='name of advection kernel')
    parser.add_argument('--compiled-name', type=str,
                        help='name of compiled kernel')
    parser.add_argument('--compile-only', action='store_true',
                        help='terminate after kernel compilation')

    args = parser.parse_args()
    return args


def should_generate_particle_seeds(args, loader):
    if args.from_seeds is not None:
        return False

    paths = loader.paths

    return not paths.seeds.exists()


def should_run_simulation(args, loader):
    return args.start_index > 0 or args.end_index is not None or (
        not loader.paths.trajectories_dir.exists())


def generate_particle_seeds(args, loader):
    paths = loader.paths
    
    print("Generating particle seeds at {}".format(paths.seeds))
    np.random.seed(args.seed)
    
    # extract valid time positions given the specified start and end date
    time_positions = loader.time_positions()
    keep_times = np.ones_like(time_positions, dtype=bool)
    
    if args.start_date is not None:
        diff = time_positions.astype('datetime64[D]') - (
            np.datetime64(args.start_date))
        keep_times &= diff.astype(int) >= 0
    if args.end_date is not None:
        diff = time_positions.astype('datetime64[D]') - (
            np.datetime64(args.end_date))
        keep_times &= diff.astype(int) < 0
    time_positions = time_positions[keep_times][:-args.runtime]

    assert len(time_positions) > 0
    
    save_seeds(paths.seeds, loader.mesh, time_positions, args.samples)


def generate_trajectories(args, loader, particle_runtime, particle_output_dt,
                          particle_dt, kernel_path):
    paths = loader.paths
    
    if args.from_seeds is not None:
        paths_alt = helper.PathIndex(
            args.directory, dataset_name=paths.dataset_name,
            simulation_name=args.from_seeds)
        seeds_loader = helper.Loader(paths_alt)
    else:
        seeds_loader = loader

    seeds = seeds_loader.seeds()
    fieldset = loader.fieldset(
        variant=args.field, chunksize=False, deferred_load=False)

    save_trajectories(
        paths.trajectories_dir, seeds, fieldset, particle_runtime,
        particle_output_dt, particle_dt, args.advection_kernel, kernel_path,
        loader.mesh, args.particles, args.radius, args.start_index,
        args.end_index, args.compile_only)


def run_simulations(args, loader, particle_runtime, particle_output_dt,
                    particle_dt, kernel_path):
    generate_trajectories(
        args, loader, particle_runtime, particle_output_dt,
        particle_dt, kernel_path)


def main():
    args = parse_args()

    paths = helper.PathIndex(args.directory, dataset_name=args.dataset,
                                simulation_name=args.simulation)
    loader = helper.Loader(paths)

    if should_generate_particle_seeds(args, loader):
        generate_particle_seeds(args, loader)
    
    if should_run_simulation(args, loader):
        particle_runtime = dt(days=args.runtime)
        particle_output_dt = dt(days=args.ostep)
        particle_dt = dt(hours=args.pstep)

        if args.compiled_name is None:
            args.compiled_name = args.advection_kernel
        kernel_path = paths.kernels_dir / args.compiled_name

        run_simulations(
            args, loader, particle_runtime, particle_output_dt,
            particle_dt, kernel_path)


if __name__ == '__main__':
    main()