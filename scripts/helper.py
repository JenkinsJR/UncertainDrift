import numpy as np
import xarray as xr
from pathlib import Path
import argparse

from core.data import mesh
from core import util

    
class PathIndex:
    def __init__(self, root_dir, mesh='glazur64', dataset_name=None,
                 simulation_name=None):
        self.root_dir = Path(root_dir)
        self.mesh = mesh
        self.dataset_name = dataset_name
        self.simulation_name = simulation_name
    
    @property
    def _extracted_dir(self):
        return self.data_dir / 'extracted'
    
    @property
    def _trajectories_dir(self):
        return self.simulation_dir / 'trajectories'
    
    @property
    def _density_dir(self):
        return self.simulation_dir / 'density'
    
    @property
    def model_dir(self):
        return self.root_dir / 'models'
    
    @property
    def mesh_dir(self):
        return self.root_dir / 'data' / self.mesh
    
    @property
    def mesh_mask(self):
        return self.mesh_dir / 'mesh_mask.nc'
    
    @property
    def dataset_dir(self):
        return self.mesh_dir / self.dataset_name
    
    @property
    def time_positions(self):
        return self.dataset_dir / 'time_positions.npy'
    
    @property
    def data_dir(self):
        return self.dataset_dir / 'data'
    
    @property
    def data_raw(self):
        return self.data_dir / 'raw'
    
    @property
    def kernels_dir(self):
        return self.mesh_dir / 'kernels'

    @property
    def simulation_dir(self):
        return self.dataset_dir / 'simulations' / self.simulation_name
    
    @property
    def trajectories_dir(self):
        return self._trajectories_dir / 'data'
    
    @property
    def seeds(self):
        return self._trajectories_dir / 'seeds.npy'
    
    @property
    def density_dir(self):
        return self._density_dir / 'data'
    
    @property
    def subsets_dir(self):
        return self._density_dir / 'subsets'
    
    @property
    def index_offsets(self):
        return self._density_dir / 'index_offsets.npy'
    
    def data_glob(self, dim):
        return str(self.data_raw / '*{}.nc'.format(dim))
    
    def extracted_dir(self, variable, variant=''):
        return self._extracted_dir / variable / variant
    
    def trajectories_branched(self, variant=''):
        return self._trajectories_dir / 'branched' / variant
    
    def density_branched(self, variant=''):
        return self._density_dir / 'branched' / variant
    
    
class Loader:
    def __init__(self, path_index):
        self.paths = path_index
        self.mesh = mesh.fetch_mesh(self.paths.mesh)(self.paths.mesh_mask)
    
    def snapshot_dataset(self, field_names, variant='train_raw', subset=None,
                         **kwargs):
        from torch.utils.data import Subset
        from core.data import datasets
        
        field_dirs = [self.paths.extracted_dir(field_name, variant)
                      for field_name in field_names]
        
        dataset = datasets.Snapshot(
            field_dirs, self.paths.density_dir, self.paths.index_offsets,
            self.paths.time_positions, self.paths.seeds, **kwargs)
    
        if subset is not None:
            subset_path = (self.paths.subsets_dir / subset).with_suffix('.npy')
            indices = np.load(subset_path, allow_pickle=True)
            dataset = Subset(dataset, indices)
    
        return dataset

    def fieldset(self, variant='raw', chunksize='auto', deferred_load=True):
        import parcels
        
        velocity_dir = self.paths.extracted_dir('velocity', variant)
        data_glob = str(velocity_dir  / '*.nc')
        filenames = {'U': {'lon': self.paths.mesh_mask,
                           'lat': self.paths.mesh_mask,
                           'data': data_glob},
                     'V': {'lon': self.paths.mesh_mask,
                           'lat': self.paths.mesh_mask,
                           'data': data_glob}}
        variables = {'U': 'U', 'V': 'V'}
        dimensions = {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'}
    
        fieldset = parcels.FieldSet.from_nemo(
            filenames, variables, dimensions, field_chunksize=chunksize,
            deferred_load=deferred_load)

        return fieldset
    
    def time_positions(self):
        return np.load(self.paths.time_positions)
    
    def seeds(self):
        return np.load(self.paths.seeds, allow_pickle=True).item()
    
    def iter_trajectories(self):
        return self.numerical_iter(self.paths.trajectories_dir)
    
    @staticmethod
    def numerical_iter(directory, ext=None):
        glob = '*'
        if ext is not None:
            glob += ext
            
        all_paths = util.path.sort_numerical(directory.glob(glob))
        for path in all_paths:
            yield xr.open_dataset(path)
        

class ArgParseFormatter(argparse.MetavarTypeHelpFormatter,
                        argparse.ArgumentDefaultsHelpFormatter):
    pass