import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import Dataset

from core import util


class Snapshot(Dataset):
    def __init__(self, field_dir, density_map_dir, index_offsets_path,
                 time_positions_path, seeds_path, next_field=True):
        self._next_field = next_field
        
        # data directories
        if type(field_dir) is not list:
            field_dir = [field_dir]
        self._field_dirs = list(map(Path, field_dir))
        self._density_map_dir = Path(density_map_dir)

        # index offsets for unravelling from a 1D to 3D index
        index_offsets = np.load(index_offsets_path, allow_pickle=True).item()
        self._time_offsets = index_offsets['time']
        self._ensemble_offsets = index_offsets['ensemble']
        
        # load fields into memory
        self._fields = self._fetch_fields()

        # define mapping between density index -> field index
        time_positions = np.load(time_positions_path)
        seed_times = list(np.load(seeds_path, allow_pickle=True).item().keys())
        self._density_to_field_index = np.searchsorted(
            time_positions, seed_times)


    def __len__(self):
        return self._time_offsets[-1]

    def __getitem__(self, index):
        # fetch 3D index
        density_index, ensemble_index, obs_index = self.unravel_index(index)
        # fetch input field, input map, and label map
        input_field = self.load_input_field(density_index, obs_index)
        input_map, label_map = self.load_density_map_pair(
            density_index, ensemble_index, obs_index)
        # stack the input field and map
        if input_field.ndim == 2:
            input_field = input_field[None]
        input_data = np.concatenate((input_field, input_map[None]))
        
        return input_data, label_map

    def _fetch_fields(self):
        fields = {}
        
        for path in self._field_dirs[0].glob('*[0-9].npy'):
            field_index = int(path.stem)
            fields[field_index] = self._load_field_from_file(field_index)
        
        return fields

    def _load_field_from_file(self, field_index):
        fields = []
        for field_dir in self._field_dirs:
            path = (field_dir / str(field_index)).with_suffix('.npy')
            field = np.load(path)
            
            if field.ndim == 2:
                field = field[None]
            fields.append(field)
        
        return np.concatenate(fields)

    def _load_field_from_index(self, density_index, obs_index):
        field_index = self._density_to_field_index[density_index] + obs_index
        
        return self._fields[field_index]
    
    def load_input_field(self, density_index, obs_index):
        input_field = self._load_field_from_index(density_index, obs_index)
        if self._next_field:
            input_field_2 = self._load_field_from_index(
                density_index, obs_index+1)
            input_field = np.concatenate((input_field, input_field_2))
            
        return input_field
    
    def load_density_map_pair(self, density_index, ensemble_index, obs_index):
        start_density_index = str(density_index)
        input_density_path = (
            self._density_map_dir / start_density_index).with_suffix('.nc')
        
        density_maps = xr.open_dataset(input_density_path).density_map
        
        return density_maps.isel(
            ensemble_id=ensemble_index, obs=[obs_index, obs_index+1]).data
    
    def unravel_index(self, index):
        if index >= len(self):
            raise IndexError('Index out of bounds')
            
        density_index = (index < self._time_offsets).argmax()
        ensemble_index = (
            index < self._ensemble_offsets[density_index]).argmax() - 1
        obs_index = index - self._ensemble_offsets[
            density_index][ensemble_index]

        return density_index, ensemble_index, obs_index
    
    def random_split(self, proportions):
        assert np.isclose(sum(proportions), 1)
        
        # map 1D indices to 3D indices
        indices_3d = pd.DataFrame(
            [self.unravel_index(i) for i in range(len(self))],
            columns=['time', 'ensemble', 'obs'])
        
        # group indices by ensemble
        ensembles = indices_3d.groupby(['time', 'ensemble'])
        ensemble_indices = ensembles.ngroup()
        n_ensembles = ensembles.ngroups
        
        # randomly assign ensembles to different sets
        ensemble_sets = util.random.split(n_ensembles, proportions)
        
        # lookup the sample indices belonging to each set
        set_indices = []
        for ensemble_set in ensemble_sets:
            within_set = np.isin(ensemble_indices, ensemble_set)
            set_indices.append(np.where(within_set)[0])
        
        return set_indices