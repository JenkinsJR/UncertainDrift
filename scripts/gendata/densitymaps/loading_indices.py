import numpy as np
import xarray as xr
from tqdm import tqdm


def save_index_offsets(density_maps_dir, save_path):
    all_paths = list(density_maps_dir.glob('*.nc'))
    all_paths = sorted(all_paths, key=lambda x: int(x.stem))

    # initialise offsets with an extra starting 0 for convenient computation
    time_offsets = np.zeros(len(all_paths)+1, dtype='int32')
    ensemble_offsets = [[0]]
    
    # for each temporal position
    for i, path in enumerate(tqdm(all_paths)):
        # find the density maps that are not empty
        file = xr.open_dataset(path)
        sums = file.density_map.sum(dim=('lat', 'lon'))
        non_empty = sums > 0

        # we set the first offset as the last offset from the previous time so
        # that computation of the relative obs is simpler
        offsets = np.zeros(non_empty.ensemble_id.size + 1, dtype='int32')
        offsets[0] = time_offsets[i]

        # for each ensemble, retrieve the values corresponding to the last
        # non-empty observation. This corresponds to the last target obs
        last_obs = offsets[1:]
        last_obs[:] = non_empty.argmin(dim='obs').data
        last_obs[last_obs==0] = non_empty.obs.size - 1

        # cumulatively sum the last observation indices to compute the offsets
        offsets.cumsum(out=offsets)

        ensemble_offsets.append(offsets)
        time_offsets[i+1] = offsets[-1]

    # remove the additional starting 0s
    time_offsets = time_offsets[1:]
    ensemble_offsets = ensemble_offsets[1:]

    # save dict to file
    offsets_map = {'time': time_offsets, 'ensemble': ensemble_offsets}
    np.save(save_path, offsets_map)


def save_dataset_groups(dataset, train_val_test_split, seed, save_path):
    assert len(train_val_test_split) == 3
    assert np.isclose(sum(train_val_test_split), 1)
    
    save_path.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)
    
    train, val, test = dataset.random_split(train_val_test_split)
    for indices, name in zip([train, val, test], ['train', 'val', 'test']):
        path = save_path / name
        np.save(path, indices)