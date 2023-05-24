import parcels

import numpy as np
import xarray as xr

from pathlib import Path


class ParticleFile(parcels.ParticleFile):
    def _delete_boundary_contacts(self, lon, lat, inplace=False):
        """
        Deletes particles once they make contact with the boundary.
    
        After contact, information is deleted for all successive time steps.
    
        Args:
            lon (observations, particles) array: longitude values.
            lat (observations, particles) array: latitude values.
            inplace (bool, optional): If True, modify in place. Defaults to False.
    
        Returns:
            lon (observations, particles) array: longitude values.
            lat (observations, particles) array: latitude values.
    
        """
        lon = np.atleast_2d(lon)
        lat = np.atleast_2d(lat)
    
        if not inplace:
            lon = lon.copy()
            lat = lat.copy()
    
        for i in range(len(lon)):
            boundary = (lon[i] >= self.lon_edge) | (lat[i] <= self.lat_edge)
            lon[i:, boundary] = np.nan
            lat[i:, boundary] = np.nan
    
        return lon, lat
    
    def convert_output(self):
        """
        Converts the npy outputs to netcdf.

        Changes to improve performance:
            Only allocates memory for the number of particles and output steps
            Prevents loading the same temporary npy files multiple times
            Saves only the variables we need and at a lower level of precision
        Behavioural changes:
            Saves particle information only if it has completed a full output
                step
            Saves ensemble ids as a coordinate

        """
        # fetch the times which are aligned with the specified output dt
        # so that we ignore data corresponding to partial observations
        out_dt = self.outputdt.total_seconds()
        out_aligned = lambda x: ((x - self.time_written[0]) % out_dt) == 0
        times_to_write = list(filter(out_aligned, self.time_written))

        # allocate memory for all output steps and particles.
        # for our purposes, we only care about writing lon and lat.
        data = np.full((2, len(times_to_write), self.maxid_written+1),
                       np.nan, np.float32)

        written_times = []
        output_index = 0
        # iterate through each temporary file saved by oceanparcels
        for tmp_file in self.file_list:
            # load in the pickle file and fetch the associated timestep
            tmp_data = np.load(tmp_file, allow_pickle=True).item()
            time = tmp_data['time'][0]
            # only write the data if:
            # 1) the time is included in the list of output aligned times
            # 2) it's the first time we are writing it (sometimes oceanparcels
            #    will duplicate information across multiple files)
            if time in times_to_write and time not in written_times:
                data[0, output_index, tmp_data['id']] = tmp_data['lon']
                data[1, output_index, tmp_data['id']] = tmp_data['lat']

                output_index += 1
                written_times.append(time)

        # convert relative times into absolute times
        time_coords = self.time_origin.fulltime(times_to_write)
        # remove particles that make contact with the boundary
        self._delete_boundary_contacts(data[0], data[1], inplace=True)
        # save data to a netcdf file
        ds = xr.Dataset(
            {
                'lon': (['obs', 'traj'], data[0]),
                'lat': (['obs', 'traj'], data[1])
            },
            coords={
                'time': ('obs', time_coords),
                'ensemble_id': ('traj', self.ensemble_ids)
            }
        )
        ds.to_netcdf(self.name)

    def separate_outputs(self):
        """
        Separates the temporary outputs by start time.

        Data for each start time is saved within its own directory.

        """
        # keep track of ids for each start time
        start_time_to_dir_id = {}
        start_time_to_file_id = {}
        start_time_to_min_pid = {}

        base_path = Path(self.tempwritedir)
        dir_id = 0
        # iterate through each temp file created by oceanparcels and split
        # its contents into directories grouped by a particle's start time
        for tmp_file in self.file_list:
            tmp_data = np.load(tmp_file, allow_pickle=True).item()
            # calculate the particle's start times and fetch its (sorted)
            # groups
            start_time = tmp_data['time'] - tmp_data['age']
            start_times, indices, counts = np.unique(
                start_time, return_index=True, return_counts=True)
            # iterate through each time group
            split_data = {}
            for t, i, count in zip(start_times, indices, counts):
                # extract the data for the current time only
                time_data = {}
                for var, data in tmp_data.items():
                    time_data[var] = data[i:i+count]
                # if it's the first time reading this time, then initialise
                # its ids
                if t not in start_time_to_dir_id:
                    # each time has its own directory id
                    start_time_to_dir_id[t] = dir_id
                    # each time has several file ids
                    start_time_to_file_id[t] = 0
                    # a time's first pid corresponds to its minimum
                    start_time_to_min_pid[t] = time_data['id'][0]

                    dir_id += 1
                else:
                    start_time_to_file_id[t] += 1
                # offset the particle ids so that its minimum pid is 0
                time_data['id'] -= start_time_to_min_pid[t]
                # the data for the current file and time is ready to save
                split_data[t] = time_data

            # for the current file, store the grouped data into their own
            # directories
            for time, data in split_data.items():
                path = base_path / str(start_time_to_dir_id[time])
                if not path.exists():
                    path.mkdir()
                path /= str(start_time_to_file_id[time])
                np.save(path, data)

    def export(self):
        """
        Overrides the export method of parcels.ParticleFile.
        
        """
        self.convert_output()