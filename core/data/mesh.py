import numpy as np
import xarray as xr

from core import util


class _Ocean:
    @property
    def grid_coordinates(self):
        return self.lon, self.lat
    
    @property
    def coastline(self, padding=0, return_indices=False):
        coastline = util.grid.mask_to_contour(self.mask, iterations=padding+1)
        ys, xs = np.where(coastline)
    
        if return_indices:
            return xs, ys
        else:
            return self.lon[xs], self.lat[ys]


class Glazur64(_Ocean):
    def __init__(self, mesh_mask_path):
        mesh_mask = xr.open_dataset(mesh_mask_path)
        
        self.mask_t = mesh_mask.tmaskutil.data[0].astype(bool)
        self.mask_u = mesh_mask.umaskutil.data[0].astype(bool)
        self.mask_v = mesh_mask.vmaskutil.data[0].astype(bool)
        self.mask = self.mask_t | self.mask_u | self.mask_v
        
        self.lon = mesh_mask.nav_lon.data[0]
        self.lat = mesh_mask.nav_lat.data[:,0]
        
        self.lon_u = mesh_mask.glamu.data[0,0]
        self.lon_v = mesh_mask.glamv.data[0,0]
        
        self.lat_u = mesh_mask.gphiu.data[0,:,0]
        self.lat_v = mesh_mask.gphiv.data[0,:,0]
        
    @property
    def boundary_coordinates(self):
        lon_edge = self.lon[-2]
        lat_edge = self.lat[1]
    
        return lon_edge, lat_edge
    
    
_meshes = {'glazur64': Glazur64}


def fetch_mesh(mesh):
    return _meshes[mesh]