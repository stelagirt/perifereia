
import math
import numpy as np
import numpy.ma as ma
from datetime import datetime, timedelta, date
import time
import xarray as xr
import cfgrib

def magnitude(a, b):
    func = lambda x, y: np.sqrt(x ** 2 + y ** 2)
    return xr.apply_ufunc(func, a, b)

def wind_direction_deg(u, v):
    atan2v = np.vectorize(math.atan2)
    return 180 + (180 / math.pi) * atan2v(u, v)

def hourly_direction(a, b):
    return xr.apply_ufunc(wind_direction_deg, a, b)

def wind_direction_disc(azim):
    wind_dir = np.ma.copy(azim)
    # wind_dir
    # wind_dir.shape, azim[1].size
    # wind_dir

    directiondict = {1: (337.5, 22.5), # N
                     2: (22.5, 67.5), # NE
                     3: (67.5, 112.5), # E
                     4: (112.5, 157.5), # SE
                     5: (157.5, 202.5), # S
                     6: (202.5, 247.5), # SW
                     7: (247.5, 292.5), # W
                     8: (292.5, 337.5)}  # NW

    for i in directiondict:
        if directiondict[i][0] > directiondict[i][1]:
            wind_dir[(~wind_dir.mask) & ((wind_dir > directiondict[i][0]) | (wind_dir <= directiondict[i][1]))] = i
        else:
            wind_dir[(~wind_dir.mask) & (wind_dir > directiondict[i][0]) & (wind_dir <= directiondict[i][1])] = i

    return wind_dir

print("Loading grib ...")
ds = xr.load_dataset("/home/aapostolakis/Documents/preprocess/1990_2020.grib", engine='cfgrib')

print("Calculating Speed ...")
ds['speed'] = magnitude(ds['u10'], ds['v10'])

print("Calculating Angle ...")
dsangle = wind_direction_deg(ds['u10'], ds['v10'])
xrdsangle = xr.DataArray(dsangle, coords=ds['speed'].coords, dims=ds['speed'].dims)
ds['angle'] = xrdsangle

print("Calculating Direction ...")
ds['direction'] = xr.apply_ufunc(wind_direction_disc, ds['angle'])

print("Saving to NetCDF ...")
ds.to_netcdf("/home/aapostolakis/Documents/preprocess/1990_2020_speed_dir.nc")

i=1
