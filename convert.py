#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import xarray as xr
import pyproj
import subprocess
import pandas as pd
import os
import numpy as np
from metpy.calc import relative_humidity_from_specific_humidity
from metpy.units import units
import calendar
import asyncio
import uuid
import re


def gen_nc_kwargs(ds):
    # Configuration options for writing the xarray object to a netcdf file,
    #  These are the options used for all datasets in the CADS
    # Keywords that are used when writing to netcdf
    to_netcdf_kwargs  = {
        # write fields without string attributes, as h5netcdf uses string attributes that causes downstream problems
        # https://github.com/Unidata/netcdf-fortran/issues/181#issuecomment-2388560203
        "engine": "netcdf4", 
    }
    # Compression options to use when writing to netcdf, note that they are dependent on the engine
    compression_options = {
        "zlib": True,
        "complevel": 1,
        "shuffle": True,
    }

    to_netcdf_kwargs.update(
    {
        # Add the compression options to the encoding of each variable in the dataset
        "encoding": {var: compression_options for var in ds}
    })
    
    to_netcdf_kwargs['encoding']['time']= {'dtype':'int64'}

    # to_netcdf_kwargs['encoding']['datetime']={}
    # to_netcdf_kwargs['encoding']['datetime']['units'] = 'seconds since 1970-01-01'

    return to_netcdf_kwargs


# In[3]:


# Calculate the rotation angle at each grid point
def calculate_rotation_angle(lon, lat, pole_lon, pole_lat):
    """Calculate the grid rotation angle."""
    lon_diff = lon - pole_lon
    sin_phi = np.sin(np.deg2rad(lat)) * np.sin(np.deg2rad(pole_lat)) + np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(pole_lat)) * np.cos(np.deg2rad(lon_diff))
    cos_phi = np.sqrt(1 - sin_phi**2)
    sin_alpha = np.cos(np.deg2rad(lat)) * np.sin(np.deg2rad(lon_diff)) / cos_phi
    cos_alpha = (np.sin(np.deg2rad(lat)) - sin_phi * np.sin(np.deg2rad(pole_lat))) / (cos_phi * np.cos(np.deg2rad(pole_lat)))
    return np.arctan2(sin_alpha, cos_alpha)

# Rotate wind components
def rotate_wind(U, V, angle):
    """Rotate wind components."""
    U_rotated = U * np.cos(angle) - V * np.sin(angle)
    V_rotated = U * np.sin(angle) + V * np.cos(angle)
    return U_rotated, V_rotated

def correct_wind_rotation(ds, pole_lon, pole_lat):
    # unit conversions
    conversion_kts_ms = 0.514444

    rotation_angle = calculate_rotation_angle(ds.longitude, ds.latitude, pole_lon, pole_lat)

    # This is 1 single time step, so time=0 just selects the single time 
    U_rotated, V_rotated = rotate_wind(ds.isel(time=0).UU, ds.isel(time=0).VV, rotation_angle)
    U_rotated = U_rotated * conversion_kts_ms
    V_rotated = V_rotated * conversion_kts_ms

    dirog = np.arctan2(ds.isel(time=0).UU, ds.isel(time=0).VV) * (180 / np.pi)
    dirog = (dirog + 360) % 360

    # met wind from sign convention
    # https://mst.nerc.ac.uk/wind_vect_convs.html
    wind_from_direction = np.arctan2(-U_rotated, -V_rotated) * (180 / np.pi)
    wind_from_direction = (wind_from_direction + 360) % 360

    wind_speed = xr.Dataset({"wind_speed":np.sqrt(U_rotated**2+V_rotated**2)}).expand_dims(dim="time").assign_coords(time=pd.to_datetime(ds.time.values))

    U_rotated = xr.Dataset({"U": U_rotated}).expand_dims(dim="time").assign_coords(time=pd.to_datetime(ds.time.values))
    V_rotated = xr.Dataset({"V": V_rotated}).expand_dims(dim="time").assign_coords(time=pd.to_datetime(ds.time.values))

    # wind_from_direction_og = xr.Dataset({"wind_from_direction_og":dirog}).expand_dims(dim="time").assign_coords(time=pd.to_datetime(time.values))
    wind_from_direction = xr.Dataset({"wind_from_direction":wind_from_direction}).expand_dims(dim="time").assign_coords(time=pd.to_datetime(ds.time.values))

    d = xr.merge([ds, U_rotated, V_rotated, wind_speed, wind_from_direction])
    d = d.drop_vars(['UU','VV'])
    return d

def specific_to_rh(ds):
    dequantified_data = xr.DataArray(
        relative_humidity_from_specific_humidity(ds.P0 * units.hPa, ds.TT * units.degC, ds.HU * units('kg/kg')).data.magnitude,
        dims=ds.dims,
        coords=ds.coords
    )
    ds['RH'] = dequantified_data
    return ds
  
def set_CF_standard_names(ds):
    # Surfaces which are defined using a coordinate value (e.g. height of 1.5 m) are indicated by a single-valued coordinate variable, not by the standard name.
    # http://cfconventions.org/Data/cf-standard-names/docs/guidelines.html
    # https://confluence.ecmwf.int/display/CKB/ERA5-Land%3A+data+documentation#ERA5Land:datadocumentation-parameterlistingParameterlistings
    cf_standard_names = {
        'FI': ('surface_downwelling_longwave_flux', 'Incoming longwave radiation at the surface', 'W/m2'),
        'FB': ('surface_downwelling_shortwave_flux', 'Incoming shortwave radiation at the surface', 'W/m2'),
        'FSD': ('surface_direct_downwelling_shortwave_flux_in_air', 'Incoming direct shortwave radiation at the surface', 'W/m2'),
        'FSF': ('surface_diffuse_downwelling_shortwave_flux_in_air', 'Incoming diffuse shortwave radiation at the surface', 'W/m2'),
        'PR': ('precipitation_amount', 'Precipitation ammount in timestep', 'm'),
        'RN': ('liquid_precipitation_amount', 'Liquid precipitation', 'm'), # ???
        'P0': ('surface_air_pressure', 'Pressure at the surface', 'mb'),
        'HU': ('specific_humidity', 'Specific humidity at surface', 'kg/kg'),
        'TT': ('air_temperature', 'Air temperature', 'C'),
        'GZ': ('geopotential_height', 'Geopotential height', 'm'),
        'U': ('eastward_wind', 'Surface U wind component', 'm s**-1'),
        'V': ('northward_wind', 'Surface V wind component', 'm s**-1'),
        'wind_speed': ('wind_speed', 'Surface wind speed', 'm s**-1'),
        'wind_from_direction': ('wind_from_direction', 'Surface wind direction', 'degree'),
        'RH': ("relative_humidity", "surface relative humidity","1"),
        "latitude": ("latitude", "latitude", "degrees_north"),
        "longitude": ("longitude", "longitude", "degrees_east")
    }
    

    for k in  ds.data_vars:
        if k == 'crs':
            continue 
        attrs = ds[k].attrs

        attrs['standard_name'] = cf_standard_names[k][0]
        attrs['long_name'] = cf_standard_names[k][1]
        attrs['units'] = cf_standard_names[k][2]
        ds[k].attrs = attrs

    ds['time'].attrs['standard_name'] = "time"
    ds['time'].attrs['long_name'] = "time"
    ds['time'].attrs['delta_t'] = 3600 #s
    ds['time'].attrs['delta_t_units'] = 's' 
       
    return ds

def set_CF_encoding(ds):
    # use the CF standard name for time and ensure the order follows CF recommendation of time x Y x X
    # https://cfconventions.org/Data/cf-conventions/cf-conventions-1.11/cf-conventions.html#dimensions
    # ds = ds.rename({'valid_time':'time'})  

    for v in ds:
        ds[v].encoding["coordinates"] = "time latitude longitude"
        ds[v].attrs.pop('grid_mapping', None) # we are no longer in rotated pole crs

    ds = ds.set_coords(["time", "latitude", "longitude"])
    
    return ds

def deaccum_precip(ds):
    ds['PR'] = ds.PR.diff(dim="time", n=1, label='lower')
    return ds

# Some of the fst files have multiple levels and might look like
# * level1 (level1) float32 1.0
# * level2 (level2) float32 0.995 1.0
# while others might only have levels or levels1
# either way, we want the one with the 0.995 and 1.0 levels, and to drop the rest
def _pick_level_coord(ds, target_vals=(0.995, 1.0), prefix="level"):
    """Return the name of the coord/dim that contains all target_vals, else None."""
    targets = list(target_vals)

    # Consider coord names: level, level1, level2, ...
    candidates = [c for c in ds.coords if c == prefix or re.match(rf"^{re.escape(prefix)}\d+$", c)]
    # Also consider dims named like that (in case it's a dim but not a coord)
    candidates += [d for d in ds.dims if d not in candidates and (d == prefix or re.match(rf"^{re.escape(prefix)}\d+$", d))]

    for name in candidates:
        if name in ds.coords:
            vals = ds.coords[name].values
        else:
            # dim without coord: can't test values
            continue

        # flatten + numeric compare with tolerance
        vals = np.asarray(vals).ravel()
        if all(np.isclose(vals, t, rtol=0, atol=1e-6).any() for t in targets):
            return name

    return None

def _drop_other_level_coords(ds, keep_name, prefix="level"):
    """Drop all coords named level*, except keep_name."""
    drop = [c for c in ds.coords if (c == prefix or c.startswith(prefix)) and c != keep_name]
    return ds.drop_vars(drop, errors="ignore")


def extract_hour(year, month, day, hour):

        # use this uuid to prevent any name collisions when we cross the 00 on the 25th hour with the next day's data
        # # each day will have a unique uuid        
        uid = uuid.uuid4()

        fstname = f'/fs/site6/eccc/mrd/rpnenv/smsh001/arcsfc/{year}/{str(month).zfill(2)}/{str(day).zfill(2)}/lam/nat.eta/{year}{str(month).zfill(2)}{str(day).zfill(2)}00_{str(hour).zfill(3)}'
        print(fstname)
        fst = xr.open_mfdataset(fstname)

        # FI:  Incoming longwave radiation at the surface (W/m2)
        # FB:  Incoming shortwave radiation (W/m2)
        # FSD: Incoming direct shortwave radiation at the surface (W/m2)
        # FSF: Incoming diffuse shortwave radiation at the surface (W/m2)
        # PR:  Total precipitation (m)
        # RN:  Liquid precipitation (m)
        # P0:  Pressure at the surface (mb)
        # HU:  Specific humidity (kg/kg)
        # TT:  Air temp (C)
        # GZ:  Geopotential height (dam)
        # UU:  Wind speed (kts)
        # VV:  Wind speed (kts)
        # this list needs at least 1 surface variable for the coord situation to make sense in the merge
        variables = ['rotated_pole', 'GZ', 'TT', 'FI', 'FB', 'FSD', 'FSF', 'PR', 'RN', 'P0', 'HU']
        variables_wind = ['UU', 'VV']

        level_name = _pick_level_coord(fst, target_vals=(0.995, 1.0), prefix="level")
        if level_name is None:
            raise ValueError("Could not find a level* coordinate containing both 0.995 and 1.0")

        # Take wind at 40m (0.995) but everything else at surface 1.5 m (level=1)
        fst = xr.merge([
                fst[variables_wind].sel({level_name: 0.995}),
                fst[variables].sel({level_name: 1.0})
            ], compat='override')

        # drop the remaining levels we don't need
        fst = _drop_other_level_coords(fst, keep_name=level_name, prefix="level")

        variables.extend(variables_wind)

        time = 1
        if 'time1' in fst.coords:
            # later fst versions use time1 for hour > 1
            fst = fst.rename({"time1":"time"})
        
        time = fst.time


        date = time.dt.strftime('%Y%m%dT%H00').values[0]
        print(date)

        # if not current_day:
        #     current_day = time.dt.strftime('%Y%m%d').values[0]

        all = []
        files_to_remove = []
        
        for v in variables:
            if v == 'rotated_pole':
                continue   

            fname = f'/home/chm003/project/hrdps/{date}-{v}-{uid}.nc'
            fname_eps4326 = f'/home/chm003/project/hrdps/{date}-{v}-{uid}-eps4326.nc'
            fst[v].to_netcdf(fname)

            subprocess.run(['gdalwarp', 
                            '-t_srs', f'EPSG:4326', 
                            f"NETCDF:{fname}:{v}", fname_eps4326])
            
            os.remove(fname)

            ds = xr.open_mfdataset(fname_eps4326)
            ds = ds.rename_vars({'Band1':v})
            ds = ds.rename({"lat":"latitude","lon":"longitude"})
            ds = ds.expand_dims(dim="time").assign_coords(time=pd.to_datetime(time.values))

            all.append(ds)

            # remove this file later once we've finished the delayed append
            files_to_remove.append(fname_eps4326)

        aligned = xr.align(*all, join="override")
        ds = xr.merge(aligned)

        if 'UU' in ds.data_vars and 'VV' in ds.data_vars:
            # correct the U and V components for the rotateion
            pole_lon = fst.rotated_pole.attrs['grid_north_pole_longitude']
            pole_lat = fst.rotated_pole.attrs['grid_north_pole_latitude']
            ds = correct_wind_rotation(ds, pole_lon, pole_lat)

        if 'P0' in ds.data_vars:
            ds = specific_to_rh(ds) #computes RH

        if 'GZ' in ds.data_vars:
            ds['GZ'] = ds.GZ*10.0 # dam -> m

        ds = set_CF_standard_names(ds) # sets all the CF names into attrs
        ds = set_CF_encoding(ds) # sets CF coord encoding
        ds.attrs = {"Conventions": "CF-1.7"}
        
        # to_netcdf_kwargs = gen_nc_kwargs(ds)
        # ds contains delayed objects, so the temp files need to remain until here

        # fname = f"/home/chm003/project/hrdps/{date}-{uid}.nc"
        # ds.to_netcdf(fname, **to_netcdf_kwargs)
        
        # clean up the tmp files AFTER we have written the nc file out above
        # print('Cleaning up')
        # for f in files_to_remove:
        #     # print(f)
        #     os.remove(f)

        return ds, files_to_remove



def extract_day(year, month, day):

    all_day = []
    files_to_remove = []
    # holds the datetime string format
    current_day = f'{str(year)}{str(month).zfill(2)}{str(day).zfill(2)}'
    
    for hour in range(0,25): # only want until 11pm, however we need the next ts to produce a deaccum precip
        d,frm = extract_hour(year, month, day, hour)
        all_day.append(d)

        files_to_remove.extend(frm)

    # print(all_day)
    ds = xr.concat(all_day, dim="time", join='override')
    # ds = xr.open_mfdataset(all_day, combine='nested', concat_dim='time', join='override')

    if 'PR' in ds.data_vars:
        ds = deaccum_precip(ds) # deaccumulate the precip into hourly sums

    # ensure the CF remains correctly nammed and CF compliant. 
    ds = set_CF_standard_names(ds) # sets all the CF names into attrs
    ds = set_CF_encoding(ds) # sets CF coord encoding
    ds.attrs = {"Conventions": "CF-1.7"}

    ds = ds.isel(time=slice(0,24)) #chop off the unneeded 25th timesteps, on keep 00-23
    
    # sanity check
    # print(ds.sel(latitude=52,longitude=-132,method="nearest").TT.values)

    to_netcdf_kwargs = gen_nc_kwargs(ds)
    ds.to_netcdf(f"/home/chm003/project/hrdps/all/{current_day}.nc", **to_netcdf_kwargs)

    for f in files_to_remove:
        os.remove(f)    


# %%

