import xarray as xr
import os
import numpy as np

# Set base paths
BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

# Load processed NetCDF files
era_path = os.path.join(PROCESSED_DATA_DIR, "era5_regridded_tas_1940_2014_us.nc")
cmip_path = os.path.join(PROCESSED_DATA_DIR, "gfdl_esm4_tas_celsius_1940_2014_us.nc")

tas_era5 = xr.open_dataarray(era_path)
tas_cmip = xr.open_dataarray(cmip_path)

print("ERA5 shape:", tas_era5.shape)
print("CMIP6 shape:", tas_cmip.shape)
