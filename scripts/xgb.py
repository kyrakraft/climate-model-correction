import xarray as xr
import os
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

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


# Flatten spatial data for ML
X = tas_cmip.values.reshape(900, -1)  # shape: (900, 1400)
y = tas_era5.values.reshape(900, -1)       # same shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(tree_method="hist", n_estimators=100, max_depth=4)

print("Training model...")
model.fit(X_train, y_train)

print("Done training Making predictions...")
y_pred = model.predict(X_test)
print("R²:", r2_score(y_test, y_pred))