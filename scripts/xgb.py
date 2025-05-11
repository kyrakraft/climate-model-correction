import xarray as xr
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score

from xgboost import XGBRegressor
from sklearn.metrics import r2_score

BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

#load processed NetCDF files
cmip_path = os.path.join(PROCESSED_DATA_DIR, "gfdl_esm4_tas_celsius_1940_2014_us.nc")
era_path = os.path.join(PROCESSED_DATA_DIR, "era5_regridded_tas_1940_2014_us.nc")

tas_cmip = xr.open_dataarray(cmip_path)
tas_era5 = xr.open_dataarray(era_path)

"""
print("CMIP6 shape:", tas_cmip.shape)
print(tas_cmip.head)

print("ERA5 shape:", tas_era5.shape)
print(tas_era5.head)
"""

print("Flattening data...")
cmip_df = tas_cmip.to_dataframe().reset_index()
era_df = tas_era5.to_dataframe().reset_index()

#truncate time to just year and month. YYYY-MM
cmip_df["time"] = cmip_df["time"].astype(str).str[:7]
era_df["valid_time"] = era_df["valid_time"].astype(str).str[:7]
era_df = era_df.rename(columns={"valid_time": "time"})

merged_df = cmip_df.merge(era_df, on=["time", "lat", "lon"])


y = merged_df["tas_celsius_era5_regridded"]

merged_df["year"] = merged_df["time"].str[:4].astype(int)
merged_df["month"] = merged_df["time"].str[5:7].astype(int)

merged_df["month_sin"] = np.sin(2 * np.pi * merged_df["month"] / 12)
merged_df["month_cos"] = np.cos(2 * np.pi * merged_df["month"] / 12)


X = merged_df[["tas_celsius", "lat", "lon", "year", "month_sin", "month_cos"]]

"""
print("DATAFRAMES:")
print(cmip_df.head)
print(era_df.head)

print("MERGED")
print(merged_df.head)
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(n_estimators=100, max_depth=6, tree_method="hist", random_state=42)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("CV beginning...")
scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')
print("CV scores: ", scores)


#UNCOMMENT TO TRAIN FINAL MODEL
"""
print("Training model...")
model.fit(X_train, y_train) 
"""

#UNCOMMENT TO MAKE FINAL PREDICTIONS
""" 
print("Done training Making predictions...")
y_pred = model.predict(X_test) 
print("R²:", r2_score(y_test, y_pred)) 
"""