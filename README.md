# Climate Model Correction

## Overview

This project aims to correct biases in climate model output using machine learning (2 models used: XGBoost and a convolutional neural network). Specifically, I compare historical surface air temperature data (2-meter temperature) from the GFDL-ESM4 global climate model (CMIP6 archive) to reanalysis data from ERA5, and train models to learn the mapping from biased model output to reference-quality observations. 

## Data acquisition:

Data spans the United States region from 1940 to 2014, covering monthly temperature values.

Data sources:
- GFDL-ESM4 (an ESM developed by NOAA) temperature
  - Data loaded via the Pangeo Cloud CMIP6 (Coupled Model Intercomparison Project) catalog
- ERA5 reanalysis temperature data (produced by the European Centre for Medium-Range Weather Forecasts, or ECMWF)
  - High-resolution global climate reanalysis (combines model simulations with observational data)
  - Downloaded using the Copernicus CDS API


## Preprocessing:

The data is preprocessed from NetCDF files and handled using xarray and numpy.

- Inputs and targets are aligned by time, latitude, and longitude, and reshaped into 4D tensors for CNN training.
- Datasets are subset to the same spatial/temporal range and converted from Kelvin to Celsius.
- ERA5 is regridded to match the coarser CMIP6 resolution.

## Modeling:

There are two models implemented:

### 1. A tabular XGBoost model trained on flattened grid data.

Features include:
  - CMIP6-predicted surface air temperature (tas)
  - Latitude and longitude
  - Year and month (with month encoded using sine and cosine for cyclicality)

### 2. A convolutional neural network that directly learns spatial corrections from gridded input.

The CNN:
  - Takes as input 2D spatial grids (25x56 resolution) with 3 channels: [CMIP6 tas, sin(month), cos(month)]
  - Uses a multi-scale architecture with downsampling and upsampling layers to capture both local and regional patterns
  - Includes batch normalization and residual-style fusion to stabilize training and preserve spatial detail
  - Uses the AdamW optimizer with decoupled weight decay for regularization
  - Employs learning rate scheduling (step decay or cosine annealing) to improve convergence
  - Is trained with Smooth L1 loss (Huber loss) and evaluated using R²

## Evaluation:

XGBoost R²: 0.97
CNN R²: 0.96
