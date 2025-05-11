#Overview

This project aims to correct systematic biases in climate model output using machine learning. Specifically, I compare historical surface air temperature data (2-meter temperature) from the GFDL-ESM4 global climate model (CMIP6 archive) to reanalysis data from ERA5, and train models to learn the mapping from biased model output to reference-quality observations.

Two types of ML models are used:
1. A tree-based XGBoost regressor trained on flattened data (time × features), which treats each spatial location independently.
2. A Convolutional Neural Network (CNN) that preserves spatial structure and learns local interactions across latitude and longitude.

Data spans the United States region from 1940 to 2014, covering monthly temperature values.

##Data acquisition:

GFDL-ESM4 temperature data loaded via the Pangeo Cloud CMIP6 catalog.

ERA5 reanalysis temperature data downloaded using the Copernicus CDS API.

##Preprocessing:

Datasets are subset to the same spatial/temporal range and converted from Kelvin to Celsius.

ERA5 is regridded to match the coarser CMIP6 resolution.

Modeling:

XGBoost is trained on flattened input and output grids.

CNN is trained on 3D tensors (time × lat × lon) to preserve spatial relationships.

Evaluation:

Model performance is assessed using standard regression metrics such as RMSE or R².

This project serves as a prototype for ML-based post-processing of climate simulations and lays the groundwork for future integration of spatial models and uncertainty quantification.
