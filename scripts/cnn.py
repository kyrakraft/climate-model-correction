import xarray as xr
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

#load processed NetCDF files
era_path = os.path.join(PROCESSED_DATA_DIR, "era5_regridded_tas_1940_2014_us.nc")
cmip_path = os.path.join(PROCESSED_DATA_DIR, "gfdl_esm4_tas_celsius_1940_2014_us.nc")

tas_era5 = xr.open_dataarray(era_path)
tas_cmip = xr.open_dataarray(cmip_path)


print("CMIP6 shape:", tas_cmip.shape)
print("lat:", tas_cmip.coords["lat"].values.shape)
print("lon:", tas_cmip.coords["lon"].values.shape)
print("time:", tas_cmip.coords["time"].values.shape)

print("ERA5 shape:", tas_era5.shape)


#extract months from CMIP time (cftime or datetime64)
months = [pd.to_datetime(str(t)).month for t in tas_cmip.time.values]
months = np.array(months)

month_sin = np.sin(2 * np.pi * months / 12)  #shape (900,)
month_cos = np.cos(2 * np.pi * months / 12)

n_time, n_lat, n_lon = tas_cmip.shape

#expand to match spatial dims
month_sin_grid = np.repeat(month_sin[:, np.newaxis, np.newaxis], n_lat, axis=1)
month_sin_grid = np.repeat(month_sin_grid, n_lon, axis=2)

month_cos_grid = np.repeat(month_cos[:, np.newaxis, np.newaxis], n_lat, axis=1)
month_cos_grid = np.repeat(month_cos_grid, n_lon, axis=2)

tas_input = tas_cmip.values.astype(np.float32)
X = np.stack([tas_input, month_sin_grid, month_cos_grid], axis=-1)  #shape (900, 25, 56, 3)
y = tas_era5.values.astype(np.float32)                              #shape (900, 25, 56)

print("X shape:", X.shape)
print("y shape:", y.shape)

#setting up pytorch dataset and dataloader
class ClimateDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()   #shape (N, 25, 56, 3)
        self.y = torch.from_numpy(y).float()   #shape (N, 25, 56)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx].permute(2, 0, 1), self.y[idx]  # (channels, lat, lon), (lat, lon)


#cnn
""" 
class CNNCorrectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1)  #output: 1 value per (lat, lon)
        )

    def forward(self, x):
        out = self.net(x)  #shape (B, 1, H, W)
        return out.squeeze(1)  #shape (B, H, W)

class CNNCorrectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.out = nn.Conv2d(64, 1, kernel_size=1)  # output per-pixel prediction

    def forward(self, x):
        x = self.block1(x)
        residual = x
        x = self.block2(x)
        x = self.block3(x)
        x = x + residual  #simple residual connection
        x = self.out(x)
        return x.squeeze(1)  #remove channel dimension. (B, H, W)
"""

class MultiScaleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.down = nn.Sequential(
            nn.MaxPool2d(2),  #halves resolution. #downsample and deeper features
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  #upsample. back to original size
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
)
        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)     # (B, 64, 25, 56)
        x2 = self.down(x1)     # (B, 128, 12, 28)
        x3 = self.up(x2)       # (B, 64, 25, 56)

        # If needed, pad x3 to match x1
        if x3.shape[-2:] != x1.shape[-2:]:
            diff_h = x1.shape[-2] - x3.shape[-2]
            diff_w = x1.shape[-1] - x3.shape[-1]
            x3 = F.pad(x3, (0, diff_w, 0, diff_h))  # pad (left, right, top, bottom)

        x_final = x1 + x3      # residual-style fusion
        out = self.out(x_final)
        return out.squeeze(1)  # (B, 25, 56)

"""
#testing that dataset returns the expected shapes
dataset = ClimateDataset(X, y)
x_sample, y_sample = dataset[0] # Grab one sample
print("x_sample shape:", x_sample.shape)  # should be (3, 25, 56)
print("y_sample shape:", y_sample.shape)  # should be (25, 56)

#testing that model runs a forward pass without error and that data types match
model = CNNCorrectionModel()
# Run one forward pass
with torch.no_grad():
    x_sample = x_sample.unsqueeze(0)  # add batch dim → shape (1, 3, 25, 56)
    y_pred = model(x_sample)
    print("y_pred shape:", y_pred.shape)  # should be (1, 25, 56)
"""


#hyperparameters
BATCH_SIZE = 15 #previously 32, 15 (best), 10 
EPOCHS = 150
LR = 1e-2 #previously 1e-3, then 1e-2 (best; everything else original vals), 1e-1
WEIGHT_DECAY=1e-2 #previously 1e-4, 1e-3
ETA_MIN=1e-4 #previously 1e-5, 1e-4 (best)

#create dataset and split
dataset = ClimateDataset(X, y)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

#initialize model, loss, optimizer
#model = CNNCorrectionModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiScaleCNN().to(device)
model = model.to(device)

#criterion = nn.MSELoss()
criterion = nn.SmoothL1Loss()

#optimizer = torch.optim.Adam(model.parameters(), lr=LR)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,  # total epochs
    eta_min=ETA_MIN  # minimum LR at end
)



def r2_score_torch(y_true, y_pred):
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * X_batch.size(0)

    model.eval()
    val_loss = 0.0
    r2_total = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            val_loss += loss.item() * X_batch.size(0)

            r2_total += r2_score_torch(y_batch, preds) * X_batch.size(0)

    train_loss /= len(train_ds)
    val_loss /= len(val_ds)
    r2_val = r2_total / len(val_ds)

    scheduler.step()
    current_lr = scheduler.optimizer.param_groups[0]["lr"]
    print(f"Epoch {epoch+1}/{EPOCHS} — Train Loss: {train_loss:.4f} — Val Loss: {val_loss:.4f} — Val R²: {r2_val:.4f} — LR: {current_lr:.6f}")