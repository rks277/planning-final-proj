# Standard import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pickle
from PIL import Image

from pathlib import Path
import os
import sys

import map_utils

N_row, N_col = 250, 500
lat_step, lon_step = 0.08, 0.08
weight_factor = 100000/ 27603.0
alpha = 0.2
import os

current_dir = os.getcwd()
# Construct the path to the data file
data_folder = "data"
result_folder = "results"

season = "summer"
start_day = 28 if season == "winter" else 196
mapL = map_utils.LagrangianMap(f"{data_folder}/share_data_2002_{season}.nc", weight_factor=weight_factor, start_day=start_day)

def visualize_particle_density(mapL, day_offset=0):
    """
    Generate density map from plastic particle positions
    
    Args:
        mapL: LagrangianMap object
    """
    # Get particle positions for the specified day
    lats = mapL.Lat_matrix[:, day_offset]
    lons = mapL.Lon_matrix[:, day_offset]
    
    # Filter out masked/invalid values
    if hasattr(lats, 'mask'):
        valid_mask = ~lats.mask & ~lons.mask
        lats = lats[valid_mask].data
        lons = lons[valid_mask].data
    
    # Create density map
    density_map = np.zeros((N_row, N_col))
    
    for lat, lon in zip(lats, lons):
        # Convert lat/lon to grid indices
        i = int((39.92 - lat) / lat_step)
        j = int((lon - (-160)) / lon_step)
        
        # Check bounds
        if 0 <= i < N_row and 0 <= j < N_col:
            density_map[i, j] += 1
    
    # Create the plot
    print("Generating density heatmap...")
    # Normalize the density map from 0-1
    max_density = np.max(density_map)
    if max_density > 0:
        density_map = density_map / max_density
    density_img_array = (density_map * 255).astype(np.uint8)
    density_img = Image.fromarray(density_img_array, mode='L')
    density_img.save(f"{season}_2002_day{day_offset}_density.png")

# Example usage: visualize day 0 (first day of the dataset)
print(f"\nPlastic particle data info:")
print(f"  Total particles tracked: {mapL.N_id}")
print(f"  Data shape: {mapL.Lat_matrix.shape} (particles x days)")
print(f"  Grid size: {N_row} x {N_col}")
print(f"  Season: {season}")

ocean_array = np.zeros((N_row, N_col, 3), dtype=np.uint8)
ocean_array[:] = (0, 100, 255) # RGB: Blueish
ocean_img = Image.fromarray(ocean_array, mode='RGB')
ocean_img.save("ocean_real.png")


# Generate particle density map
visualize_particle_density(mapL)





