import os

import geopandas as gpd
import pyogrio

d = "/Users/ole/backup3"
e = "/Users/ole/backup3/extracted"

if not os.path.exists(e):
    os.makedirs(e)

for file in os.listdir(d):
    print(file)
    if not file.endswith(".gpkg"):
        continue

    print(f"Processing {file}")

    try:
        grid = gpd.read_file(os.path.join(d, file), layer="grid")
    except pyogrio.errors.DataLayerError as err:
        print(f"Error reading {file}: {err}")
        continue

    grid.to_file(os.path.join(e, file.split(".")[0] + "_GRID.gpkg"), driver="GPKG", layer="grid")