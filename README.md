# citypy
A CLI tool for analysing urban form using open source geospatial data.

### How to install
1. Clone the repository
2. Make sure you have and active Python environment with [Poetry](https://python-poetry.org) installed.
3. Run `poetry install`
4. `citypy` is now installed in this environment as a CLI tool

### How to use
1. Create a `.secrets.toml` file in the `config` folder, and add your [OpenTopography API Key](https://opentopography.org/blog/introducing-api-keys-access-opentopography-global-datasets)
```toml
[opentopography]
api_key = "<your_key>
```
`citypy` will not work without this key
2. Download a city
```bash
citypy download "<city name>" "<country name>"
```
There is an option to provide a WGS84 bounding box as csv string with the `--bbox` option
Now a `"<City_Name>_<COUNTRY NAME>.gpkg"` GeoPackage will be created in the current directory
3. Generate all necessary layers and metrics
```bash
citypy process "<City_Name>_<COUNTRY NAME>.gpkg"
```
4. Contextualize the metrics for each element's neighbourhood
```bash
citypy contextualize "<City_Name>_<COUNTRY NAME>.gpkg" --regular --buildings
```
5. Apply clusters to the roads layer
```bash
citypy clusters apply --file "<City_Name>_<COUNTRY NAME>.gpkg" -l road_edges --col "<cluster_column_name>" --clusters citypy_gmm13_road_clusters.gmm --column-filter="type_category=street"
```
6. Apply building classes to the buildings layer
```bash
citypy process "<City_Name>_<COUNTRY NAME>.gpkg" -s building_class
```
7. Encode the resulting city to add aggregated grids to the `gpkg` and get local typology templates
```bash
citypy encode "<City_Name>_<COUNTRY NAME>.gpkg" --road-cluster-column "<cluster_column_name>" --building-class-column building_class -o ./Output_Folder/
```

The GeoPackage can be opened with the open source GIS software QGIS.

#### OvertureMaps support (Experimental)
Overture Maps has much better building footprint coverage than OpenStreetMap, at the cost of quality.
You can use Overture footprints by enabling the following options in `config/settings.toml`
```toml
[overturemaps]
use_overture_for_buildings = true # set to true

[local_overture]
use_local_parquet = true # set to true
release_folder = "~/overture_db/2024-07-22.0/" # set to downloaded overture data
```
Warning, using Overture without first downloading the entire Overture world wide dataset (hundreds of GB) can be very slow and fail because of request timeout. If you want to use Overture data, it's best to download the entire dataset.