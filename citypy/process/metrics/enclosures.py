import geopandas as gpd
import momepy
import pandas as pd

from citypy.process.metrics import Metric
from config import cfg


class EnclosuresRectangularity(Metric):
    attr_name = cfg.layers.enclosures.attributes.rectangularity

    def _calculate(self, enclosures: gpd.GeoDataFrame, **kwargs) -> gpd.GeoDataFrame:
        return momepy.Rectangularity(enclosures).series


class EnclosuresElongation(Metric):
    attr_name = cfg.layers.enclosures.attributes.elongation

    def _calculate(self, enclosures: gpd.GeoDataFrame, **kwargs) -> gpd.GeoDataFrame:
        return momepy.Elongation(enclosures).series


class EnclosuresFractalDimension(Metric):
    attr_name = cfg.layers.enclosures.attributes.fractal_dimension

    def _calculate(self, enclosures: gpd.GeoDataFrame, **kwargs) -> gpd.GeoDataFrame:
        return momepy.FractalDimension(enclosures).series


class EnclosureArea(Metric):
    attr_name = "area"
    priority = 1  # Before buildingenclosureid

    def _calculate(self, enclosures: gpd.GeoDataFrame, **kwargs) -> pd.Series:
        return enclosures.area


class EnclosuresCircularCompactness(Metric):
    attr_name = cfg.layers.enclosures.attributes.circular_compactness

    def _calculate(self, enclosures: gpd.GeoDataFrame, **kwargs) -> gpd.GeoDataFrame:
        return momepy.CircularCompactness(enclosures).series


class EnclosuresPerimeterAreaRatio(Metric):
    attr_name = cfg.layers.enclosures.attributes.perimeter_area_ratio

    def _calculate(self, enclosures: gpd.GeoDataFrame, **kwargs) -> gpd.GeoDataFrame:
        return momepy.Perimeter(enclosures).series / momepy.Area(enclosures).series


class EnclosuresBuildingAreaRatio(Metric):
    attr_name = cfg.layers.enclosures.attributes.enclosurue_building_area_ratio

    def _calculate(
        self,
        enclosures: gpd.GeoDataFrame,
        buildings: gpd.GeoDataFrame,
        **kwargs,
    ) -> gpd.GeoDataFrame:
        # tesselation has buildings.unique_id and enclosures.unique_id
        buildings_area = buildings.groupby(cfg.layers.enclosures.unique_id).area.sum()
        return buildings_area / enclosures.area
