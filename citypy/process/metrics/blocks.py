import geopandas as gpd
import momepy

from config import cfg

from .baseclass import Metric


# class BlocksCircularCompactness(Metric):
#     attr_name = cfg.layers.blocks.attributes.circular_compactness
#
#     def _calculate(self, blocks: gpd.GeoDataFrame, **kwargs) -> gpd.GeoDataFrame:
#         return momepy.CircularCompactness(blocks).series
#
#
# class BlocksEquivalentRectangularIndex(Metric):
#     attr_name = cfg.layers.blocks.attributes.equivalent_rectangular_index
#
#     def _calculate(self, blocks: gpd.GeoDataFrame, **kwargs) -> gpd.GeoDataFrame:
#         return momepy.EquivalentRectangularIndex(blocks).series
#
#
# class BlocksRectangularity(Metric):
#     attr_name = cfg.layers.blocks.attributes.rectangularity
#
#     def _calculate(self, blocks: gpd.GeoDataFrame, **kwargs) -> gpd.GeoDataFrame:
#         return momepy.Rectangularity(blocks).series
