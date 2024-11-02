import geopandas as gpd
import momepy
from libpysal import graph

from config import cfg

from .baseclass import Layer


# class SpatialWeights(Layer):
#     layer_name = "spatial_weights"
#
#     def _generate(
#         self, buildings: gpd.GeoDataFrame, extents: gpd.GeoDataFrame, **kwargs
#     ):
#         tessellation = momepy.Tessellation(
#             buildings,
#             unique_id=cfg.layers.buildings.unique_id,
#             limit=extents.to_crs(buildings.crs).union_all(),
#             segment=4.0,  # Very high, so tesselation is inaccurate but that's okay
#             verbose=True,
#         )
#
#         contiguity = graph.Graph.build_contiguity(tessellation.tessellation)
#
#         return {self.layer_name: tesselation.tessellation}
