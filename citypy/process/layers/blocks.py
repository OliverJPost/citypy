import geopandas as gpd
import momepy

from config import cfg

from .baseclass import Layer

#
# class Blocks(Layer):
#     layer_name = "blocks"
#
#     def _generate(self, tesselation, road_edges, buildings, **kwargs):
#         # remove block id column from buildings and tesselation
#         if "block_id" in buildings.columns:
#             buildings = buildings.drop(columns=["block_id"])
#         if "block_id" in tesselation.columns:
#             tesselation = tesselation.drop(columns=["block_id"])
#
#         blocks = momepy.Blocks(
#             tesselation,
#             road_edges,
#             buildings,
#             cfg.layers.blocks.unique_id,
#             cfg.layers.buildings.unique_id,
#         )
#         buildings = buildings.assign(block_id=blocks.buildings_id)
#         tesselation = tesselation.assign(block_id=blocks.tessellation_id)
#         return {
#             self.layer_name: blocks.blocks,
#             "buildings": buildings,
#             "tesselation": tesselation,
#         }
