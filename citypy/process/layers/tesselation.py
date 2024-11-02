import geopandas as gpd
import momepy

from config import cfg

from .baseclass import Layer


class EnclosuresTesselation(Layer):
    layer_name = "tesselation"
    priority = 2

    def _generate(
        self,
        buildings: gpd.GeoDataFrame,
        road_edges: gpd.GeoDataFrame,
        enclosures: gpd.GeoDataFrame,
        **kwargs,
    ):
        area_too_small = buildings.area < cfg.layers.tesselation.area_threshold_m2

        # Join buildings together with same "building_group_id"
        buildings = buildings[~area_too_small].dissolve(
            by=cfg.layers.buildings.attributes.building_group_id
        )
        buildings = buildings[~buildings.is_empty]

        # no multipolygon enclosures
        enclosures = enclosures[enclosures.geom_type == "Polygon"]
        tesselation = momepy.enclosed_tessellation(
            buildings.set_index(cfg.layers.buildings.unique_id),
            enclosures,
            segment=2.0,
            n_jobs=1,
        )
        tesselation[cfg.layers.buildings.unique_id] = tesselation.index

        return {self.layer_name: tesselation}


#
# class MorphologicalTesselation(Layer):
#     layer_name = "tesselation"
#     priority = 2
#
#     def _generate(self, buildings: gpd.GeoDataFrame, **kwargs):
#         limit = momepy.buffered_limit(buildings, 400)
#         tesselation = momepy.Tessellation(
#             buildings,
#             unique_id=cfg.layers.buildings.unique_id,
#             limit=limit,
#             verbose=True,
#         )
#
#         return {self.layer_name: tesselation.tessellation}
