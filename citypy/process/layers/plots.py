import geopandas as gpd
from shapely import LineString

from citypy.process.layers import Layer
from citypy.process.metrics import BEFORE_PLOTS
from config import cfg

# class PlotSeeds(Layer):
#     layer_name = "plot_seeds"
#
#     def _generate(self, road_edges, buildings, tesselation, **kwargs):
#         return {self.layer_name: plot_seeds}


class PlotSeeds(Layer):
    layer_name = "plots"

    def _generate(self, road_edges, buildings, tesselation, **kwargs):
        # class TesselationPlotID(Metric):
        #     attr_name = cfg.layers.tesselation.attributes.tesselation_plot_id
        #     priority = BEFORE_PLOTS + 1  # After ClosestRoadID
        #
        #     def _calculate(
        #         self,
        #         road_edges: gpd.GeoDataFrame,
        #         buildings: gpd.GeoDataFrame,
        #         tesselation: gpd.GeoDataFrame,
        #         **kwargs,
        #     ) -> gpd.GeoDataFrame:
        road_edge_dict = road_edges.set_index(cfg.layers.road_edges.unique_id).geometry

        non_outhouse_buildings = buildings[
            buildings.area > cfg.layers.tesselation.area_threshold_m2
        ]
        non_outhouse_buildings["geometry"] = non_outhouse_buildings.buffer(-1.0)
        buildings_inset = non_outhouse_buildings[~non_outhouse_buildings.is_empty]
        buildings_inset = buildings_inset[buildings_inset.geometry.is_valid]

        closest_road_geometry = buildings_inset[
            cfg.layers.buildings.attributes.closest_road_edge_id
        ].map(road_edge_dict)

        closest_road_geometry = gpd.GeoSeries(
            closest_road_geometry, index=buildings_inset.index
        )
        closest_road_geometry.crs = buildings_inset.crs

        line_to_road = gpd.GeoSeries.shortest_line(
            buildings_inset.centroid, closest_road_geometry
        )

        gdf = gpd.GeoDataFrame(
            {
                "geometry": line_to_road,
                cfg.layers.buildings.unique_id: buildings_inset[
                    cfg.layers.buildings.unique_id
                ],
            },
            crs=buildings_inset.crs,
        )
        return {self.layer_name: gdf}
