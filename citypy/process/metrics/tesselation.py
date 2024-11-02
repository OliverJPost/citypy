import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString
from tqdm import tqdm

from citypy.process.metrics import BEFORE_PLOTS, Metric
from config import cfg


class TesselationPlotID(Metric):
    attr_name = cfg.layers.plots.unique_id
    priority = BEFORE_PLOTS + 1  # After ClosestRoadID

    def _calculate(
        self,
        road_edges: gpd.GeoDataFrame,
        buildings: gpd.GeoDataFrame,
        tesselation: gpd.GeoDataFrame,
        **kwargs,
    ) -> gpd.GeoDataFrame:
        BUILDING_ID = cfg.layers.buildings.unique_id
        tesselation[BUILDING_ID] = tesselation[BUILDING_ID].astype("Int64")
        new_plot_id = 0
        road_edge_dict = road_edges.set_index(cfg.layers.road_edges.unique_id).geometry
        non_outhouse_buildings = buildings[
            buildings.area > cfg.layers.tesselation.area_threshold_m2
        ]
        # convert to gdf
        non_outhouse_buildings = gpd.GeoDataFrame(
            non_outhouse_buildings,
            crs=buildings.crs,
        )
        # non_outhouse_buildings["geometry"] = non_outhouse_buildings.buffer(-1.0)
        # buildings_inset = non_outhouse_buildings[~non_outhouse_buildings.is_empty]
        # buildings_inset = buildings_inset[buildings_inset.geometry.is_valid]

        closest_road_geometry = non_outhouse_buildings[
            cfg.layers.buildings.attributes.closest_road_edge_id
        ].map(road_edge_dict)

        closest_road_geometry = gpd.GeoSeries(
            closest_road_geometry, index=non_outhouse_buildings.index
        )
        closest_road_geometry.crs = non_outhouse_buildings.crs

        line_to_road = gpd.GeoSeries.shortest_line(
            non_outhouse_buildings.centroid, closest_road_geometry
        )
        # Shorten line on both sides by 0.5 m to prevent intersection with plots on the other side of the street
        line_to_road_shortened = []
        for line in line_to_road:
            if line is None or line.length < 1:
                line_to_road_shortened.append(LineString())
            else:
                start = line.interpolate(0.5)
                end = line.interpolate(line.length - 0.5)
                line_to_road_shortened.append(LineString([start, end]))

        line_to_road = gpd.GeoSeries(
            line_to_road_shortened, index=non_outhouse_buildings.index
        )

        line_gdf = gpd.GeoDataFrame(
            {"line_to_road": line_to_road, BUILDING_ID: non_outhouse_buildings.index},
            geometry="line_to_road",
            crs=non_outhouse_buildings.crs,
        )

        intersections = gpd.sjoin(
            line_gdf,
            tesselation,
            how="inner",
            predicate="intersects",
            lsuffix="source_building",
            rsuffix="tesselation",
        )

        tesselation = tesselation.set_index(BUILDING_ID)
        non_outhouse_buildings = non_outhouse_buildings.set_index(BUILDING_ID)

        # initiate plot id to None
        tesselation[self.attr_name] = None

        for building_id, group in tqdm(
            intersections.groupby(BUILDING_ID + "_source_building")
        ):
            if building_id not in tesselation.index:
                continue  # skip buildings outside of tesselation

            building_centroid = non_outhouse_buildings.loc[building_id][
                "geometry"
            ].centroid
            min_distance = float("inf")
            min_id = None
            for _, row in group.iterrows():
                if pd.isna(row[BUILDING_ID + "_tesselation"]):
                    continue  # Skip tesselation cells originating from an enclosure without buildings, these are not plots

                if (
                    row[BUILDING_ID + "_tesselation"]
                    == row[BUILDING_ID + "_source_building"]
                ):
                    continue

                distance = building_centroid.distance(
                    non_outhouse_buildings.loc[
                        int(row[BUILDING_ID + "_tesselation"])
                    ].geometry
                )
                if distance < min_distance:
                    min_distance = distance
                    min_id = row[BUILDING_ID + "_tesselation"]

            current_plot_id = tesselation.loc[building_id, self.attr_name]
            if isinstance(current_plot_id, pd.Series):
                current_plot_id = current_plot_id.iloc[0]
            if current_plot_id is None:
                current_plot_id = new_plot_id
                new_plot_id += 1
                tesselation.loc[building_id, self.attr_name] = current_plot_id

            if min_id is None:
                continue

            intersection_existing_plot_id = tesselation.loc[min_id, self.attr_name]
            if isinstance(intersection_existing_plot_id, pd.Series):
                intersection_existing_plot_id = intersection_existing_plot_id.iloc[0]

            if not intersection_existing_plot_id:
                tesselation.loc[min_id, self.attr_name] = current_plot_id
            else:
                tesselation.loc[
                    tesselation[self.attr_name] == current_plot_id,
                    self.attr_name,
                ] = intersection_existing_plot_id

        tesselation = tesselation.reset_index()
        return tesselation[self.attr_name]
