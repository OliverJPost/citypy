from pathlib import Path

import catboost
import geopandas as gpd
import libpysal
import momepy
import networkx as nx
import numpy as np
import pandas as pd
from geopandas import sjoin
from libpysal.graph import Graph

from config import cfg

from ...cli.commands.classes import categ_columns, get_classification_columns
from . import BEFORE_LAYERS, BEFORE_PLOTS
from .baseclass import Metric


def lazy_generate_spatial_weights(
    tessellation: gpd.GeoDataFrame, verbose=True
) -> libpysal.weights.contiguity.Queen:
    return libpysal.weights.contiguity.Queen.from_dataframe(
        tessellation, ids=cfg.layers.buildings.unique_id, silence_warnings=True
    )


def lazy_generate_high_spatial_weights(
    original_weights: libpysal.weights.contiguity.Queen, verbose=True
) -> libpysal.weights.contiguity.Queen:
    return momepy.sw_high(k=3, weights=original_weights)


class BuildingArea(Metric):
    attr_name = cfg.layers.buildings.attributes.area
    priority = 1  # Before EnclosureID

    def _calculate(self, buildings: gpd.GeoDataFrame, **kwargs) -> gpd.GeoDataFrame:
        return buildings.area


class BuildingPerimeter(Metric):
    attr_name = cfg.layers.buildings.attributes.perimeter

    def _calculate(self, buildings: gpd.GeoDataFrame, **kwargs) -> gpd.GeoDataFrame:
        return buildings.length


class BuildingEquivalentRectangularIndex(Metric):
    attr_name = cfg.layers.buildings.attributes.equivalent_rectangular_index

    def _calculate(self, buildings: gpd.GeoDataFrame, **kwargs) -> gpd.GeoDataFrame:
        return momepy.EquivalentRectangularIndex(buildings).series


class BuildingElongation(Metric):
    attr_name = cfg.layers.buildings.attributes.elongation

    def _calculate(self, buildings: gpd.GeoDataFrame, **kwargs) -> gpd.GeoDataFrame:
        return momepy.Elongation(buildings).series


class BuildingSharedWallsRatio(Metric):
    attr_name = cfg.layers.buildings.attributes.shared_walls_ratio

    def _calculate(self, buildings: gpd.GeoDataFrame, **kwargs) -> gpd.GeoDataFrame:
        return momepy.SharedWallsRatio(buildings).series


class BuildingNeighborDistance(Metric):
    attr_name = cfg.layers.buildings.attributes.neighbor_distance

    def _calculate(
        self, buildings: gpd.GeoDataFrame, spatial_weights: gpd.GeoDataFrame, **kwargs
    ) -> gpd.GeoDataFrame:
        return momepy.NeighborDistance(
            buildings, spatial_weights, cfg.layers.buildings.unique_id
        ).series


# class BuildingMeanInterbuildingDistance(Metric):
#     attr_name = cfg.layers.buildings.attributes.mean_interbuilding_distance
#
#     def _calculate(
#         self, buildings: gpd.GeoDataFrame, spatial_weights: Graph, **kwargs
#     ) -> gpd.GeoDataFrame:
#         return momepy.MeanInterbuildingDistance(
#             buildings, spatial_weights., cfg.layers.buildings.unique_id
#         ).series


# class BuildingAdjacency(Metric):
#     attr_name = cfg.layers.buildings.attributes.adjacency
#
#     def _calculate(
#         self, buildings: gpd.GeoDataFrame, spatial_weights: gpd.GeoDataFrame, **kwargs
#     ) -> gpd.GeoDataFrame:
#         return momepy.BuildingAdjacency(
#             buildings, spatial_weights, cfg.layers.buildings.unique_id
#         ).series


class BuildingClosestRoadEdgeID(Metric):
    attr_name = cfg.layers.buildings.attributes.closest_road_edge_id
    priority = BEFORE_PLOTS

    def _calculate(
        self, buildings: gpd.GeoDataFrame, road_edges: gpd.GeoDataFrame, **kwargs
    ) -> gpd.GeoDataFrame:
        return momepy.get_nearest_street(buildings, road_edges, max_distance=200)


class BuildingGroupID(Metric):
    attr_name = cfg.layers.buildings.attributes.building_group_id
    priority = BEFORE_LAYERS

    def _calculate(self, buildings: gpd.GeoDataFrame, **kwargs) -> gpd.GeoDataFrame:
        touching_pairs = sjoin(buildings, buildings, how="inner", predicate="touches")
        touching_pairs["index_left"] = touching_pairs.index
        touching_pairs = touching_pairs[
            touching_pairs.index_left != touching_pairs.index_right
        ]

        # Create graph where buildings are nodes and edges represent touching pairs
        G = nx.Graph()
        G.add_nodes_from(buildings.index)
        G.add_edges_from(
            touching_pairs[["index_left", "index_right"]].itertuples(index=False)
        )

        # Find connected components (i.e., groups of touching buildings)
        components = nx.connected_components(G)

        # Assign a unique ID to each connected component
        group_ids = {}
        for i, component in enumerate(components):
            for node in component:
                group_ids[node] = i

        return buildings.index.map(group_ids)


class BuildingGroupEnclosurePerimeterCoverage(Metric):
    attr_name = (
        cfg.layers.buildings.attributes.building_group_enclosure_perimeter_coverage
    )

    def _calculate(
        self, buildings: gpd.GeoDataFrame, enclosures: gpd.GeoDataFrame, **kwargs
    ) -> pd.Series:
        OFFSET = -10
        # Get the group ID of each building
        building_group_id = buildings[cfg.layers.buildings.attributes.building_group_id]

        # Dissolve buildings into groups
        merged_building_groups = buildings.dissolve(
            by=cfg.layers.buildings.attributes.building_group_id, aggfunc="sum"
        )
        result = {}
        for i, building_group in merged_building_groups.iterrows():
            enclosure = enclosures.loc[enclosures.intersects(building_group.geometry)]
            if enclosure.empty:
                continue
            perimeter_offset = enclosure.geometry.buffer(OFFSET).boundary
            intersection = building_group.geometry.intersection(perimeter_offset)
            perimeter_coverage_ratio = intersection.length / perimeter_offset.length
            group_buildings = building_group_id[building_group_id == i]
            result.update(
                {
                    building_id: perimeter_coverage_ratio.iloc[0]
                    for building_id in group_buildings.index
                }
            )

        return pd.Series(result)


class BuildingGroupAreaStd(Metric):
    attr_name = cfg.layers.buildings.attributes.building_group_area_std

    def _calculate(self, buildings: gpd.GeoDataFrame, **kwargs) -> pd.Series:
        area_std = buildings.groupby(
            cfg.layers.buildings.attributes.building_group_id
        ).area.std()
        # Map back to buildings
        building_group_area_std = buildings[
            cfg.layers.buildings.attributes.building_group_id
        ].map(area_std)
        return building_group_area_std


class BuildingGroupElongationStd(Metric):
    attr_name = cfg.layers.buildings.attributes.building_group_elongation_std

    def _calculate(self, buildings: gpd.GeoDataFrame, **kwargs) -> pd.Series:
        area_std = buildings.groupby(
            cfg.layers.buildings.attributes.building_group_id
        ).elongation.std()
        # Map back to buildings
        building_group_area_std = buildings[
            cfg.layers.buildings.attributes.building_group_id
        ].map(area_std)
        return building_group_area_std


class BuildingGroupCourtyardIndex(Metric):
    attr_name = cfg.layers.buildings.attributes.building_group_courtyard_index

    def _calculate(self, buildings: gpd.GeoDataFrame, **kwargs) -> pd.Series:
        buiding_groups = buildings.dissolve(
            by=cfg.layers.buildings.attributes.building_group_id, aggfunc="sum"
        )
        courtyard_area = momepy.CourtyardArea(buiding_groups).series
        courtyard_index = momepy.CourtyardIndex(buiding_groups, courtyard_area).series
        # Map back to buildings
        building_group_courtyard_index = buildings[
            cfg.layers.buildings.attributes.building_group_id
        ].map(courtyard_index)

        # Cap at 0-1, some values are negative for some reason
        building_group_courtyard_index = building_group_courtyard_index.clip(0, 1)

        return building_group_courtyard_index


class BuildingEnclosureID(Metric):
    attr_name = cfg.layers.enclosures.unique_id
    priority = 2  # Needed by other metrics, but after enclosure area

    def _calculate(
        self, buildings: gpd.GeoDataFrame, enclosures: gpd.GeoDataFrame, **kwargs
    ) -> pd.Series:
        # Spatial join with intersection predicate
        building_enclosure = sjoin(
            buildings,
            enclosures,
            how="inner",
            predicate="intersects",
            rsuffix="enclosure",
        )

        # Keep only one per building, by biggest area
        building_enclosure = (
            building_enclosure.sort_values(by="area_enclosure", ascending=False)
            .drop_duplicates(subset=[cfg.layers.buildings.unique_id])
            .reset_index(drop=True)
        )

        # Ensure the index used for mapping is unique
        unique_building_enclosure = building_enclosure.drop_duplicates(
            subset=[cfg.layers.buildings.unique_id]
        )

        # Map back to buildings using reindex to handle missing values
        try:
            building_enclosure_id = (
                buildings.index.to_series()
                .map(
                    unique_building_enclosure.set_index(cfg.layers.buildings.unique_id)[
                        cfg.layers.enclosures.unique_id
                    ]
                )
                .reindex(buildings.index)
            )
        except KeyError:
            building_enclosure_id = (
                buildings.index.to_series()
                .map(
                    unique_building_enclosure.set_index(cfg.layers.buildings.unique_id)[
                        cfg.layers.enclosures.unique_id + "_enclosure"
                    ]
                )
                .reindex(buildings.index)
            )

        return building_enclosure_id


class BuildingFormFactor(Metric):
    attr_name = cfg.layers.buildings.attributes.form_factor

    def _calculate(self, buildings: gpd.GeoDataFrame, **kwargs) -> pd.Series:
        volumes = momepy.Volume(
            buildings, cfg.layers.buildings.attributes.approximate_height, "area"
        ).series
        return momepy.FormFactor(
            buildings,
            volumes,
            areas="area",
            heights=cfg.layers.buildings.attributes.approximate_height,
        ).series


class BuildingLandUseCategory(Metric):
    attr_name = cfg.layers.buildings.attributes.land_use_category

    def _calculate(
        self, buildings: gpd.GeoDataFrame, landuse: gpd.GeoDataFrame, **kwargs
    ) -> pd.Series:
        # sjoin building with landuse
        building_landuse = sjoin(
            buildings, landuse, how="inner", predicate="intersects", rsuffix=""
        )
        # keep intersection with biggest area as most likely one TODO better heuristic
        building_landuse = building_landuse.sort_values(
            "area", ascending=False
        ).drop_duplicates(cfg.layers.buildings.unique_id)

        return building_landuse["landuse_class"]


class BuildingShapeIndex(Metric):
    attr_name = cfg.layers.buildings.attributes.shape_index

    def _calculate(self, buildings: gpd.GeoDataFrame, **kwargs) -> pd.Series:
        longest_axis = momepy.LongestAxisLength(buildings).series
        return momepy.ShapeIndex(buildings, longest_axis).series


class BuildingOrientation(Metric):
    attr_name = cfg.layers.buildings.attributes.orientation
    priority = 3

    def _calculate(self, buildings: gpd.GeoDataFrame, **kwargs) -> pd.Series:
        return momepy.orientation(buildings)


# class BuildingNeighbourhoodImperviousLandCoverPercentage(Metric):
#     attr_name = (
#         cfg.layers.buildings.attributes.neighbourhood_impervious_land_cover_percentage
#     )
#
#     def _calculate(
#         self, buildings: gpd.GeoDataFrame, land_cover: gpd.GeoDataFrame, **kwargs
#     ) -> pd.Series:
#         # for each building, get the oobb and grow it by 20 meters in all directions
#         # then intersect with land cover and calculate the percentage of impervious land cover
#         oobbs = buildings.envelope
#
#         building_landuse = sjoin(
#             buildings, landuse, how="inner", predicate="intersects", rsuffix=""
#         )
#         # keep intersection with biggest area as most likely one TODO better heuristic
#         building_landuse = building_landuse.sort_values(
#             "area", ascending=False
#         ).drop_duplicates(cfg.layers.buildings.unique_id)
#         # Map back to buildings
#         building_landuse_category = buildings.index.map(
#             building_landuse[cfg.layers.landuse.unique_id]
#         )
#         # Calculate impervious land cover percentage
#         impervious_land_cover = building_landuse[
#             building_landuse["type_category"].isin(
#                 cfg.layers.landuse.impervious_land_cover_categories
#             )
#         ]
#         impervious_land_cover_percentage = (
#             impervious_land_cover.groupby(cfg.layers.buildings.unique_id).area.sum()
#             / buildings.area
#         )
#         return impervious_land_cover_percentage


class BuildingAlignment(Metric):
    attr_name = cfg.layers.buildings.attributes.alignment

    def _calculate(
        self, buildings: gpd.GeoDataFrame, spatial_weights, **kwargs
    ) -> pd.Series:
        orientations = buildings[BuildingOrientation.attr_name]
        return momepy.alignment(orientations, spatial_weights)


class BuildingSquareness(Metric):
    attr_name = cfg.layers.buildings.attributes.squareness
    priority = 1

    def _calculate(self, buildings: gpd.GeoDataFrame, **kwargs) -> pd.Series:
        return momepy.Squareness(buildings).series


class BuildingApproximateHeight(Metric):
    attr_name = cfg.layers.buildings.attributes.approximate_height
    priority = 1

    def _calculate(self, buildings: gpd.GeoDataFrame, **kwargs) -> pd.Series:
        # Convert to numeric, coercing errors to NaN
        if (
            cfg.layers.buildings.base_attributes.real_height not in buildings.columns
            or cfg.layers.buildings.base_attributes.real_levels not in buildings.columns
        ):
            return buildings[cfg.layers.buildings.base_attributes.ghsl_height].astype(
                np.float64
            )

        approx_height = (
            pd.to_numeric(
                buildings[cfg.layers.buildings.base_attributes.real_height],
                errors="coerce",
            )
            .fillna(
                pd.to_numeric(
                    buildings[cfg.layers.buildings.base_attributes.real_levels],
                    errors="coerce",
                )
                * cfg.layers.buildings.attributes.approximate_height_level_height_m
            )
            .fillna(
                buildings[cfg.layers.buildings.base_attributes.ghsl_height],
            )
        )

        # Return as float64
        return approx_height.astype(np.float64)


#
# class BuildingGroupCourtyardCount(Metric):
#     attr_name = cfg.layers.buildings.attributes.building_group_courtyard_count
#
#     def _calculate(self, buildings: gpd.GeoDataFrame, **kwargs) -> pd.Series:
#         buiding_groups = buildings.dissolve(
#             by=cfg.layers.buildings.attributes.building_group_id, aggfunc="sum"
#         )
#         return momepy.Courtyards(buiding_groups).series
#
#         # # Ensure merged_building_groups is a GeoDataFrame
#         # merged_building_groups = gpd.GeoDataFrame(
#         #     merged_building_groups, geometry="geometry"
#         # )
#         #
#         # # Get the perimeter of each enclosure and convert to GeoDataFrame
#         # enclosure_perimeter = enclosures.geometry.buffer(-5).boundary
#         # enclosure_perimeter_gdf = gpd.GeoDataFrame(
#         #     geometry=enclosure_perimeter, crs=enclosures.crs
#         # )
#         #
#         # # Perform the intersection
#         # intersection = gpd.overlay(
#         #     merged_building_groups, enclosure_perimeter_gdf, how="intersection"
#         # )
#         #
#         # # Calculate the perimeter coverage ratio
#         # perimeter_coverage_ratio = intersection.length / enclosure_perimeter_gdf.length
#         # # filter out inf
#         # perimeter_coverage_ratio = perimeter_coverage_ratio.replace(
#         #     [np.inf, -np.inf], np.nan
#         # )
#         #
#         # # Map the calculated ratios back to the building group IDs
#         # building_group_enclosure_perimeter_coverage = building_group_id.map(
#         #     perimeter_coverage_ratio
#         # )
#
#         return building_group_enclosure_perimeter_coverage


# Disabled because of use of spatial weights
class BuildingCoveredArea(Metric):
    attr_name = cfg.layers.buildings.attributes.covered_area

    def _calculate(
        self, buildings: gpd.GeoDataFrame, spatial_weights: gpd.GeoDataFrame, **kwargs
    ) -> gpd.GeoDataFrame:
        return momepy.CoveredArea(
            buildings, spatial_weights, cfg.layers.buildings.unique_id
        ).series


class BuildingClass(Metric):
    attr_name = cfg.layers.buildings.attributes.building_class
    priority = 99999  # after everything

    def _calculate(self, buildings: gpd.GeoDataFrame, **kwargs) -> pd.Series:
        model = catboost.CatBoostClassifier(
            iterations=1000,  # Number of boosting iterations
            learning_rate=0.1,  # Learning rate
            depth=6,  # Depth of the tree
            loss_function="MultiClass",  # Specify multiclass classification
            verbose=True,  # Enable verbose output for tracking the training process
            cat_features=categ_columns,
        )
        fn = Path(__file__).parent / "building_classes.cbm"
        model.load_model(fname=str(fn), format="cbm")
        features = get_classification_columns(buildings)
        labels = model.predict(features)

        return pd.Series(labels.flatten())
