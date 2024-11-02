import geopandas as gpd
import momepy

from citypy.process.layers.baseclass import Layer
from config import cfg


class Enclosures(Layer):
    layer_name = "enclosures"
    priority = 1

    def _generate(self, road_edges, extents, **kwargs):
        extents_boundary = extents.boundary.unary_union
        boundary_gdf = gpd.GeoDataFrame(geometry=[extents_boundary], crs=extents.crs)

        enclosures = momepy.enclosures(
            road_edges,
            limit=extents.to_crs(road_edges.crs),
            enclosure_id=cfg.layers.enclosures.unique_id,
            additional_barriers=[boundary_gdf],
        )  # TODO also add additional barriers
        return {self.layer_name: enclosures}
