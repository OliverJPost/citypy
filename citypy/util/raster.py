import geopandas as gpd
import numpy as np
import rasterio
import rasterio.crs
import rasterio.mask
import rasterio.transform
from rasterio.merge import merge
from shapely import Point


class GeoRaster:
    def __init__(
        self, raster: np.array, transform: rasterio.transform, crs: rasterio.crs.CRS
    ):
        if raster.dtype not in [np.uint16, np.int16, np.float32]:
            raise ValueError(
                f"Raster data type must be one of uint16, int16, or float32, not {raster.dtype}"
            )

        self.raster = raster
        self.transform = transform
        self.crs = crs

    @classmethod
    def from_tiles(cls, georasters: list["GeoRaster"]) -> "GeoRaster":
        # Collect the raster datasets and their transforms
        datasets = []
        for georaster in georasters:
            datasets.append((georaster.raster, georaster.transform))

        # Use rasterio's merge function to merge the datasets
        merged_raster, merged_transform = merge(
            [
                rasterio.io.MemoryFile().open(
                    driver="GTiff",
                    width=raster.shape[2],
                    height=raster.shape[1],
                    count=raster.shape[0],
                    dtype=raster.dtype,
                    transform=transform,
                    crs=georaster.crs,
                )
                for (raster, transform) in datasets
            ]
        )

        # Assuming all input rasters have the same CRS
        return cls(merged_raster, merged_transform, georasters[0].crs)

    @classmethod
    def from_file(cls, filepath: str) -> "GeoRaster":
        with rasterio.open(filepath) as src:
            raster = src.read()
            transform = src.transform
            crs = src.crs
        return cls(raster, transform, crs)

    def reprojected(self, dst_crs: rasterio.crs.CRS) -> "GeoRaster":
        reprojected_raster, dst_transform = reproject_raster(
            self.raster, self.transform, self.crs, dst_crs
        )
        return GeoRaster(reprojected_raster, dst_transform, dst_crs)

    def sample(self, x: float, y: float) -> list[float | None]:
        col, row = ~self.transform * (x, y)
        row = int(row)
        col = int(col)
        try:
            return self.raster[:, row, col]
        except IndexError:
            return [None]

    def to_gdf(self) -> gpd.GeoDataFrame:
        points = []
        for row in range(self.raster.shape[1]):
            for col in range(self.raster.shape[2]):
                x, y = self.transform * (col, row)
                points.append(
                    {"geometry": Point(x, y), "value": self.raster[0, row, col]}
                )
        return gpd.GeoDataFrame(points, crs=self.crs.to_string())


def reproject_raster(
    raster: np.array,
    src_transform: rasterio.transform,
    src_crs: rasterio.crs.CRS,
    dst_crs: rasterio.crs.CRS,
) -> np.array:
    if len(raster.shape) == 3:
        if raster.shape[0] == 1:
            bands = [raster[0]]
        else:
            bands = [raster[i] for i in range(raster.shape[0])]
    else:
        bands = [raster]

    reprojected_bands = []
    for band in bands:
        reprojected_band, dst_transform = reproject_raster_band(
            band, src_transform, src_crs, dst_crs
        )
        reprojected_bands.append(reprojected_band)

    reprojected_raster = np.stack(reprojected_bands)

    return reprojected_raster, dst_transform


def reproject_raster_band(
    raster: np.array,
    src_transform: rasterio.transform,
    src_crs: rasterio.crs.CRS,
    dst_crs: rasterio.crs.CRS,
) -> np.array:
    src_width = raster.shape[1]
    src_height = raster.shape[0]
    left, top = src_transform * (0, 0)
    right, bottom = src_transform * (src_width, src_height)
    dst_transform, width, height = rasterio.warp.calculate_default_transform(
        src_crs=src_crs,
        dst_crs=dst_crs,
        width=src_width,
        height=src_height,
        left=left,
        right=right,
        top=top,
        bottom=bottom,
        src_transform=src_transform,
    )
    reprojected_raster = np.zeros((height, width), dtype=raster.dtype)

    rasterio.warp.reproject(
        raster,
        reprojected_raster,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=rasterio.enums.Resampling.nearest,
    )

    return reprojected_raster, dst_transform
