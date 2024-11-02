import geopandas as gpd

LANDUSE_CLASSES = {
    "residential": ["residential"],
    "industrial": ["industrial", "factory", "manufacture", "warehouse"],
    "retail": ["retail", "shop", "supermarket", "mall"],
    "commercial": ["commercial", "office", "bank", "government", "institutional"],
    # "green": [
    #     "nature",
    #     "park",
    #     "forest",
    #     "wood",
    #     "wetland",
    #     "meadow",
    #     "grass",
    #     "allotments",
    #     "farmland",
    #     "animal_keeping",
    #     "flowerbed",
    #     "farmyard",
    #     "vineyard",
    #     "orchard",
    #     "cemetery",
    #     "recreation_ground",
    #     "garden",
    #     "conservation",
    #     "recreation",
    #     "playground",
    #     "pitch",
    #     "sports",
    #     "golf_course",
    #     "meadow",
    # ],
}


def generate_landuse(landuse: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Generate landuse data for a city.

    Args:
        city (City): The city object.

    Returns:
        gpd.GeoDataFrame: The landuse data.
    """
    landuse["landuse_class"] = None
    for landuse_class, landuse_types in LANDUSE_CLASSES.items():
        landuse.loc[landuse["landuse"].isin(landuse_types), "landuse_class"] = (
            landuse_class
        )
    return landuse
