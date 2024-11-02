import json
from unittest.mock import MagicMock, patch

import geopandas as gpd
import osmnx
import pytest


@pytest.fixture
def mock_raw_building_data():
    with patch("osmnx.features_from_place") as mock_get:
        mock_get.return_value = gpd.read_file(
            "test_data/raw_buildings.gpkg", engine="pyogrio"
        )
        yield mock_get
