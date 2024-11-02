from test_util import column_exists, drop_if_exists, is_float, no_nulls, not_all_zero

#
# def test_add_building_area(processed_city):
#     drop_if_exists(processed_city.buildings, "area")
#     add_building_area(processed_city.buildings)
#     assert column_exists("area", processed_city.buildings)
#     assert is_float("area", processed_city.buildings)
#     assert no_nulls("area", processed_city.buildings)
#     assert not_all_zero("area", processed_city.buildings)
#
#
# def test_add_building_perimeter(processed_city):
#     drop_if_exists(processed_city.buildings, "perimeter")
#     add_building_perimeter(processed_city.buildings)
#     assert column_exists("perimeter", processed_city.buildings)
#     assert is_float("perimeter", processed_city.buildings)
#     assert no_nulls("perimeter", processed_city.buildings)
#     assert not_all_zero("perimeter", processed_city.buildings)
#
#
# def test_add_building_equivalent_rectangular_index(processed_city):
#     drop_if_exists(processed_city.buildings, "equivalent_rectangular_index")
#     add_building_equivalent_rectangular_index(processed_city.buildings)
#     assert column_exists("equivalent_rectangular_index", processed_city.buildings)
#     assert is_float("equivalent_rectangular_index", processed_city.buildings)
#     assert no_nulls("equivalent_rectangular_index", processed_city.buildings)
#     assert not_all_zero("equivalent_rectangular_index", processed_city.buildings)
#
#
# def test_add_building_elongation(processed_city):
#     drop_if_exists(processed_city.buildings, "elongation")
#     add_building_elongation(processed_city.buildings)
#     assert column_exists("elongation", processed_city.buildings)
#     assert is_float("elongation", processed_city.buildings)
#     assert no_nulls("elongation", processed_city.buildings)
#     assert not_all_zero("elongation", processed_city.buildings)
#
#
# def test_add_buildings_shared_walls_ratio(processed_city):
#     drop_if_exists(processed_city.buildings, "shared_walls_ratio")
#     add_building_shared_walls_ratio(processed_city.buildings)
#     assert column_exists("shared_walls_ratio", processed_city.buildings)
#     assert is_float("shared_walls_ratio", processed_city.buildings)
#     assert no_nulls("shared_walls_ratio", processed_city.buildings)
#     assert not_all_zero("shared_walls_ratio", processed_city.buildings)
#
#
# def test_add_building_neighbor_distance(processed_city, queen1):
#     drop_if_exists(processed_city.buildings, "neighbor_distance")
#     add_building_neighbor_distance(processed_city.buildings, queen1)
#     assert column_exists("neighbor_distance", processed_city.buildings)
#     assert is_float("neighbor_distance", processed_city.buildings)
#     assert no_nulls("neighbor_distance", processed_city.buildings)
#     assert not_all_zero("neighbor_distance", processed_city.buildings)
#
#
# def test_add_building_mean_interbuilding_distance(processed_city, queen1):
#     drop_if_exists(processed_city.buildings, "mean_interbuilding_distance")
#     add_building_mean_interbuilding_distance(processed_city.buildings, queen1)
#     assert column_exists("mean_interbuilding_distance", processed_city.buildings)
#     assert is_float("mean_interbuilding_distance", processed_city.buildings)
#     assert no_nulls("mean_interbuilding_distance", processed_city.buildings)
#     assert not_all_zero("mean_interbuilding_distance", processed_city.buildings)
#
#
# def test_add_building_adjacency(processed_city, queen3):
#     drop_if_exists(processed_city.buildings, "adjacency")
#     add_building_adjacency(processed_city.buildings, queen3)
#     assert column_exists("adjacency", processed_city.buildings)
#     assert is_float("adjacency", processed_city.buildings)
#     assert no_nulls("adjacency", processed_city.buildings)
#     assert not_all_zero("adjacency", processed_city.buildings)
