import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load the data into a DataFrame
df = pd.read_csv("~/Desktop/Result_13.csv")


def format_city(city_name):
    parts = city_name.split("_")
    name = ""
    for part in parts:
        if part == part.upper():
            break
        name += part + " "
    return name.title()


# Define the colour mapping for each building class
colours = {
    "apartments": "#9341b3",
    "big_commercial": "#3148dc",
    "complex": "#07f035",
    "detached": "#9b1e14",
    "filled_block": "#e68653",
    "industrial": "#5d4b20",
    "irregular_block": "#10dfff",
    "perimeter_block": "#d6e048",
    "terraced": "#469374",
}

# Determine the number of rows and columns for the grid
unique_cities = df["city"].unique()
unique_cities = [c for c in unique_cities if c != "Paris_FRANCE"]
n_cities = len(unique_cities)
n_cols = 5  # Define the number of columns
n_rows = (n_cities + n_cols - 1) // n_cols  # Calculate the required rows

# Create a subplot grid for all cities
fig = make_subplots(
    rows=n_rows,
    cols=n_cols,
    subplot_titles=[format_city(city) for city in unique_cities],
    specs=[
        [{"type": "domain"} for _ in range(n_cols)] for _ in range(n_rows)
    ],  # Pie charts require "domain" type
    vertical_spacing=0.05,
)

# Add a pie chart for each city in the corresponding grid position
for i, city in enumerate(unique_cities):
    row = (i // n_cols) + 1
    col = (i % n_cols) + 1

    city_data = df[df["city"] == city]

    # Filter out building classes with a percentage below 1%
    city_data = city_data[city_data["percentage_of_city"] >= 1]

    building_types = city_data[
        "buildings::building_class::most_intersecting::mode_kernel3"
    ]
    percentages = city_data["percentage_of_city"]

    # Add the pie chart to the specific row and column
    fig.add_trace(
        go.Pie(
            labels=building_types,
            values=percentages,
            marker=dict(
                colors=[
                    colours[bt]
                    for bt in building_types
                    if bt
                    != "buildings::building_class::most_intersecting::mode_kernel3"
                ]
            ),
            textinfo="label+percent",
            insidetextorientation="auto",
        ),
        row=row,
        col=col,
    )

# Adjust subplot title font size
for annotation in fig["layout"]["annotations"]:
    annotation["font"] = dict(size=50)  # Set larger font size for subplot titles

# Update layout for the entire figure
fig.update_layout(
    height=650
    * n_rows,  # Set the overall figure height proportional to the number of rows
    width=800
    * n_cols,  # Set the width of the figure proportional to the number of columns
    showlegend=False,  # Hide the legend
    font=dict(size=25),
    # subplot labels a little higher
    margin=dict(t=100),
)

# Save the figure as a PNG file
fig.write_image("building_types_grid.png", width=800 * n_cols, height=650 * n_rows)

df = pd.read_csv("~/Desktop/Result_14.csv")
colours = {
    "0": "#f0c808",
    "1": "#5d4b20",
    "10": "#469374",
    "11": "#9341b3",
    "12": "#e3427d",
    "2": "#e68653",
    "3": "#9a1500",
    "4": "#26ff00",
    "5": "#3a56e6",
    "6": "#009dff",
    "7": "#035c00",
    "8": "#d6e13b",
    "9": "#28e3d3",
}


# Determine the number of rows and columns for the grid
unique_cities = df["city"].unique()
unique_cities = [c for c in unique_cities if c != "Paris_FRANCE"]
n_cities = len(unique_cities)
n_cols = 5  # Define the number of columns
n_rows = (n_cities + n_cols - 1) // n_cols  # Calculate the required rows

# Create a subplot grid for all cities
fig = make_subplots(
    rows=n_rows,
    cols=n_cols,
    subplot_titles=[format_city(city) for city in unique_cities],
    specs=[
        [{"type": "domain"} for _ in range(n_cols)] for _ in range(n_rows)
    ],  # Pie charts require "domain" type
    vertical_spacing=0.05,
)

# Add a pie chart for each city in the corresponding grid position
for i, city in enumerate(unique_cities):
    row = (i // n_cols) + 1
    col = (i % n_cols) + 1

    city_data = df[df["city"] == city]

    # Filter out building classes with a percentage below 1%
    city_data = city_data[city_data["percentage_of_city"] >= 1]

    building_types = city_data[
        "road_edges::cluster_street_gmm13::most_intersecting::mode_kernel3"
    ]
    percentages = city_data["percentage_of_city"]

    # Add the pie chart to the specific row and column
    fig.add_trace(
        go.Pie(
            labels=building_types,
            values=percentages,
            marker=dict(
                colors=[
                    colours[str(bt)]
                    for bt in building_types
                    if bt
                    != "road_edges::cluster_street_gmm13::most_intersecting::mode_kernel3"
                ]
            ),
            textinfo="label+percent",
            insidetextorientation="auto",
        ),
        row=row,
        col=col,
    )

# Adjust subplot title font size
for annotation in fig["layout"]["annotations"]:
    annotation["font"] = dict(size=50)  # Set larger font size for subplot titles

# Update layout for the entire figure
fig.update_layout(
    height=650
    * n_rows,  # Set the overall figure height proportional to the number of rows
    width=800
    * n_cols,  # Set the width of the figure proportional to the number of columns
    showlegend=False,  # Hide the legend
    font=dict(size=25),
    # subplot labels a little higher
    margin=dict(t=100),
)

# Save the figure as a PNG file
fig.write_image("road_types_grid.png", width=800 * n_cols, height=650 * n_rows)
