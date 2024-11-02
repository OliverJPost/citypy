import inspect
from collections import defaultdict
from importlib import import_module

from citypy.process.metrics.baseclass import GraphMetric, Metric
from citypy.process.metrics.constants import BEFORE_LAYERS, BEFORE_PLOTS

metric_modules = [
    "buildings",
    "road_edges",
    "road_nodes",
    "blocks",
    "enclosures",
    "tesselation",
]

before_plots_metrics = []
all_column_metrics = []
all_graph_metrics = []

for module_name in metric_modules:
    module = import_module(f"citypy.process.metrics.{module_name}")
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and issubclass(obj, Metric) and obj != Metric:
            all_column_metrics.append((module_name, obj()))
        if inspect.isclass(obj) and issubclass(obj, GraphMetric) and obj != GraphMetric:
            all_graph_metrics.append((module_name, obj()))

before_plots_metrics = [
    metric
    for metric in all_column_metrics
    if getattr(metric[1], "priority", 0) >= BEFORE_PLOTS
]
before_layers_metrics = [
    metric
    for metric in all_column_metrics
    if getattr(metric[1], "priority", 0) >= BEFORE_LAYERS
]
all_column_metrics = [
    metric
    for metric in all_column_metrics
    if getattr(metric[1], "priority", 0) < BEFORE_PLOTS
]

# sort by priority if hasattr priority. Lowest priority int is first
before_plots_metrics.sort(key=lambda x: x[1].priority)
before_layers_metrics.sort(key=lambda x: x[1].priority)
all_column_metrics.sort(key=lambda x: getattr(x[1], "priority", 999))
all_graph_metrics.sort(key=lambda x: getattr(x[1], "priority", 999))
