import inspect
from importlib import import_module

from citypy.process.layers.baseclass import Layer

layer_modules = ["blocks", "enclosures", "tesselation", "spatial_weights"]

all_layers = []

for module_name in layer_modules:
    module = import_module(f".{module_name}", __package__)
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and issubclass(obj, Layer) and obj != Layer:
            all_layers.append(obj())

# sort by priority if hasattr priority. Lowest priority int is first
all_layers.sort(key=lambda x: getattr(x, "priority", 999))
