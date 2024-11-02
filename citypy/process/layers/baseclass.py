from abc import ABC, abstractmethod
from typing import List

import geopandas as gpd

from citypy.console import console
from citypy.logging_setup import log_time
from citypy.util.gpkg import GeoPackage


class Layer(ABC):
    def __init__(self):
        pass

    @log_time
    def generate(self, **kwargs) -> dict[str, gpd.GeoDataFrame]:
        console.log(f"Generating layer: '{self.layer_name}'...")
        return self._generate(**kwargs)

    @abstractmethod
    def _generate(self, **kwargs) -> dict[str, gpd.GeoDataFrame]:
        pass

    @property
    @abstractmethod
    def layer_name(self):
        pass

    def should_generate(
        self, gpkg: GeoPackage, select: List[str], overwrite: bool
    ) -> bool:
        if getattr(self, "disabled", False):
            console.log(
                f"Skipping layer [bold]{self.layer_name}[/bold]... [dim]Disabled.[/dim]"
            )
            return False
        if select and self.layer_name not in select:
            console.log(
                f"Skipping layer [bold]{self.layer_name}[/bold]... [dim]Not selected.[/dim]"
            )
            return False
        if self.layer_name in gpkg.list_all_layers() and not overwrite:
            console.log(
                f"Layer [bold]{self.layer_name}[/bold] already exist. [dim]Use --overwrite (-o) to overwrite.[/dim]"
            )
            return False
        return True
