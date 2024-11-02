from abc import ABC, abstractmethod
from typing import List

import geopandas as gpd
import networkx as nx

from citypy.console import conditional_status, console, display_status_if_verbose
from citypy.logging_setup import log_time


class Metric(ABC):
    def __init__(self):
        pass

    @log_time
    def calculate(self, verbose=True, **kwargs) -> gpd.GeoDataFrame:
        console.log(
            f"Calculating metric {self.__class__.__name__} as '{self.attr_name}'..."
        )
        result = self._calculate(**kwargs)
        return result

    @abstractmethod
    def _calculate(self, **kwargs) -> gpd.GeoDataFrame:
        pass

    def should_calculate(
        self, gdf: gpd.GeoDataFrame, select: List[str], overwrite: bool
    ) -> bool:
        if select and self.attr_name not in select:
            console.log(
                f"Skipping column [bold]{self.attr_name}[/bold]... [dim]Not selected.[/dim]"
            )
            return False
        if self.attr_name in gdf.columns and not overwrite:
            console.log(
                f"Column [bold]{self.attr_name}[/bold] already exist. [dim]Use --overwrite (-o) to overwrite.[/dim]"
            )
            return False
        return True

    @property
    @abstractmethod
    def attr_name(self) -> str:
        pass


class GraphMetric:
    def __init__(self):
        pass

    @log_time
    def calculate(self, verbose=True, **kwargs) -> nx.MultiDiGraph:
        console.log(
            f"Calculating metric {self.__class__.__name__} as '{self.attr_name}'..."
        )
        return self._calculate(**kwargs)

    @abstractmethod
    def _calculate(self, **kwargs) -> nx.MultiDiGraph:
        pass

    @property
    @abstractmethod
    def attr_name(self) -> str:
        pass

    def should_calculate(
        self, gdf: gpd.GeoDataFrame, select: List[str], overwrite: bool
    ) -> bool:
        if getattr(self, "disabled", False):
            console.log(
                f"Skipping layer [bold]{self.attr_name}[/bold]... [dim]Disabled.[/dim]"
            )
            return False
        if select and self.attr_name not in select:
            console.log(
                f"Skipping column [bold]{self.attr_name}[/bold]... [dim]Not selected.[/dim]"
            )
            return False
        if self.attr_name in gdf.columns and not overwrite:
            console.log(
                f"Column [bold]{self.attr_name}[/bold] already exist. [dim]Use --overwrite (-o) to overwrite.[/dim]"
            )
            return False
        return True
