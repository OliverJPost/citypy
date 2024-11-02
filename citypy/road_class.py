from enum import Enum


class EdgeClass(Enum):
    HIGHWAY = 4
    MAJOR = 3
    STREET = 2
    EXTRA = 1
    NONE = 0

    @classmethod
    def from_highway(cls, highway: str):
        if highway in ("motorway", "motorway_link", "trunk", "trunk_link"):
            return cls.HIGHWAY
        elif highway in ("primary", "secondary", "tertiary"):
            return cls.MAJOR
        elif highway in (
            "living_street",
            "residential",
            "pedestrian",
            "unclassified",
            "service",
        ):
            return cls.STREET
        else:
            return cls.EXTRA

    def __lt__(self, other):
        return self.value < other.value

    def __le__(self, other):
        return self.value <= other.value
