from __future__ import annotations

from typing import List

from onkia.models import WaterTempPreference

_RAW_DATA = [
    ("Atlantic Salmon", 45, "50-62", None),
    ("Black Crappie", 60, "71", 75),
    ("Bluegills", 58, "69-72", 80),
    ("Blue Catfish", None, "77-82", None),
    ("Brook Trout", 44, "58", 70),
    ("Brown Bullhead", 65, "78-82", 85),
    ("Brown Trout", 44, "56-66", 75),
    ("Burbot", None, "52", None),
    ("Channel Catfish", 55, "82-89", None),
    ("Chinook Salmon", 44, "55", 60),
    ("Coho Salmon", 44, "55", 60),
    ("Flathead Catfish", 81, "85", 90),
    ("Grayling", None, None, 65),
    ("Lake Trout", 40, "50-55", 60),
    ("Largemouth Bass", 50, "65-75", 85),
    ("Muskellunge", 55, "63-67", 78),
    ("Northern Pike", 55, "65", 74),
    ("Pumpkinseed", None, "81", None),
    ("Rainbow Trout", 44, "61", 75),
    ("Rock Bass", None, "70", None),
    ("Smallmouth Bass", 60, "65-70", 73),
    ("Splake", 42, "50-66", 68),
    ("Steelhead Trout", 38, "55-60", None),
    ("Sunfish", 50, "58", 68),
    ("Walleye", 50, "64-70", 76),
    ("White Bass", 55, "65-70", 80),
    ("White Catfish", None, "80-85", None),
    ("White Crappie", None, "61", None),
    ("White Perch", None, "89", None),
    ("Yellow Perch", 58, "68", 75),
]


def _parse_preferences() -> List[WaterTempPreference]:
    return [
        WaterTempPreference(
            species=species,
            lower_avoidance=lower,
            optimum=str(optimum) if optimum is not None else "N/A",
            upper_avoidance=upper,
        )
        for species, lower, optimum, upper in _RAW_DATA
    ]


WATER_TEMP_PREFERENCES: List[WaterTempPreference] = _parse_preferences()
