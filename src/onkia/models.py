from __future__ import annotations

from typing import Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


def _to_camel(name: str) -> str:
    parts = name.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


class OnkiaBaseModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=_to_camel,
        populate_by_name=True,
        extra="ignore",
    )


class Morphology(OnkiaBaseModel):
    area: float
    max_depth: int
    mean_depth: float
    shore_length: float
    littoral_area: float


class Bbox(OnkiaBaseModel):
    epsg4326: List[float] = Field(alias="epsg:4326")
    epsg26915: List[int] = Field(alias="epsg:26915")


class Reg(OnkiaBaseModel):
    species: List[str]
    text: str


class SpecialFishingReg(OnkiaBaseModel):
    location: str
    loc_display_type: int
    regs: List[Reg]


class Resource(OnkiaBaseModel):
    lake_health: int
    ice_out: int
    fca: int
    water_levels: int
    water_access: int
    water_quality: int
    sentinel_lake: int
    ice_in: int
    lake_map: int
    special_fishing_regs: int
    lake_survey: int
    fish_stocking: int


class Lake(OnkiaBaseModel):
    morphology: Morphology
    border: str = ""
    county_id: int
    fish_species: List[str]
    notes: str = ""
    special_fishing_regs: List[SpecialFishingReg] = []
    bbox: Bbox
    apr_ids: List[str]
    point: Bbox
    county: str
    mapid: List[str]
    name: str
    resources: Resource
    nearest_town: str
    id: str
    invasive_species: List[str] = []


class Access(OnkiaBaseModel):
    lake_access_comments: str = ""
    location: str = ""
    owner_type_id: str
    public_use_auth_code: str
    access_type_id: str


class Limit(OnkiaBaseModel):
    fish_count: List[List[int]]
    maximum_length: int
    minimum_length: int


class Length(OnkiaBaseModel):
    wae: Optional[Limit] = None
    blg: Optional[Limit] = None
    blb: Optional[Limit] = None
    pmk: Optional[Limit] = None
    yeb: Optional[Limit] = None
    gsf: Optional[Limit] = None
    cap: Optional[Limit] = None
    tlc: Optional[Limit] = None
    yep: Optional[Limit] = None
    bof: Optional[Limit] = None
    lmb: Optional[Limit] = None
    hsf: Optional[Limit] = None
    blc: Optional[Limit] = None
    brb: Optional[Limit] = None
    rkb: Optional[Limit] = None
    nop: Optional[Limit] = None
    gos: Optional[Limit] = None
    wts: Optional[Limit] = None


class FishCatchSummary(OnkiaBaseModel):
    quartile_count: str
    cpue: Union[float, str] = Field(alias="CPUE")
    species: str
    total_catch: int
    total_weight: Union[float, str]
    quartile_weight: str
    gear: str
    average_weight: Union[float, str]
    gear_count: Union[float, int]


class Survey(OnkiaBaseModel):
    survey_sub_type: str
    lengths: Optional[Union[Length, dict]] = None
    survey_type: str
    survey_date: str
    narrative: str = ""
    survey_id: int
    fish_catch_summaries: List[FishCatchSummary] = []
    header_info: List[Optional[str]] = []


class SurveyOverview(OnkiaBaseModel):
    littoral_acres: float
    accesses: List[Access] = []
    mean_depth_feet: Optional[float] = None
    max_depth_feet: int
    area_acres: float
    lake_name: str = ""
    average_water_clarity: Union[float, str] = 0.0
    surveys: List[Survey] = []
    dow_number: int = 0
    shore_length_miles: float
    sampled_plants: List = []
    water_clarity: List[List[Union[float, str]]] = []
    office_code: str = ""


class WaterLevelReading(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    chr_id: str = Field(alias="CHR_ID")
    elevation: Union[float, str] = Field(alias="ELEVATION")
    read_date: str = Field(alias="READ_DATE")
    datum_adj: str = Field(alias="DATUM_ADJ")


class WaterTempPreference(OnkiaBaseModel):
    species: str
    lower_avoidance: Optional[float] = None
    optimum: str
    upper_avoidance: Optional[float] = None


class DepthPreference(OnkiaBaseModel):
    species: str
    cold_range: str
    optimal_range: str
    warm_range: str


class BathymetryContour(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    depth_ft: float
    geometry: Optional[Dict] = None
