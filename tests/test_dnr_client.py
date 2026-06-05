from __future__ import annotations

import json
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from onkia.dnr_client import MnDnrLakeTopographyService
from onkia.models import (
    BathymetryContour,
    DepthPreference,
    FishCatchSummary,
    Lake,
    Morphology,
    SpecialFishingReg,
    Survey,
    SurveyOverview,
    WaterLevelReading,
    WaterTempPreference,
)
from onkia.water_temp import WATER_TEMP_PREFERENCES, DEPTH_PREFERENCES


@pytest.fixture
def service():
    return MnDnrLakeTopographyService()


MOCK_LAKE_RAW = {
    "morphology": {
        "max_depth": 73,
        "littoral_area": 1595.6,
        "shore_length": 34.83,
        "area": 3186.89,
        "mean_depth": 19.2,
    },
    "point": {"epsg:26915": [412312, 5017445], "epsg:4326": [-94.118377, 45.305151]},
    "id": "86025200",
    "county_id": 86,
    "nearest_town": "Annandale",
    "bbox": {
        "epsg:4326": [-94.167073, 45.276006, -94.069682, 45.334296],
        "epsg:26915": [408510, 5014180, 416113, 5020710],
    },
    "name": "Clearwater",
    "apr_ids": ["2009"],
    "invasiveSpecies": ["Eurasian watermilfoil", "zebra mussel"],
    "resources": {
        "iceIn": 1,
        "waterLevels": 1,
        "lakeHealth": 1,
        "waterAccess": 1,
        "iceOut": 1,
        "waterQuality": 0,
        "fca": 1,
        "specialFishingRegs": 1,
        "lakeMap": 1,
        "sentinelLake": 0,
        "fishStocking": 1,
        "lakeSurvey": 1,
    },
    "mapid": ["C0779"],
    "border": "",
    "fishSpecies": ["black crappie", "bluegill", "walleye"],
    "county": "Wright",
    "notes": "",
    "specialFishingRegs": [
        {
            "location": "Including connected Clearwater and Grass lakes",
            "locDisplayType": 1,
            "regs": [
                {"species": ["Crappie"], "text": "Daily limit five."},
                {"species": ["Sunfish"], "text": "Daily limit ten."},
            ],
        }
    ],
}

MOCK_SURVEY_RAW = {
    "averageWaterClarity": "5.8",
    "waterClarity": [["07/21/2014", "4.5"], ["08/01/2022", "6.5"]],
    "littoralAcres": 1595.6,
    "sampledPlants": [],
    "areaAcres": 3186.89,
    "maxDepthFeet": 73,
    "shoreLengthMiles": 34.83,
    "officeCode": "F315",
    "dowNumber": 86025200,
    "surveys": [
        {
            "surveySubType": "",
            "surveyType": "Standard Survey",
            "surveyDate": "2014-07-21",
            "narrative": "",
            "surveyId": 12345,
            "fishCatchSummaries": [
                {
                    "gearCount": 20,
                    "averageWeight": "0.55",
                    "totalWeight": 6.1,
                    "gear": "Standard trap nets",
                    "CPUE": "0.55",
                    "quartileCount": "0.3-2.1",
                    "quartileWeight": "0.4-0.8",
                    "species": "BLB",
                    "totalCatch": 11,
                }
            ],
            "headerInfo": [None],
        }
    ],
}

MOCK_WATER_LEVELS_CSV = (
    "CHR_ID,ELEVATION,READ_DATE,DATUM_ADJ\r\n"
    "86025200,990.58,1932-01-01,NGVD 29\r\n"
    "86025200,991.05,1939-01-01,NGVD 29\r\n"
)

MOCK_SPECIES_CSV = (
    "FOM ID,Scientific Name,Common Name,Gear,Catch\r\n"
    "433927,Ambloplites rupestris,rock bass,Gill net (all),42\r\n"
)


class TestGetLake:
    @patch("onkia.dnr_client.requests.get")
    def test_returns_lake_model(self, mock_get, service):
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": [MOCK_LAKE_RAW]}
        mock_get.return_value = mock_response

        lake = service.get_lake("Clearwater", county_id=86)

        assert isinstance(lake, Lake)
        assert lake.name == "Clearwater"
        assert lake.county_id == 86
        assert lake.id == "86025200"
        assert lake.morphology.max_depth == 73
        assert lake.morphology.area == 3186.89

    @patch("onkia.dnr_client.requests.get")
    def test_passes_correct_params(self, mock_get, service):
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": [MOCK_LAKE_RAW]}
        mock_get.return_value = mock_response

        service.get_lake("Clearwater", county_id=86)

        mock_get.assert_called_once_with(
            MnDnrLakeTopographyService._API_BY_NAME_AND_COUNTY,
            params={"name": "Clearwater", "county": "86"},
            timeout=15,
        )

    @patch("onkia.dnr_client.requests.get")
    def test_returns_none_when_empty_results(self, mock_get, service):
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_get.return_value = mock_response

        assert service.get_lake("Nonexistent") is None

    @patch("onkia.dnr_client.requests.get")
    def test_handles_invasive_species(self, mock_get, service):
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": [MOCK_LAKE_RAW]}
        mock_get.return_value = mock_response

        lake = service.get_lake("Clearwater")
        assert "Eurasian watermilfoil" in lake.invasive_species

    @patch("onkia.dnr_client.requests.get")
    def test_handles_special_fishing_regs(self, mock_get, service):
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": [MOCK_LAKE_RAW]}
        mock_get.return_value = mock_response

        lake = service.get_lake("Clearwater")
        assert len(lake.special_fishing_regs) == 1
        assert lake.special_fishing_regs[0].regs[0].species == ["Crappie"]


class TestGetSurvey:
    @patch("onkia.dnr_client.requests.get")
    def test_returns_survey_overview(self, mock_get, service):
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": MOCK_SURVEY_RAW}
        mock_get.return_value = mock_response

        survey = service.get_survey("86025200")

        assert isinstance(survey, SurveyOverview)
        assert survey.max_depth_feet == 73
        assert survey.area_acres == 3186.89
        assert len(survey.surveys) == 1

    @patch("onkia.dnr_client.requests.get")
    def test_survey_fish_catch_summaries(self, mock_get, service):
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": MOCK_SURVEY_RAW}
        mock_get.return_value = mock_response

        survey = service.get_survey("86025200")
        catch = survey.surveys[0].fish_catch_summaries[0]
        assert isinstance(catch, FishCatchSummary)
        assert catch.species == "BLB"
        assert catch.total_catch == 11


class TestGetSpecies:
    @patch("onkia.dnr_client.requests.get")
    def test_returns_list_of_dicts(self, mock_get, service):
        mock_response = MagicMock()
        mock_response.text = MOCK_SPECIES_CSV
        mock_get.return_value = mock_response

        species = service.get_species("Clearwater")

        assert len(species) == 1
        assert species[0]["Common Name"] == "rock bass"
        assert species[0]["Catch"] == "42"


class TestGetWaterLevels:
    @patch("onkia.dnr_client.requests.get")
    def test_returns_water_level_readings(self, mock_get, service):
        mock_response = MagicMock()
        mock_response.text = MOCK_WATER_LEVELS_CSV
        mock_get.return_value = mock_response

        levels = service.get_water_levels("86025200")

        assert len(levels) == 2
        assert all(isinstance(l, WaterLevelReading) for l in levels)
        assert float(levels[0].elevation) == 990.58
        assert levels[0].datum_adj == "NGVD 29"


class TestGetDepth:
    def test_returns_none(self, service):
        assert service.get_depth("C0779") is None


class TestWaterTempPreferences:
    def test_count(self):
        assert len(WATER_TEMP_PREFERENCES) == 30

    def test_first_entry(self):
        assert WATER_TEMP_PREFERENCES[0].species == "Atlantic Salmon"
        assert WATER_TEMP_PREFERENCES[0].lower_avoidance == 45
        assert WATER_TEMP_PREFERENCES[0].optimum == "50-62"
        assert WATER_TEMP_PREFERENCES[0].upper_avoidance is None

    def test_walleye(self):
        wae = next(w for w in WATER_TEMP_PREFERENCES if w.species == "Walleye")
        assert wae.lower_avoidance == 50
        assert wae.optimum == "64-70"
        assert wae.upper_avoidance == 76

    def test_all_are_water_temp_preference(self):
        assert all(isinstance(w, WaterTempPreference) for w in WATER_TEMP_PREFERENCES)


class TestModels:
    def test_lake_from_camelCase(self):
        lake = Lake.model_validate(MOCK_LAKE_RAW)
        assert lake.name == "Clearwater"
        assert lake.county_id == 86
        assert lake.invasive_species == ["Eurasian watermilfoil", "zebra mussel"]

    def test_survey_overview_from_camelCase(self):
        survey = SurveyOverview.model_validate(MOCK_SURVEY_RAW)
        assert survey.max_depth_feet == 73
        assert survey.office_code == "F315"
        assert survey.dow_number == 86025200

    def test_water_level_reading(self):
        reading = WaterLevelReading(
            chr_id="86025200",
            elevation=990.58,
            read_date="1932-01-01",
            datum_adj="NGVD 29",
        )
        assert reading.chr_id == "86025200"
        assert reading.elevation == 990.58

    def test_fish_catch_summary_handles_string_cpue(self):
        catch = FishCatchSummary(
            quartile_count="0.3-2.1",
            cpue="0.55",
            species="BLB",
            total_catch=11,
            total_weight="6.1",
            quartile_weight="0.4-0.8",
            gear="Standard trap nets",
            average_weight="0.55",
            gear_count=20,
        )
        assert catch.cpue == "0.55"


class TestDepthPreferences:
    def test_count(self):
        assert len(DEPTH_PREFERENCES) == 9

    def test_first_entry(self):
        assert DEPTH_PREFERENCES[0].species == "Largemouth Bass"
        assert DEPTH_PREFERENCES[0].optimal_range == "4-10"

    def test_walleye_depth(self):
        wae = next(d for d in DEPTH_PREFERENCES if d.species == "Walleye")
        assert wae.optimal_range == "8-18"
        assert wae.cold_range == "20-35"
        assert wae.warm_range == "20-30"

    def test_all_are_depth_preference(self):
        assert all(isinstance(d, DepthPreference) for d in DEPTH_PREFERENCES)


class TestBathymetryContourModel:
    def test_basic(self):
        c = BathymetryContour(depth_ft=25.0)
        assert c.depth_ft == 25.0

    def test_with_geometry(self):
        c = BathymetryContour(
            depth_ft=15.0,
            geometry={"type": "Polygon", "coordinates": [[[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]]},
        )
        assert c.geometry["type"] == "Polygon"
