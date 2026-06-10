from __future__ import annotations

import csv
import io
import logging
from typing import Any, Dict, List, Mapping, Optional, Protocol

import requests

from onkia.models import Lake, SurveyOverview, WaterLevelReading


class DnrApiUnavailableError(Exception):
    """Raised when the DNR API cannot be reached or returns an unparseable response."""


# MN DNR county IDs are the 1-based alphabetical index of Minnesota's 87
# counties (verified against search.cgi county_id values).
MN_COUNTIES: Dict[int, str] = {
    1: "Aitkin", 2: "Anoka", 3: "Becker", 4: "Beltrami", 5: "Benton",
    6: "Big Stone", 7: "Blue Earth", 8: "Brown", 9: "Carlton", 10: "Carver",
    11: "Cass", 12: "Chippewa", 13: "Chisago", 14: "Clay", 15: "Clearwater",
    16: "Cook", 17: "Cottonwood", 18: "Crow Wing", 19: "Dakota", 20: "Dodge",
    21: "Douglas", 22: "Faribault", 23: "Fillmore", 24: "Freeborn",
    25: "Goodhue", 26: "Grant", 27: "Hennepin", 28: "Houston", 29: "Hubbard",
    30: "Isanti", 31: "Itasca", 32: "Jackson", 33: "Kanabec", 34: "Kandiyohi",
    35: "Kittson", 36: "Koochiching", 37: "Lac qui Parle", 38: "Lake",
    39: "Lake of the Woods", 40: "Le Sueur", 41: "Lincoln", 42: "Lyon",
    43: "McLeod", 44: "Mahnomen", 45: "Marshall", 46: "Martin", 47: "Meeker",
    48: "Mille Lacs", 49: "Morrison", 50: "Mower", 51: "Murray",
    52: "Nicollet", 53: "Nobles", 54: "Norman", 55: "Olmsted",
    56: "Otter Tail", 57: "Pennington", 58: "Pine", 59: "Pipestone",
    60: "Polk", 61: "Pope", 62: "Ramsey", 63: "Red Lake", 64: "Redwood",
    65: "Renville", 66: "Rice", 67: "Rock", 68: "Roseau", 69: "St. Louis",
    70: "Scott", 71: "Sherburne", 72: "Sibley", 73: "Stearns", 74: "Steele",
    75: "Stevens", 76: "Swift", 77: "Todd", 78: "Traverse", 79: "Wabasha",
    80: "Wadena", 81: "Waseca", 82: "Washington", 83: "Watonwan",
    84: "Wilkin", 85: "Winona", 86: "Wright", 87: "Yellow Medicine",
}


class LakeTopographyService(Protocol):
    def get_lake(self, name: str, county_id: Optional[int] = 86) -> Optional[Lake]:
        ...

    def get_survey(self, lake_id: str) -> Optional[SurveyOverview]:
        ...

    def get_species(self, name: str, county_id: int = 86, state_id: int = 1) -> List[Dict[str, Any]]:
        ...

    def get_water_levels(self, lake_id: str) -> List[WaterLevelReading]:
        ...

    def get_stocking(self, lake_id: str) -> Any:
        ...


def _csv_to_dicts(data: str, delimiter: str = ",") -> List[Dict[str, str]]:
    csvio = io.StringIO(data, newline="")
    return list(csv.DictReader(csvio, delimiter=delimiter))


class MnDnrLakeTopographyService:
    # NOTE: maps2.dnr.state.mn.us no longer resolves and the old
    # services.dnr.state.mn.us/api/lakefinder/by_name endpoint returns 403.
    # The LakeFinder CGIs now live on maps.dnr.state.mn.us.
    _API_BY_NAME_AND_COUNTY = "https://maps.dnr.state.mn.us/cgi-bin/lakefinder/search.cgi"
    _API_MAP_HOST = "https://maps.dnr.state.mn.us/cgi-bin/lakefinder/detail.cgi"
    _API_SPECIES_REPORT = "https://maps2.dnr.state.mn.us/cgi-bin/fom.cgi"
    _API_WATER_LEVELS_REPORT = "https://files.dnr.state.mn.us/cgi-bin/lk_levels_dump.cgi"
    _API_STOCKING_REPORT = "https://files.dnr.state.mn.us/cgi-bin/lk_stocking.cgi"

    def __init__(self, logger: Optional[logging.Logger] = None):
        self._logger = logger or logging.getLogger(__name__)

    def _send_request(
        self,
        endpoint: str,
        params: Optional[Mapping[str, str]],
    ) -> Any:
        self._logger.debug("Sending request to %s", endpoint)
        try:
            response = requests.get(endpoint, params=params, timeout=15)
        except requests.exceptions.ConnectionError as exc:
            self._logger.warning("Connection error for %s", endpoint)
            raise DnrApiUnavailableError(f"Connection error contacting DNR API: {endpoint}") from exc
        except requests.exceptions.Timeout as exc:
            self._logger.warning("Timeout for %s", endpoint)
            raise DnrApiUnavailableError(f"Request timed out contacting DNR API: {endpoint}") from exc
        try:
            response_json = response.json()
        except requests.exceptions.JSONDecodeError as exc:
            self._logger.warning("JSON decode error for %s", endpoint)
            raise DnrApiUnavailableError(f"Invalid JSON response from DNR API: {endpoint}") from exc
        return response_json.get("results", response_json.get("result"))

    def _send_csv_request(
        self,
        endpoint: str,
        params: Optional[Mapping[str, str]],
        delimiter: str = ",",
    ) -> List[Dict[str, str]]:
        self._logger.debug("Sending request to %s", endpoint)
        try:
            raw_response = requests.get(endpoint, params=params, timeout=15)
        except requests.exceptions.ConnectionError:
            self._logger.warning("Connection error for %s", endpoint)
            return []
        except requests.exceptions.Timeout:
            self._logger.warning("Timeout for %s", endpoint)
            return []
        return _csv_to_dicts(raw_response.text, delimiter=delimiter)

    def get_lake(self, name: str, county_id: Optional[int] = 86) -> Optional[Lake]:
        """Search LakeFinder by name.

        ``county_id`` scopes the search to one county (DNR county IDs, see
        ``MN_COUNTIES``); pass ``None`` to search statewide. Names like
        "Big Swan" exist in several counties, so county scoping picks the
        intended lake.
        """
        params = {"name": name}
        if county_id is not None:
            params["county"] = str(county_id)
        results = self._send_request(self._API_BY_NAME_AND_COUNTY, params=params)
        if not results:
            return None
        try:
            candidates = list(results)
        except TypeError:
            return None
        if not candidates:
            return None
        # Prefer an exact (case-insensitive) name match over a prefix match.
        raw = next(
            (c for c in candidates if str(c.get("name", "")).lower() == name.lower()),
            candidates[0],
        )
        try:
            return Lake.model_validate(raw)
        except Exception:
            self._logger.warning("Failed to validate lake data for %s", name)
            return None

    def get_survey(self, lake_id: str) -> Optional[SurveyOverview]:
        raw = self._send_request(
            self._API_MAP_HOST,
            params={"type": "lake_survey", "id": lake_id},
        )
        if not raw:
            return None
        try:
            return SurveyOverview.model_validate(raw)
        except Exception:
            self._logger.warning("Failed to validate survey data for lake %s", lake_id)
            return None

    def get_species(
        self,
        name: str,
        county_id: int = 86,
        state_id: int = 1,
    ) -> List[Dict[str, Any]]:
        return self._send_csv_request(
            self._API_SPECIES_REPORT,
            params={
                "wb_name": name,
                "county_id": str(county_id),
                "state_id": str(state_id),
                "mode": "getdata",
                "name": "csv",
                "wb_type": "Lake",
                "orderby": "",
            },
        )

    def get_water_levels(self, lake_id: str) -> List[WaterLevelReading]:
        raw_rows = self._send_csv_request(
            self._API_WATER_LEVELS_REPORT,
            params={"id": lake_id, "format": "csv"},
        )
        return [WaterLevelReading.model_validate(row) for row in raw_rows]

    def get_depth(self, map_id: str) -> None:
        return None

    def get_stocking(self, lake_id: str) -> Any:
        import xml.etree.ElementTree as ET

        try:
            raw_response = requests.get(
                self._API_STOCKING_REPORT,
                params={"downum": lake_id},
                timeout=15,
            )
        except requests.exceptions.ConnectionError:
            self._logger.warning("Connection error for stocking API (lake %s)", lake_id)
            return None
        except requests.exceptions.Timeout:
            self._logger.warning("Timeout for stocking API (lake %s)", lake_id)
            return None
        try:
            root = ET.fromstring(raw_response.text)
        except ET.ParseError:
            self._logger.warning("XML parse error for stocking API (lake %s)", lake_id)
            return None
        self._logger.info("Stocking data retrieved for lake %s", lake_id)
        return root
