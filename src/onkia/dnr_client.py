from __future__ import annotations

import csv
import io
import logging
from typing import Any, Dict, List, Mapping, Optional, Protocol

import requests

from onkia.models import Lake, SurveyOverview, WaterLevelReading


class DnrApiUnavailableError(Exception):
    """Raised when the DNR API cannot be reached or returns an unparseable response."""


class LakeTopographyService(Protocol):
    def get_lake(self, name: str, county_id: int = 86) -> Optional[Lake]:
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
    _API_BY_NAME_AND_COUNTY = "http://services.dnr.state.mn.us/api/lakefinder/by_name/v1"
    _API_MAP_HOST = "https://maps2.dnr.state.mn.us/cgi-bin/lakefinder/detail.cgi"
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

    def get_lake(self, name: str, county_id: int = 86) -> Optional[Lake]:
        results = self._send_request(
            self._API_BY_NAME_AND_COUNTY,
            params={"name": name, "county": str(county_id)},
        )
        if not results:
            return None
        try:
            raw = next(iter(results), None)
        except (TypeError, StopIteration):
            return None
        if raw is None:
            return None
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
