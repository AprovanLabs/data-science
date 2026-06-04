from __future__ import annotations

import csv
import io
import logging
from typing import Any, Dict, List, Mapping, Optional, Protocol

import requests

from onkia.models import Lake, SurveyOverview, WaterLevelReading


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
        response = requests.get(endpoint, params=params)
        response_json = response.json()
        return response_json.get("results", response_json.get("result"))

    def _send_csv_request(
        self,
        endpoint: str,
        params: Optional[Mapping[str, str]],
        delimiter: str = ",",
    ) -> List[Dict[str, str]]:
        self._logger.debug("Sending request to %s", endpoint)
        raw_response = requests.get(endpoint, params=params)
        return _csv_to_dicts(raw_response.text, delimiter=delimiter)

    def get_lake(self, name: str, county_id: int = 86) -> Optional[Lake]:
        results = self._send_request(
            self._API_BY_NAME_AND_COUNTY,
            params={"name": name, "county": str(county_id)},
        )
        if not results:
            return None
        raw = next(iter(results), None)
        if raw is None:
            return None
        return Lake.model_validate(raw)

    def get_survey(self, lake_id: str) -> Optional[SurveyOverview]:
        raw = self._send_request(
            self._API_MAP_HOST,
            params={"type": "lake_survey", "id": lake_id},
        )
        if not raw:
            return None
        return SurveyOverview.model_validate(raw)

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

        raw_response = requests.get(
            self._API_STOCKING_REPORT,
            params={"downum": lake_id},
        )
        root = ET.fromstring(raw_response.text)
        self._logger.info("Stocking data retrieved for lake %s", lake_id)
        return root
