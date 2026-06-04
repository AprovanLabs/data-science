import sys
from pathlib import Path

import pytest
import pandas as pd
import great_expectations as gx
from great_expectations.expectations import (
    ExpectColumnToExist,
    ExpectColumnValuesToNotBeNull,
    ExpectColumnValuesToMatchRegex,
    ExpectColumnValuesToBeInSet,
    ExpectColumnValuesToBeBetween,
    ExpectColumnValuesToBeInTypeList,
    ExpectColumnValuesToBeOfType,
    ExpectColumnValuesToMatchStrftimeFormat,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

GEAR_TYPES = [
    "Standard gill nets",
    "Standard trap nets",
    "Gill net (all)",
    "Trap net (all)",
    "Electroshockers (all)",
    "Small mesh fyke net",
    "Large mesh fyke net",
    "Seine",
    "Rotenone",
    "Other",
]


def _load_fixture_lake_metadata(fixtures_dir: Path = FIXTURES_DIR) -> pd.DataFrame:
    import json
    path = fixtures_dir / "lake_metadata.json"
    with open(path) as f:
        raw = json.load(f)
    records = []
    for lake in raw if isinstance(raw, list) else [raw]:
        morph = lake.get("morphology", {})
        records.append({
            "id": lake.get("id"),
            "name": lake.get("name"),
            "county": lake.get("county"),
            "county_id": lake.get("county_id"),
            "area": morph.get("area"),
            "max_depth": morph.get("max_depth"),
            "mean_depth": morph.get("mean_depth"),
            "shore_length": morph.get("shore_length"),
            "littoral_area": morph.get("littoral_area"),
        })
    return pd.DataFrame(records)


def _load_fixture_fish_survey(fixtures_dir: Path = FIXTURES_DIR) -> pd.DataFrame:
    import json
    path = fixtures_dir / "fish_survey.json"
    with open(path) as f:
        raw = json.load(f)
    records = []
    for survey in raw if isinstance(raw, list) else [raw]:
        for summary in survey.get("fishCatchSummaries", []):
            records.append({
                "species": summary.get("species"),
                "total_catch": summary.get("totalCatch"),
                "total_weight": summary.get("totalWeight"),
                "cpue": summary.get("CPUE"),
                "average_weight": summary.get("averageWeight"),
                "gear": summary.get("gear"),
                "gear_count": summary.get("gearCount"),
                "quartile_count": summary.get("quartileCount"),
                "quartile_weight": summary.get("quartileWeight"),
                "survey_date": survey.get("survey_date"),
                "survey_type": survey.get("survey_type"),
            })
    return pd.DataFrame(records)


def _load_fixture_species_occurrence(fixtures_dir: Path = FIXTURES_DIR) -> pd.DataFrame:
    path = fixtures_dir / "species_occurrence.csv"
    df = pd.read_csv(path, dtype=str)
    df[" Catch"] = pd.to_numeric(df.get(" Catch", pd.Series(dtype=float)), errors="coerce")
    return df


def _load_fixture_water_levels(fixtures_dir: Path = FIXTURES_DIR) -> pd.DataFrame:
    path = fixtures_dir / "water_levels.csv"
    df = pd.read_csv(path, dtype=str)
    df["ELEVATION"] = pd.to_numeric(df["ELEVATION"], errors="coerce")
    return df


FIXTURE_LOADERS = {
    "lake_metadata": _load_fixture_lake_metadata,
    "fish_survey": _load_fixture_fish_survey,
    "species_occurrence": _load_fixture_species_occurrence,
    "water_levels": _load_fixture_water_levels,
}


def _build_lake_metadata_suite(context):
    suite = context.suites.add(gx.ExpectationSuite(name="lake_metadata"))
    for col in ["id", "name", "county", "county_id", "area", "max_depth"]:
        suite.add_expectation(ExpectColumnToExist(column=col))
    for col in ["id", "name", "county", "county_id"]:
        suite.add_expectation(ExpectColumnValuesToNotBeNull(column=col))
    suite.add_expectation(ExpectColumnValuesToMatchRegex(
        column="id", regex=r"^86\d{6}$",
    ))
    suite.add_expectation(ExpectColumnValuesToBeInSet(
        column="county", value_set=["Wright"],
    ))
    suite.add_expectation(ExpectColumnValuesToBeInTypeList(
        column="county_id", type_list=["int64", "int32", "int"],
    ))
    for col in ["area", "max_depth"]:
        suite.add_expectation(ExpectColumnValuesToBeBetween(
            column=col, min_value=0,
        ))
    return suite


def _build_fish_survey_suite(context):
    suite = context.suites.add(gx.ExpectationSuite(name="fish_survey"))
    for col in ["species", "total_catch", "total_weight", "gear", "cpue", "survey_date"]:
        suite.add_expectation(ExpectColumnToExist(column=col))
    for col in ["species", "total_catch", "gear"]:
        suite.add_expectation(ExpectColumnValuesToNotBeNull(column=col))
    suite.add_expectation(ExpectColumnValuesToMatchRegex(
        column="species", regex=r"^[A-Z]{3}$",
    ))
    suite.add_expectation(ExpectColumnValuesToBeBetween(
        column="total_catch", min_value=0,
    ))
    suite.add_expectation(ExpectColumnValuesToBeBetween(
        column="total_weight", min_value=0,
    ))
    suite.add_expectation(ExpectColumnValuesToBeInSet(
        column="gear", value_set=GEAR_TYPES,
    ))
    suite.add_expectation(ExpectColumnValuesToMatchStrftimeFormat(
        column="survey_date", strftime_format="%Y-%m-%d",
    ))
    return suite


def _build_species_occurrence_suite(context):
    suite = context.suites.add(gx.ExpectationSuite(name="species_occurrence"))
    required_cols = [
        "FOM ID", " Scientific Name", " Common Name",
        " DOW Number", " Catch", " Date", " Gear",
        " Waterbody Type", " County",
    ]
    for col in required_cols:
        suite.add_expectation(ExpectColumnToExist(column=col))
    non_null_cols = [
        "FOM ID", " Scientific Name", " Common Name",
        " DOW Number", " Date",
    ]
    for col in non_null_cols:
        suite.add_expectation(ExpectColumnValuesToNotBeNull(column=col))
    suite.add_expectation(ExpectColumnValuesToMatchRegex(
        column=" DOW Number", regex=r"^\d{8}$",
    ))
    suite.add_expectation(ExpectColumnValuesToBeInSet(
        column=" Waterbody Type", value_set=["Lake", "Stream", "River"],
    ))
    suite.add_expectation(ExpectColumnValuesToBeInSet(
        column=" County", value_set=["Wright"],
    ))
    suite.add_expectation(ExpectColumnValuesToBeOfType(
        column="FOM ID", type_="str",
    ))
    suite.add_expectation(ExpectColumnValuesToMatchStrftimeFormat(
        column=" Date", strftime_format="%Y-%m-%d",
    ))
    return suite


def _build_water_levels_suite(context):
    suite = context.suites.add(gx.ExpectationSuite(name="water_levels"))
    for col in ["CHR_ID", "ELEVATION", "READ_DATE", "DATUM_ADJ"]:
        suite.add_expectation(ExpectColumnToExist(column=col))
    for col in ["CHR_ID", "ELEVATION", "READ_DATE"]:
        suite.add_expectation(ExpectColumnValuesToNotBeNull(column=col))
    suite.add_expectation(ExpectColumnValuesToMatchRegex(
        column="CHR_ID", regex=r"^\d{8}$",
    ))
    suite.add_expectation(ExpectColumnValuesToMatchStrftimeFormat(
        column="READ_DATE", strftime_format="%Y-%m-%d",
    ))
    suite.add_expectation(ExpectColumnValuesToBeOfType(
        column="ELEVATION", type_="float",
    ))
    return suite


SUITE_BUILDERS = {
    "lake_metadata": _build_lake_metadata_suite,
    "fish_survey": _build_fish_survey_suite,
    "species_occurrence": _build_species_occurrence_suite,
    "water_levels": _build_water_levels_suite,
}


def _validate_fixture(suite_name: str) -> dict:
    context = gx.get_context(mode="ephemeral")
    builder = SUITE_BUILDERS[suite_name]
    suite = builder(context)
    loader = FIXTURE_LOADERS[suite_name]
    df = loader()

    ds = context.data_sources.add_pandas(name=f"ds_{suite_name}")
    asset = ds.add_dataframe_asset(name=f"asset_{suite_name}")
    batch_def = asset.add_batch_definition_whole_dataframe(name=f"batch_{suite_name}")
    vd = gx.ValidationDefinition(
        name=f"vd_{suite_name}",
        data=batch_def,
        suite=suite,
    )
    result = vd.run(batch_parameters={"dataframe": df})
    return {
        "suite": suite_name,
        "success": result.success,
        "statistics": result.statistics,
    }


@pytest.fixture(params=list(SUITE_BUILDERS.keys()))
def suite_name(request):
    return request.param


def test_expectation_suite_builds(suite_name):
    context = gx.get_context(mode="ephemeral")
    builder = SUITE_BUILDERS[suite_name]
    suite = builder(context)
    assert suite.name == suite_name
    assert len(suite.expectation_configurations) > 0


def test_fixture_data_validates_against_suite(suite_name):
    result = _validate_fixture(suite_name)
    assert result["success"], (
        f"Validation failed for suite '{suite_name}': "
        f"{result['statistics']['unsuccessful_expectations']} expectations failed"
    )


def test_validate_all_suites():
    sys_path = str(Path(__file__).resolve().parent.parent / "scripts")
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    from validate_data import validate_all

    results = validate_all(data_dir=FIXTURES_DIR)
    for r in results:
        assert r["success"], (
            f"Validation failed for suite '{r['suite']}': "
            f"{r['statistics']['unsuccessful_expectations']} expectations failed"
        )
