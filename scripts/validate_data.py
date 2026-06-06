from pathlib import Path

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

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _load_lake_metadata(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    import json
    path = data_dir / "lake_metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Lake metadata not found: {path}")
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


def _load_fish_survey(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    import json
    path = data_dir / "fish_survey.json"
    if not path.exists():
        raise FileNotFoundError(f"Fish survey data not found: {path}")
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


def _load_species_occurrence(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    path = data_dir / "species_occurrence.csv"
    if not path.exists():
        raise FileNotFoundError(f"Species occurrence data not found: {path}")
    df = pd.read_csv(path, dtype=str)
    df[" Catch"] = pd.to_numeric(df.get(" Catch", pd.Series(dtype=float)), errors="coerce")
    return df


def _load_water_levels(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    path = data_dir / "water_levels.csv"
    if not path.exists():
        raise FileNotFoundError(f"Water level data not found: {path}")
    df = pd.read_csv(path, dtype=str)
    df["ELEVATION"] = pd.to_numeric(df["ELEVATION"], errors="coerce")
    return df


DATA_LOADERS = {
    "lake_metadata": _load_lake_metadata,
    "fish_survey": _load_fish_survey,
    "species_occurrence": _load_species_occurrence,
    "water_levels": _load_water_levels,
}

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


def validate_suite(suite_name: str, data_dir: Path = DATA_DIR) -> dict:
    context = gx.get_context(mode="ephemeral")
    builder = SUITE_BUILDERS[suite_name]
    suite = builder(context)
    loader = DATA_LOADERS[suite_name]
    df = loader(data_dir)

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
        "result": result,
    }


def validate_all(data_dir: Path = DATA_DIR) -> list[dict]:
    results = []
    for suite_name in SUITE_BUILDERS:
        results.append(validate_suite(suite_name, data_dir))
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate DNR data using Great Expectations")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help=f"Directory containing cached data files (default: {DATA_DIR})",
    )
    args = parser.parse_args()
    results = validate_all(data_dir=args.data_dir)
    all_passed = True
    for r in results:
        status = "PASSED" if r["success"] else "FAILED"
        stats = r["statistics"]
        print(f"\n{r['suite']}: {status}")
        print(f"  Evaluated: {stats['evaluated_expectations']}  "
              f"Successful: {stats['successful_expectations']}  "
              f"Unsuccessful: {stats['unsuccessful_expectations']}  "
              f"Skipped: {stats.get('success_percent', 0)}")
        if not r["success"]:
            all_passed = False

    print()
    if all_passed:
        print("All validations passed.")
        return 0
    else:
        print("Some validations failed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
