"""
Run this notebook:
    marimo edit notebooks/02_species_analysis.py   # interactive UI
    python notebooks/02_species_analysis.py        # CI / script mode (no data fetched)
"""

import marimo

__generated_with = "0.11.0"
app = marimo.App(width="medium", title="Wright County — Species Catch Trends")


@app.cell
def _setup_path():
    import sys
    import pathlib

    _src = pathlib.Path(__file__).parent.parent / "src"
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))
    return


@app.cell
def _imports():
    import marimo as mo
    import pandas as pd
    return mo, pd


@app.cell
def _header(mo):
    mo.md("""
    # Wright County — Species Catch Trends

    Explore fish catch trends over time for target species across Wright County lakes,
    using occurrence records from the MN DNR Fishes of Minnesota (FOM) database.

    **Target species:** Largemouth Bass · Northern Pike · Sunfish · Crappie · Walleye
    """)
    return


@app.cell
def _controls(mo):
    WRIGHT_COUNTY_LAKES = [
        "Clearwater", "Sylvia", "Pulaski", "Charlotte", "Beebe",
        "Ann", "Constance", "Martha", "Ida", "Frances",
        "Twin", "Eagle", "Bass", "Pleasant", "French",
        "Silver", "Waverly", "Pelican", "Lime", "Indian",
    ]

    # Target species groups
    TARGET_SPECIES = {
        "Largemouth Bass": ["largemouth bass"],
        "Northern Pike": ["northern pike"],
        "Sunfish (all)": ["bluegill", "pumpkinseed", "green sunfish", "hybrid sunfish", "sunfish"],
        "Black Crappie": ["black crappie"],
        "White Crappie": ["white crappie"],
        "Walleye": ["walleye"],
    }

    lake_select = mo.ui.dropdown(
        options=WRIGHT_COUNTY_LAKES,
        value="Clearwater",
        label="Lake",
    )
    species_select = mo.ui.multiselect(
        options=list(TARGET_SPECIES.keys()),
        value=["Largemouth Bass", "Northern Pike", "Walleye", "Black Crappie"],
        label="Species to display",
    )
    reload_btn = mo.ui.run_button(label="Load species data")

    mo.vstack([
        mo.hstack([lake_select, reload_btn], gap=2, align="end"),
        species_select,
    ])
    return WRIGHT_COUNTY_LAKES, TARGET_SPECIES, lake_select, reload_btn, species_select


@app.cell
def _load_data(mo, lake_select, reload_btn):
    mo.stop(
        not reload_btn.value,
        mo.callout(
            mo.md("Select a lake and species above, then click **Load species data**."),
            kind="info",
        ),
    )

    from onkia.dnr_client import MnDnrLakeTopographyService

    _svc = MnDnrLakeTopographyService()
    raw_species = _svc.get_species(lake_select.value, county_id=86, state_id=1)
    lake_name_loaded = lake_select.value

    return raw_species, lake_name_loaded


@app.cell
def _build_df(mo, pd, raw_species, lake_name_loaded):
    import warnings

    if not raw_species:
        mo.callout(mo.md("No species data returned. Try a different lake."), kind="warn")
        species_df = pd.DataFrame()
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _df = pd.DataFrame(raw_species)

        # Strip leading/trailing spaces from all column names
        _df.columns = [c.strip() for c in _df.columns]

        # Parse key columns
        _df["catch"] = pd.to_numeric(_df.get("Catch", _df.get(" Catch", pd.Series(dtype=float))), errors="coerce")
        _df["date"] = pd.to_datetime(
            _df.get("Date", _df.get(" Date", pd.Series(dtype=str))),
            errors="coerce",
        )
        _df["common_name"] = (
            _df.get("Common Name", _df.get(" Common Name", pd.Series(dtype=str)))
            .str.strip()
            .str.lower()
        )
        _df["year"] = _df["date"].dt.year

        species_df = _df[["date", "year", "common_name", "catch"]].dropna(subset=["date", "common_name"])

    return species_df,


@app.cell
def _filter_df(mo, pd, species_df, species_select, TARGET_SPECIES):
    import itertools

    if species_df.empty:
        filtered_df = pd.DataFrame()
        mo.md("No data to display.")
    else:
        # Build set of lowercase species names matching the selected groups
        _selected_names = set(itertools.chain.from_iterable(
            TARGET_SPECIES.get(grp, []) for grp in species_select.value
        ))

        # Map each row's common_name to its group label
        _name_to_group = {}
        for _group, _names in TARGET_SPECIES.items():
            for _n in _names:
                _name_to_group[_n] = _group

        filtered_df = species_df[species_df["common_name"].isin(_selected_names)].copy()
        filtered_df["species_group"] = filtered_df["common_name"].map(_name_to_group)

    return filtered_df,


@app.cell
def _trend_chart(mo, pd, filtered_df, lake_name_loaded):
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    if filtered_df.empty:
        mo.md("No matching species records to plot.")
    else:
        # Aggregate by year + species_group: sum catch per survey year
        _agg = (
            filtered_df.groupby(["year", "species_group"], as_index=False)["catch"]
            .sum()
            .sort_values("year")
        )

        _fig, _ax = plt.subplots(figsize=(12, 5))

        for _grp, _data in _agg.groupby("species_group"):
            _ax.plot(_data["year"], _data["catch"], marker="o", label=_grp, linewidth=2)

        _ax.set_title(f"Species Catch Trends — {lake_name_loaded} Lake (Wright County)", fontsize=14)
        _ax.set_xlabel("Year")
        _ax.set_ylabel("Total Catch (all gear types)")
        _ax.legend(title="Species", bbox_to_anchor=(1.01, 1), loc="upper left")
        _ax.grid(True, alpha=0.3)
        _fig.tight_layout()

        plt.gca()
        _fig
    return


@app.cell
def _summary_table(mo, pd, filtered_df):
    if filtered_df.empty:
        return

    _summary = (
        filtered_df.groupby("species_group")
        .agg(
            surveys=("date", "nunique"),
            total_catch=("catch", "sum"),
            avg_catch_per_survey=("catch", "mean"),
            max_catch=("catch", "max"),
            first_recorded=("date", "min"),
            last_recorded=("date", "max"),
        )
        .reset_index()
        .rename(columns={"species_group": "Species Group"})
        .sort_values("total_catch", ascending=False)
    )

    _summary["avg_catch_per_survey"] = _summary["avg_catch_per_survey"].round(1)
    _summary["first_recorded"] = _summary["first_recorded"].dt.strftime("%Y-%m-%d")
    _summary["last_recorded"] = _summary["last_recorded"].dt.strftime("%Y-%m-%d")

    mo.vstack([
        mo.md("## Summary by Species Group"),
        mo.ui.table(_summary),
    ])
    return


@app.cell
def _raw_records_table(mo, pd, filtered_df):
    if filtered_df.empty:
        return

    _display = filtered_df.copy()
    _display["date"] = _display["date"].dt.strftime("%Y-%m-%d")
    _display = _display.sort_values("date", ascending=False)

    mo.vstack([
        mo.md(f"## All Matching Records ({len(_display)} rows)"),
        mo.ui.table(_display[["date", "species_group", "common_name", "catch"]], pagination=True),
    ])
    return


if __name__ == "__main__":
    app.run()
