"""
Run this notebook:
    marimo edit notebooks/01_data_exploration.py   # interactive UI
    python notebooks/01_data_exploration.py        # CI / script mode (no data fetched)
"""

import marimo

__generated_with = "0.11.0"
app = marimo.App(width="medium", title="Wright County Lakes — Data Exploration")


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
    # Wright County Lakes — Data Exploration

    Load lake metadata, survey data, and water levels for any lake in
    **Wright County, Minnesota** (county ID 86) from the MN DNR LakeFinder API.

    Select a lake and click **Load lake data** to begin.
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

    lake_select = mo.ui.dropdown(
        options=WRIGHT_COUNTY_LAKES,
        value="Clearwater",
        label="Lake",
    )
    reload_btn = mo.ui.run_button(label="Load lake data")

    mo.hstack([lake_select, reload_btn], gap=2, align="end")
    return WRIGHT_COUNTY_LAKES, lake_select, reload_btn


@app.cell
def _load_data(mo, lake_select, reload_btn):
    mo.stop(
        not reload_btn.value,
        mo.callout(
            mo.md("Select a lake above and click **Load lake data** to fetch from the DNR API."),
            kind="info",
        ),
    )

    from onkia import MnDnrLakeTopographyService

    _svc = MnDnrLakeTopographyService()
    _name = lake_select.value

    lake = _svc.get_lake(_name, county_id=86)
    survey = _svc.get_survey(lake.id) if lake else None
    water_levels = _svc.get_water_levels(lake.id) if lake else []

    return lake, survey, water_levels


@app.cell
def _lake_summary(mo, lake):
    if lake is None:
        mo.callout(mo.md("Lake not found. Try a different name."), kind="warn")
    else:
        mo.md(f"""
        ## {lake.name}

        | Field | Value |
        |---|---|
        | County | {lake.county} |
        | Nearest Town | {lake.nearest_town} |
        | Area (acres) | {lake.morphology.area:,.1f} |
        | Max Depth (ft) | {lake.morphology.max_depth} |
        | Mean Depth (ft) | {lake.morphology.mean_depth:.1f} |
        | Shore Length (mi) | {lake.morphology.shore_length:.2f} |
        | Littoral Area (acres) | {lake.morphology.littoral_area:.1f} |
        | Fish Species | {", ".join(lake.fish_species) if lake.fish_species else "None listed"} |
        | Invasive Species | {", ".join(lake.invasive_species) if lake.invasive_species else "None reported"} |
        """)
    return


@app.cell
def _map(mo, lake):
    if lake is None:
        return

    import folium

    # point.epsg4326 is [lon, lat]
    _coords = lake.point.epsg4326
    _lat, _lon = _coords[1], _coords[0]

    _m = folium.Map(location=[_lat, _lon], zoom_start=13)
    folium.Marker(
        location=[_lat, _lon],
        popup=folium.Popup(
            f"<b>{lake.name}</b><br>"
            f"Area: {lake.morphology.area:,.1f} acres<br>"
            f"Max Depth: {lake.morphology.max_depth} ft<br>"
            f"Nearest Town: {lake.nearest_town}",
            max_width=250,
        ),
        tooltip=lake.name,
        icon=folium.Icon(color="blue", icon="tint", prefix="fa"),
    ).add_to(_m)

    mo.Html(_m._repr_html_())
    return


@app.cell
def _survey_overview(mo, survey):
    if survey is None:
        mo.md("No survey data available.")
    else:
        mo.md(f"""
        ## Survey Overview

        | Field | Value |
        |---|---|
        | Area (acres) | {survey.area_acres:,.1f} |
        | Max Depth (ft) | {survey.max_depth_feet} |
        | Mean Depth (ft) | {survey.mean_depth_feet if survey.mean_depth_feet is not None else "N/A"} |
        | Shore Length (mi) | {survey.shore_length_miles:.2f} |
        | Littoral Acres | {survey.littoral_acres:.1f} |
        | Avg Water Clarity (ft) | {survey.average_water_clarity} |
        | Surveys on record | {len(survey.surveys)} |
        | Public Accesses | {len(survey.accesses)} |
        """)
    return


@app.cell
def _catch_table(mo, pd, survey):
    if survey is None or not survey.surveys:
        mo.md("No fish catch data available.")
    else:
        _rows = []
        for _s in survey.surveys:
            for _f in _s.fish_catch_summaries:
                _rows.append({
                    "Survey Date": _s.survey_date,
                    "Survey Type": _s.survey_type,
                    "Species": _f.species,
                    "Total Catch": _f.total_catch,
                    "CPUE": _f.cpue,
                    "Avg Weight (lbs)": _f.average_weight,
                    "Gear": _f.gear,
                })

        if _rows:
            _df = pd.DataFrame(_rows).sort_values("Survey Date", ascending=False)
            mo.vstack([
                mo.md(f"## Fish Catch Summaries ({len(_rows)} records across {len(survey.surveys)} surveys)"),
                mo.ui.table(_df, pagination=True),
            ])
        else:
            mo.md("No fish catch summaries in the survey data.")
    return


@app.cell
def _water_levels(mo, pd, water_levels):
    if not water_levels:
        mo.md("No water level data available.")
    else:
        _df = pd.DataFrame([
            {
                "Date": wl.read_date,
                "Elevation": wl.elevation,
                "Datum Adj": wl.datum_adj,
            }
            for wl in water_levels
        ]).sort_values("Date", ascending=False)

        mo.vstack([
            mo.md(f"## Water Level History ({len(_df)} readings)"),
            mo.ui.table(_df.head(100), pagination=True),
        ])
    return


if __name__ == "__main__":
    app.run()
