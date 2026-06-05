"""
Run this notebook:
    marimo edit notebooks/03_lake_day_analysis.py   # interactive UI
    python notebooks/03_lake_day_analysis.py        # CI / script mode (no data fetched)

Given a lake and a date, this notebook shows:
  - Water temperature preference match for each target species
  - Most recent survey fish catch data
  - Stocking history
  - Fishing technique recommendations
"""

import marimo

__generated_with = "0.11.0"
app = marimo.App(width="medium", title="Lake Day Analysis — Wright County")


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
    import datetime
    return datetime, mo, pd


# ---------------------------------------------------------------------------
# Fishing technique lookup keyed by (species, temp_zone)
# temp_zone: "cold" = below lower avoidance, "cool" = between lower and optimum,
#            "optimal" = within optimum range, "warm" = above optimum, "hot" = above upper avoidance
# ---------------------------------------------------------------------------
_TECHNIQUES = {
    "Largemouth Bass": {
        "cold":    ["Slow-roll finesse jigs near bottom structure", "Drop-shot with small finesse worm"],
        "cool":    ["Slow-twitched jerkbaits", "Ned rig on rocky points", "Swim jigs along weed edges"],
        "optimal": ["Topwater frogs/poppers over vegetation", "Texas-rigged plastic worm", "Spinnerbait along weedlines", "Hollow-body frog in lily pads"],
        "warm":    ["Deep-diving crankbaits", "Carolina rig on main-lake structure", "Football jig"],
        "hot":     ["Very early morning topwater", "Deep shade structure with drop-shot"],
    },
    "Northern Pike": {
        "cold":    ["Large jigging spoons near bottom", "Slow-twitched jerkbaits in deep water"],
        "cool":    ["Jerkbaits along weed edges", "Bucktail spinners", "Live sucker rigs"],
        "optimal": ["Large spinnerbaits", "Flashy spoons over weedlines", "Swimbaits", "Top-water gliders"],
        "warm":    ["Deep-running crankbaits", "Vertical jigging in deepest cool water"],
        "hot":     ["Target deep, cool water only; pike inactive near surface"],
    },
    "Black Crappie": {
        "cold":    ["Small tube jigs tipped with waxworm, fished very slowly"],
        "cool":    ["Small jigs (1/16 oz) under slip float", "Micro-spoons"],
        "optimal": ["Small jig under bobber near brush piles", "Minnow under slip float", "Vertical jigging with small spoon"],
        "warm":    ["Night fishing with light near docks", "Small jig in deeper water"],
        "hot":     ["Night fishing with lights", "Brush-pile fishing in 12–18 ft"],
    },
    "Sunfish": {
        "cold":    ["Tiny jig tipped with waxworm under bobber"],
        "cool":    ["Small beetle-spin", "Wax worm under bobber near weeds"],
        "optimal": ["Small jig or fly under bobber", "Cricket on light hook", "Small spinner in shallow weeds"],
        "warm":    ["Small jig near deeper weed edges", "Live bait under dock shade"],
        "hot":     ["Live crickets near shade", "Early morning in shallow weeds"],
    },
    "Walleye": {
        "cold":    ["Jigging spoon tipped with minnow in deep holes"],
        "cool":    ["Jig-and-minnow along rock structure", "Live minnow under slip bobber on points"],
        "optimal": ["Jig-and-minnow on main-lake structure", "Crawler harness trolled at 1.2–1.5 mph", "Slip bobber with leech near breaklines"],
        "warm":    ["Night fishing along rocky shorelines", "Deep-water crawler harness trolling"],
        "hot":     ["Night fishing only; deep main-lake structure during day"],
    },
}

# Seasonal water temperature estimates for Wright County, MN (°F), by month
_SEASONAL_TEMP = {1: 33, 2: 33, 3: 36, 4: 46, 5: 58, 6: 68, 7: 78, 8: 76, 9: 67, 10: 55, 11: 44, 12: 36}


@app.cell
def _header(mo):
    mo.md("""
    # Lake Day Analysis — Wright County

    Enter a lake, date, and optional water temperature to get fishing guidance:
    water temperature suitability per species, recent survey catch data,
    stocking history, and technique recommendations.
    """)
    return


@app.cell
def _controls(mo, datetime):
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
    date_input = mo.ui.date(
        value=datetime.date.today(),
        label="Date",
    )
    water_temp_input = mo.ui.number(
        start=32,
        stop=95,
        step=1,
        value=0,
        label="Water temp (°F, 0 = estimate from date)",
    )
    reload_btn = mo.ui.run_button(label="Analyze")

    mo.vstack([
        mo.hstack([lake_select, date_input, water_temp_input, reload_btn], gap=2, align="end"),
        mo.md("*Set water temp to 0 to use a seasonal estimate based on the date.*"),
    ])
    return WRIGHT_COUNTY_LAKES, date_input, lake_select, reload_btn, water_temp_input


@app.cell
def _load_data(mo, lake_select, date_input, water_temp_input, reload_btn):
    mo.stop(
        not reload_btn.value,
        mo.callout(
            mo.md("Fill in the controls above and click **Analyze** to load data."),
            kind="info",
        ),
    )

    from onkia.dnr_client import MnDnrLakeTopographyService
    from onkia.water_temp import WATER_TEMP_PREFERENCES
    import xml.etree.ElementTree as ET

    _svc = MnDnrLakeTopographyService()
    _name = lake_select.value

    lake = _svc.get_lake(_name, county_id=86)
    survey = _svc.get_survey(lake.id) if lake else None

    # Stocking data (XML)
    _stocking_root = None
    if lake:
        try:
            _stocking_root = _svc.get_stocking(lake.id)
        except Exception:
            _stocking_root = None

    analysis_date = date_input.value
    raw_water_temp = water_temp_input.value
    water_temp_prefs = WATER_TEMP_PREFERENCES

    return ET, analysis_date, lake, raw_water_temp, survey, water_temp_prefs, _stocking_root


@app.cell
def _compute_water_temp(mo, analysis_date, raw_water_temp):
    import datetime

    _month = analysis_date.month if hasattr(analysis_date, "month") else datetime.date.today().month
    _seasonal = _SEASONAL_TEMP.get(_month, 60)

    if raw_water_temp and raw_water_temp > 0:
        water_temp_f = float(raw_water_temp)
        temp_source = f"user-provided ({water_temp_f:.0f}°F)"
    else:
        water_temp_f = float(_seasonal)
        temp_source = f"seasonal estimate for month {_month} ({water_temp_f:.0f}°F)"

    mo.md(f"""
    ## Conditions on {analysis_date}

    **Water Temperature:** {water_temp_f:.0f}°F *(source: {temp_source})*
    """)
    return water_temp_f, temp_source


@app.cell
def _temp_preference_table(mo, pd, water_temp_f, water_temp_prefs):
    TARGET_SPECIES_NAMES = [
        "Largemouth Bass", "Northern Pike", "Black Crappie",
        "Bluegills", "Sunfish", "Walleye",
    ]

    def _parse_optimum_range(optimum_str):
        """Return (low, high) for the optimum range string like '65-75' or '64'."""
        if not optimum_str or optimum_str == "N/A":
            return None, None
        parts = optimum_str.split("-")
        try:
            if len(parts) == 2:
                return float(parts[0]), float(parts[1])
            return float(parts[0]), float(parts[0])
        except ValueError:
            return None, None

    def _temp_zone(pref, temp):
        """Classify temp relative to species preferences."""
        opt_low, opt_high = _parse_optimum_range(pref.optimum)
        lower_av = pref.lower_avoidance
        upper_av = pref.upper_avoidance

        if upper_av is not None and temp > upper_av:
            return "hot", "🔴 Above avoidance — avoid"
        if opt_high is not None and opt_low is not None:
            if opt_low <= temp <= opt_high:
                return "optimal", "🟢 Optimal"
        if lower_av is not None and temp < lower_av:
            return "cold", "🔵 Below avoidance — very slow"
        if opt_low is not None and temp < opt_low:
            return "cool", "🟡 Below optimum — slower"
        if opt_high is not None and temp > opt_high:
            return "warm", "🟠 Above optimum — deeper/slower"
        return "cool", "🟡 Below optimum — slower"

    _rows = []
    for _pref in water_temp_prefs:
        if _pref.species not in TARGET_SPECIES_NAMES:
            continue
        _zone, _label = _temp_zone(_pref, water_temp_f)
        _rows.append({
            "Species": _pref.species,
            "Lower Avoidance (°F)": _pref.lower_avoidance or "N/A",
            "Optimum (°F)": _pref.optimum,
            "Upper Avoidance (°F)": _pref.upper_avoidance or "N/A",
            "Status at Current Temp": _label,
            "_zone": _zone,
        })

    _df = pd.DataFrame(_rows)

    mo.vstack([
        mo.md("## Water Temperature Suitability"),
        mo.ui.table(_df.drop(columns=["_zone"])),
    ])
    return


@app.cell
def _technique_recommendations(mo, water_temp_f, water_temp_prefs):
    TARGET_SPECIES_NAMES = ["Largemouth Bass", "Northern Pike", "Black Crappie", "Bluegills", "Walleye"]

    def _parse_optimum_range(optimum_str):
        if not optimum_str or optimum_str == "N/A":
            return None, None
        parts = optimum_str.split("-")
        try:
            return (float(parts[0]), float(parts[1])) if len(parts) == 2 else (float(parts[0]), float(parts[0]))
        except ValueError:
            return None, None

    def _zone(pref, temp):
        opt_low, opt_high = _parse_optimum_range(pref.optimum)
        lower_av = pref.lower_avoidance
        upper_av = pref.upper_avoidance
        if upper_av is not None and temp > upper_av:
            return "hot"
        if opt_low is not None and opt_high is not None and opt_low <= temp <= opt_high:
            return "optimal"
        if lower_av is not None and temp < lower_av:
            return "cold"
        if opt_low is not None and temp < opt_low:
            return "cool"
        return "warm"

    # Map Bluegills to Sunfish in techniques table
    _SPECIES_TO_TECHNIQUE_KEY = {
        "Largemouth Bass": "Largemouth Bass",
        "Northern Pike": "Northern Pike",
        "Black Crappie": "Black Crappie",
        "Bluegills": "Sunfish",
        "Walleye": "Walleye",
    }

    _sections = []
    for _pref in water_temp_prefs:
        if _pref.species not in TARGET_SPECIES_NAMES:
            continue
        _z = _zone(_pref, water_temp_f)
        _key = _SPECIES_TO_TECHNIQUE_KEY.get(_pref.species, _pref.species)
        _tips = _TECHNIQUES.get(_key, {}).get(_z, ["General slow presentations near cover."])
        _bullet_list = "\n".join(f"  - {t}" for t in _tips)
        _sections.append(f"**{_pref.species}** *(optimum: {_pref.optimum}°F)*\n{_bullet_list}")

    mo.vstack([
        mo.md("## Fishing Technique Recommendations"),
        mo.md("\n\n".join(_sections)),
    ])
    return


@app.cell
def _recent_surveys(mo, pd, survey):
    if survey is None or not survey.surveys:
        mo.md("No survey data available.")
    else:
        # Most recent 3 surveys
        _recent = sorted(survey.surveys, key=lambda s: s.survey_date, reverse=True)[:3]

        _sections = []
        for _s in _recent:
            _rows = [
                {
                    "Species": f.species,
                    "Total Catch": f.total_catch,
                    "CPUE": f.cpue,
                    "Avg Weight (lbs)": f.average_weight,
                    "Gear": f.gear,
                }
                for f in _s.fish_catch_summaries
            ]
            if _rows:
                _df = pd.DataFrame(_rows).sort_values("Total Catch", ascending=False)
                _sections.append(mo.vstack([
                    mo.md(f"### Survey: {_s.survey_date} — {_s.survey_type}"),
                    mo.ui.table(_df),
                ]))
            else:
                _sections.append(mo.md(f"### Survey: {_s.survey_date} — no catch data"))

        mo.vstack([mo.md("## Recent Survey Data (last 3 surveys)")] + _sections)
    return


@app.cell
def _stocking_history(mo, pd, _stocking_root):
    if _stocking_root is None:
        mo.md("No stocking data available (or not supported for this lake).")
    else:
        _rows = []
        for _record in _stocking_root.iter("item"):
            _row = {child.tag: child.text for child in _record}
            if _row:
                _rows.append(_row)

        if _rows:
            _df = pd.DataFrame(_rows)
            mo.vstack([
                mo.md(f"## Stocking History ({len(_df)} records)"),
                mo.ui.table(_df, pagination=True),
            ])
        else:
            mo.md("Stocking XML parsed but no records found.")
    return


if __name__ == "__main__":
    app.run()
