include .env
export

.PHONY: install
install:
	poetry install
	pip install -e .

.env:
	echo "" > .env

.PHONY: check
check:
	@echo "=== 1. Syntax check ==="
	python -c "import ast, pathlib; [ast.parse(p.read_text()) for p in pathlib.Path('app').rglob('*.py')]"
	python -c "import ast, pathlib; [ast.parse(p.read_text()) for p in pathlib.Path('src').rglob('*.py')]"
	@echo "=== 2. Import validation ==="
	PYTHONPATH=src python -c "\
from onkia.dnr_client import MnDnrLakeTopographyService; \
from onkia.water_temp import WATER_TEMP_PREFERENCES; \
from onkia.models import WaterTempPreference, DepthPreference, BathymetryContour; \
from onkia.weather import get_weather_for_window; \
from onkia.usgs_glm import get_temperature_profile, get_thermocline_depth; \
from onkia.bathymetry import available_lakes, load_contours; \
from onkia.plan_generator import generate_evening_plan; \
print('All onkia imports OK')"
	@echo "=== 3. Smoke tests ==="
	PYTHONPATH=src pytest tests/test_app_smoke.py -v
	@echo "=== 4. Full test suite ==="
	PYTHONPATH=src pytest tests/ -v
