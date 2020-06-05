from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PATH_TO_HOUSING_DATA = PROJECT_ROOT.joinpath(Path("data/housing/train.csv"))
PATH_TO_WEATHER_DATA = PROJECT_ROOT.joinpath(Path("data/weather/operations.csv"))
