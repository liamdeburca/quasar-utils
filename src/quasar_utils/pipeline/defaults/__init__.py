__all__ = [
    "DIRECTORY_TO_DEFAULTS",
    "FITTING_INFO",
    "LINE_LIST",
    "MEASURE_LIST",
]

from pathlib import Path

from quasar_typing.pathlib import CSVPath, JSONPath

_this_path: Path = Path(__file__).resolve()

DIRECTORY_TO_DEFAULTS: Path = _this_path.parent

FITTING_INFO: JSONPath = DIRECTORY_TO_DEFAULTS / "fitting_info.json"
LINE_LIST: CSVPath = DIRECTORY_TO_DEFAULTS / "line_list.csv"
MEASURE_LIST: CSVPath = DIRECTORY_TO_DEFAULTS / "measure_list.csv"
