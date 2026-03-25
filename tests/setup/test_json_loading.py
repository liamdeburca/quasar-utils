from pathlib import Path
from json import load as load_json

_this_file = Path(__file__).resolve()
_json_path = _this_file.parents[2] / "notebooks" / "fitting_info.json"

def test_has_found_json_file():
    assert _json_path.exists()

def test_all_fields_are_valid():
    with open(_json_path, "r") as f:
        d = load_json(f)

    for parent_name, parent_field in d.items():
        for name, field in parent_field.items():
            try:
                _ = field['value']
                _ = field['parse_as']
            except:
                value = field.get("value", "<missing>")
                parse_as = field.get("parse_as", "<missing>")
                msg = f"Field ({parent_name}-{name}) is invalid: "
                msg += f"value={value}, parse_as={parse_as}"
                raise KeyError(msg)

def test_all_fields_load():
    from quasar_utils.setup.utils.json_field import JSONField
    with open(_json_path, "r") as f:
        d = load_json(f)

    for parent_field in d.keys():
        for field in d[parent_field].keys():
            try:
                _ = JSONField.load_from_json(d, field, parent_field)
            except Exception as e:
                msg = f"Failed to load field ({parent_field}-{field}): {e}"
                raise ValueError(msg) from e

def test_absorption_info_can_load_json():
    from quasar_utils.setup.absorption import AbsorptionInfo
    try:
        _ = AbsorptionInfo.from_json(_json_path)
    except Exception as e:
        msg = f"Failed to load 'AbsorptionInfo' from JSON: {e}"
        raise ValueError(msg) from e

def test_balmer_info_can_load_json():
    from quasar_utils.setup.balmer import BalmerInfo
    try:
        _ = BalmerInfo.from_json(_json_path)
    except Exception as e:
        msg = f"Failed to load 'BalmerInfo' from JSON: {e}"
        raise ValueError(msg) from e
    
def test_continuum_info_can_load_json():
    from quasar_utils.setup.continuum import ContinuumInfo
    try:
        _ = ContinuumInfo.from_json(_json_path)
    except Exception as e:
        msg = f"Failed to load 'ContinuumInfo' from JSON: {e}"
        raise ValueError(msg) from e
    
def test_iron_info_can_load_json():
    from quasar_utils.setup.iron import IronInfo
    try:
        _ = IronInfo.from_json(_json_path)
    except Exception as e:
        msg = f"Failed to load 'IronInfo' from JSON: {e}"
        raise ValueError(msg) from e
    
def test_lines_info_can_load_json():
    from quasar_utils.setup.lines import LinesInfo
    try:
        _ = LinesInfo.from_json(_json_path)
    except Exception as e:
        msg = f"Failed to load 'LinesInfo' from JSON: {e}"
        raise ValueError(msg) from e
    
def test_loading_info_can_load_json():
    from quasar_utils.setup.loading import LoadingInfo
    try:
        _ = LoadingInfo.from_json(_json_path)
    except Exception as e:
        msg = f"Failed to load 'LoadingInfo' from JSON: {e}"
        raise ValueError(msg) from e
    
def test_nonlinear_info_can_load_json():
    from quasar_utils.setup.nonlinear import NonLinearInfo
    try:
        _ = NonLinearInfo.from_json(_json_path)
    except Exception as e:
        msg = f"Failed to load 'NonLinearInfo' from JSON: {e}"
        raise ValueError(msg) from e
        
def test_units_info_can_load_json():
    from quasar_utils.setup.units import UnitsInfo
    try:
        _ = UnitsInfo.from_json(_json_path)
    except Exception as e:
        msg = f"Failed to load 'UnitsInfo' from JSON: {e}"
        raise ValueError(msg) from e
