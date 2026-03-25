from pathlib import Path

_this_file = Path(__file__).resolve()
_json_path = _this_file.parents[2] / "notebooks" / "fitting_info.json"

def test_can_instantiate_info():
    from quasar_utils.setup.info import Info
    info = Info()
    assert info.is_updated
    assert info.loading['sigma_res'] < 2.4e-4

def test_can_load_info_from_json():
    from quasar_utils.setup.info import Info
    try:
        info = Info.from_json(_json_path)
        assert info.is_updated
        assert info.loading['sigma_res'] < 2.4e-4
    except Exception as e:
        msg = f"Failed to load 'Info' from JSON: {e}"
        raise ValueError(msg)
