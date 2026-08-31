from pathlib import Path

import pytest

_this_file: Path = Path(__file__).resolve()

@pytest.fixture(scope="function")
def json_path():
    path = _this_file.parent / "temp.json" 
    yield path
    path.unlink()

@pytest.fixture(scope="function")
def yaml_path():
    path = _this_file.parent / "temp.yaml" 
    yield path
    path.unlink()

def test_json_dump(json_path: Path):
    from quasar_utils.setup import Info
    info = Info()
    info.to_json(json_path)

def test_yaml_dump(yaml_path: Path):
    from quasar_utils.setup import Info
    info = Info()
    info.to_yaml(yaml_path)

def test_json_dump_and_load(json_path: Path):
    from quasar_utils.setup import Info
    info = Info()
    info.to_json(json_path)
    _ = Info.from_json(json_path, create_copy=True)

def test_yaml_dump_and_load(yaml_path: Path):
    from quasar_utils.setup import Info
    info = Info()
    info.to_yaml(yaml_path)
    _ = Info.from_yaml(yaml_path, create_copy=True)