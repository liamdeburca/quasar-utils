from dataclasses import fields
from pathlib import Path

import pytest

from quasar_utils.setup import Info

_this_file: Path = Path(__file__).resolve()

def _validate_hashes(subtests: pytest.Subtests, tst: Info, ctr: Info):
    for _field in sorted(fields(tst), key=lambda f: f.name):        
        tst_field = getattr(tst, _field.name)
        ctr_field = getattr(ctr, _field.name)
        h1 = tst_field.create_hash("sha256")
        h2 = ctr_field.create_hash("sha256")
        with subtests.test(msg=f"Validating field: {_field.name}"):
            assert h1 == h2

    h1 = tst.create_hash("sha256")
    h2 = ctr.create_hash("sha256")
    with subtests.test(msg="Validating overall hash"):
        assert h1 == h2


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
    info = Info()
    info.to_json(json_path)

def test_yaml_dump(yaml_path: Path):
    info = Info()
    info.to_yaml(yaml_path)

def test_json_dump_and_load(subtests: pytest.Subtests, json_path: Path):
    tst = Info()
    tst.to_json(json_path)
    ctr = Info.from_json(json_path, create_copy=True)
    _validate_hashes(subtests, tst, ctr)

def test_yaml_dump_and_load(subtests: pytest.Subtests, yaml_path: Path):
    tst = Info()
    tst.to_yaml(yaml_path)
    ctr = Info.from_yaml(yaml_path, create_copy=True)
    _validate_hashes(subtests, tst, ctr)
