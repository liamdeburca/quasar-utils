import logging
import pickle
from pathlib import Path

import pytest

from quasar_utils.pipeline.sub_dir import SubDir
from quasar_utils.pipeline.input_dir import InputDir
from quasar_utils.pipeline.output_dir import OutputDir


def test_pickling_subdir_before_handlers(tmp_path: Path):
    in_file = tmp_path / "pick1.asc"
    in_file.write_text("dummy")

    sub = SubDir(in_file, tmp_path / "pick1_out")

    data = pickle.dumps(sub)
    sub2 = pickle.loads(data)

    assert str(sub2.in_file) == str(sub.in_file)
    assert sub2.handlers == {}

    logger = logging.getLogger("quasar")
    with sub2:
        logger.info("post-pickle message")
        assert Path(sub2._main_log).exists()


def test_pickling_subdir_after_handlers(tmp_path: Path):
    in_file = tmp_path / "pick2.asc"
    in_file.write_text("dummy")

    sub = SubDir(in_file, tmp_path / "pick2_out")

    logger = logging.getLogger("quasar")
    with sub:
        logger.info("before-pickle")

    data = pickle.dumps(sub)
    sub3 = pickle.loads(data)

    with sub3:
        logger.info("after-unpickle")
        assert Path(sub3._main_log).exists()


def test_pickling_outputdir(tmp_path: Path):
    d = tmp_path / "inputs"
    d.mkdir()
    f1 = d / "a.asc"
    f2 = d / "b.asc"
    f1.write_text("a")
    f2.write_text("b")

    input_dir = InputDir(d)
    out = OutputDir(input_dir, path=tmp_path / "odir")
    data = pickle.dumps(out)
    out2 = pickle.loads(data)

    assert len(out2.subdirs) == len(out.subdirs)
    for s in out2.subdirs:
        with s:
            logging.getLogger("quasar").info("ok")
            assert Path(s._main_log).exists()
