import logging
from pathlib import Path

import pytest

from quasar_utils.pipeline.sub_dir import SubDir


def test_subdir_lazy_and_lifecycle(tmp_path: Path):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    in_file = in_dir / "spec1.asc"
    in_file.write_text("dummy")

    out_root = tmp_path / "out_root"
    sub_out = out_root / "spec1_out"

    sub = SubDir(in_file, sub_out)

    # No files or handlers before use
    assert not Path(sub._debug_log).exists()
    assert not Path(sub._main_log).exists()
    assert sub.handlers == {}

    quasar_logger = logging.getLogger("quasar")
    prev_level = quasar_logger.level

    with sub:
        assert Path(sub._debug_log).exists()
        assert Path(sub._main_log).exists()
        assert sub.handlers

        quasar_logger.info("hello world")
        text = Path(sub._main_log).read_text()
        assert "hello world" in text

        assert isinstance(quasar_logger.level, int)

    # After exit
    assert sub.handlers == {}
    assert quasar_logger.level == prev_level
