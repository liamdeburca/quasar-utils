import logging
from pathlib import Path

import pytest

from quasar_utils.pipeline.sub_dir import SubDir


def test_subdir_exception_cleanup(tmp_path: Path):
    in_file = tmp_path / "spec2.asc"
    in_file.write_text("dummy")
    out_dir = tmp_path / "spec2_out"
    sub = SubDir(in_file, out_dir)

    quasar_logger = logging.getLogger("quasar")
    prev_level = quasar_logger.level

    with pytest.raises(RuntimeError):
        with sub:
            quasar_logger.info("about to raise")
            raise RuntimeError("boom")

    assert sub.handlers == {}
    assert quasar_logger.level == prev_level
