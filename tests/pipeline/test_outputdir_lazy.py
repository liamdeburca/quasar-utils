import logging
from pathlib import Path

from quasar_utils.pipeline.input_dir import InputDir
from quasar_utils.pipeline.output_dir import OutputDir


def test_outputdir_iteration_is_lazy(tmp_path: Path):
    d = tmp_path / "in2"
    d.mkdir()
    (d / "one.asc").write_text("1")
    (d / "two.asc").write_text("2")

    input_dir = InputDir(d)
    out = OutputDir(input_dir, path=tmp_path / "out2")

    for s in out.subdirs:
        assert not Path(s._debug_log).exists()
        assert not Path(s._main_log).exists()

    s = next(iter(out.subdirs))
    with s:
        logging.getLogger("quasar").info("logging after lazy")
        assert Path(s._main_log).exists()
