"""Unit tests for SubDir pickling."""

import pickle
import tempfile
from pathlib import Path
from logging import FileHandler, INFO, DEBUG

import pytest

from quasar_utils.pipeline.sub_dir import SubDir


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        in_file = tmpdir / "test_input.fits"
        in_file.touch()
        out_dir = tmpdir / "test_output"
        out_dir.mkdir()
        yield in_file, out_dir


class TestSubDirPickle:
    """Tests for SubDir pickling functionality."""

    def test_basic_pickle_and_unpickle(self, temp_dirs):
        """Test that SubDir can be pickled and unpickled."""
        in_file, out_dir = temp_dirs
        subdir = SubDir(in_file, out_dir)
        
        # Pickle and unpickle
        pickled = pickle.dumps(subdir)
        unpickled = pickle.loads(pickled)
        
        # Verify basic attributes
        assert unpickled.in_file == subdir.in_file
        assert unpickled._out_dir == subdir._out_dir
        assert unpickled._debug_log == subdir._debug_log
        assert unpickled._main_log == subdir._main_log
        assert unpickled._profile == subdir._profile

    def test_pickle_with_custom_log_paths(self, temp_dirs):
        """Test pickling SubDir with custom log file paths."""
        in_file, out_dir = temp_dirs
        debug_log = out_dir / "custom_debug.log"
        main_log = out_dir / "custom_main.log"
        profile = out_dir / "custom_profile.csv"
        
        subdir = SubDir(
            in_file, 
            out_dir,
            _debug_log=debug_log,
            _main_log=main_log,
            _profile=profile,
        )
        
        # Pickle and unpickle
        pickled = pickle.dumps(subdir)
        unpickled = pickle.loads(pickled)
        
        # Verify custom paths are preserved
        assert unpickled._debug_log == debug_log
        assert unpickled._main_log == main_log
        assert unpickled._profile == profile

    def test_pickle_with_current_log(self, temp_dirs):
        """Test pickling SubDir with current_log messages."""
        in_file, out_dir = temp_dirs
        current_log = [
            "Log entry 1\n",
            "Log entry 2\n",
            "Log entry 3\n",
        ]
        
        subdir = SubDir(in_file, out_dir, current_log=current_log)
        
        # Pickle and unpickle
        pickled = pickle.dumps(subdir)
        unpickled = pickle.loads(pickled)
        
        # Verify log messages are preserved
        assert unpickled.current_log == current_log

    def test_pickle_equality(self, temp_dirs):
        """Test that unpickled SubDir is equal to original."""
        in_file, out_dir = temp_dirs
        subdir = SubDir(in_file, out_dir)
        
        # Pickle and unpickle
        pickled = pickle.dumps(subdir)
        unpickled = pickle.loads(pickled)
        
        # Verify equality
        assert unpickled == subdir

    def test_pickle_hash_consistency(self, temp_dirs):
        """Test that unpickled SubDir has the same hash as original."""
        in_file, out_dir = temp_dirs
        subdir = SubDir(in_file, out_dir)
        
        # Pickle and unpickle
        pickled = pickle.dumps(subdir)
        unpickled = pickle.loads(pickled)
        
        # Verify hash consistency
        assert hash(unpickled) == hash(subdir)

    def test_pickle_empty_handlers(self, temp_dirs):
        """Test pickling SubDir with empty handlers dict."""
        in_file, out_dir = temp_dirs
        subdir = SubDir(in_file, out_dir, handlers={})
        
        # Pickle and unpickle
        pickled = pickle.dumps(subdir)
        unpickled = pickle.loads(pickled)
        
        # Verify handlers are empty
        assert unpickled.handlers == {}

    def test_pickle_multiple_times(self, temp_dirs):
        """Test that SubDir can be pickled and unpickled multiple times."""
        in_file, out_dir = temp_dirs
        subdir = SubDir(in_file, out_dir)
        
        # Pickle and unpickle multiple times
        obj = subdir
        for _ in range(3):
            pickled = pickle.dumps(obj)
            obj = pickle.loads(pickled)
        
        # Verify the object is still equal to the original
        assert obj == subdir
        assert obj.in_file == subdir.in_file
        assert obj._out_dir == subdir._out_dir

    def test_pickle_with_empty_current_log(self, temp_dirs):
        """Test pickling SubDir with empty current_log."""
        in_file, out_dir = temp_dirs
        subdir = SubDir(in_file, out_dir, current_log=[])
        
        # Pickle and unpickle
        pickled = pickle.dumps(subdir)
        unpickled = pickle.loads(pickled)
        
        # Verify empty log is preserved
        assert unpickled.current_log == []

    def test_getstate_returns_dict(self, temp_dirs):
        """Test that __getstate__ returns a dictionary with expected keys."""
        in_file, out_dir = temp_dirs
        subdir = SubDir(in_file, out_dir)
        
        state = subdir.__getstate__()
        
        # Verify all expected keys are present
        expected_keys = {
            'in_file', '_out_dir', '_debug_log', '_main_log', '_profile',
            'current_log', 'handlers'
        }
        assert set(state.keys()) == expected_keys

    def test_getstate_has_string_paths(self, temp_dirs):
        """Test that __getstate__ converts paths to strings."""
        in_file, out_dir = temp_dirs
        subdir = SubDir(in_file, out_dir)
        
        state = subdir.__getstate__()
        
        # Verify paths are strings
        assert isinstance(state['in_file'], str)
        assert isinstance(state['_out_dir'], str)
        assert isinstance(state['_debug_log'], str)
        assert isinstance(state['_main_log'], str)
        assert isinstance(state['_profile'], str)
