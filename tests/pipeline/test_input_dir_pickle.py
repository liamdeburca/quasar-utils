"""Unit tests for InputDir pickling."""

import pickle
import tempfile
from pathlib import Path

import pytest

from quasar_utils.pipeline.input_dir import InputDir


@pytest.fixture
def temp_dir_with_fits():
    """Create a temporary directory with a FITS file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        fits_file = tmpdir / "test_spectrum.fits"
        fits_file.touch()
        yield tmpdir, fits_file


@pytest.fixture
def temp_dir_with_asc():
    """Create a temporary directory with an ASCII file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        asc_file = tmpdir / "test_spectrum.asc"
        asc_file.touch()
        yield tmpdir, asc_file


@pytest.fixture
def temp_dir_with_multiple_files():
    """Create a temporary directory with multiple valid files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        fits_file = tmpdir / "spectrum1.fits"
        asc_file = tmpdir / "spectrum2.asc"
        fits_file.touch()
        asc_file.touch()
        yield tmpdir, [fits_file, asc_file]


@pytest.fixture
def temp_dir_with_ten_asc_files():
    """Create a temporary directory with 10 ASCII files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        asc_files = []
        for i in range(10):
            asc_file = tmpdir / f"spectrum_{i:02d}.asc"
            asc_file.touch()
            asc_files.append(asc_file)
        yield tmpdir, asc_files


class TestInputDirPickle:
    """Tests for InputDir pickling functionality."""

    def test_basic_pickle_and_unpickle_single_file(self, temp_dir_with_fits):
        """Test that InputDir with a single file can be pickled and unpickled."""
        tmpdir, fits_file = temp_dir_with_fits
        input_dir = InputDir(fits_file)
        
        # Pickle and unpickle
        pickled = pickle.dumps(input_dir)
        unpickled = pickle.loads(pickled)
        
        # Verify basic attributes
        assert unpickled.path == input_dir.path
        assert unpickled.directory == input_dir.directory
        assert unpickled.files == input_dir.files

    def test_pickle_and_unpickle_directory(self, temp_dir_with_multiple_files):
        """Test that InputDir with a directory can be pickled and unpickled."""
        tmpdir, files = temp_dir_with_multiple_files
        input_dir = InputDir(tmpdir)
        
        # Pickle and unpickle
        pickled = pickle.dumps(input_dir)
        unpickled = pickle.loads(pickled)
        
        # Verify attributes are preserved
        assert unpickled.directory == input_dir.directory
        assert set(unpickled.files) == set(input_dir.files)

    def test_pickle_preserves_path(self, temp_dir_with_fits):
        """Test that the path attribute is preserved after pickling."""
        tmpdir, fits_file = temp_dir_with_fits
        input_dir = InputDir(fits_file)
        
        # Pickle and unpickle
        pickled = pickle.dumps(input_dir)
        unpickled = pickle.loads(pickled)
        
        # Verify path is preserved
        assert unpickled.path == fits_file

    def test_pickle_preserves_directory(self, temp_dir_with_fits):
        """Test that the directory attribute is preserved after pickling."""
        tmpdir, fits_file = temp_dir_with_fits
        input_dir = InputDir(fits_file)
        
        # Pickle and unpickle
        pickled = pickle.dumps(input_dir)
        unpickled = pickle.loads(pickled)
        
        # Verify directory is preserved
        assert unpickled.directory == tmpdir

    def test_pickle_preserves_files_list(self, temp_dir_with_multiple_files):
        """Test that the files list is preserved after pickling."""
        tmpdir, files = temp_dir_with_multiple_files
        input_dir = InputDir(tmpdir)
        
        # Pickle and unpickle
        pickled = pickle.dumps(input_dir)
        unpickled = pickle.loads(pickled)
        
        # Verify files are preserved (order may differ)
        assert set(unpickled.files) == set(input_dir.files)
        assert len(unpickled.files) == len(input_dir.files)

    def test_pickle_multiple_times(self, temp_dir_with_fits):
        """Test that InputDir can be pickled and unpickled multiple times."""
        tmpdir, fits_file = temp_dir_with_fits
        input_dir = InputDir(fits_file)
        
        # Pickle and unpickle multiple times
        obj = input_dir
        for _ in range(3):
            pickled = pickle.dumps(obj)
            obj = pickle.loads(pickled)
        
        # Verify the object is still correct
        assert obj.path == input_dir.path
        assert obj.directory == input_dir.directory
        assert obj.files == input_dir.files

    def test_getstate_returns_dict_with_path(self, temp_dir_with_fits):
        """Test that __getstate__ returns a dictionary with path."""
        tmpdir, fits_file = temp_dir_with_fits
        input_dir = InputDir(fits_file)
        
        state = input_dir.__getstate__()
        
        # Verify structure
        assert isinstance(state, dict)
        assert 'path' in state
        assert state['path'] == fits_file

    def test_pickle_asc_file(self, temp_dir_with_asc):
        """Test pickling InputDir with an ASCII file."""
        tmpdir, asc_file = temp_dir_with_asc
        input_dir = InputDir(asc_file)
        
        # Pickle and unpickle
        pickled = pickle.dumps(input_dir)
        unpickled = pickle.loads(pickled)
        
        # Verify attributes
        assert unpickled.path == asc_file
        assert unpickled.files == [asc_file]

    def test_pickle_directory_with_fits_file(self, temp_dir_with_multiple_files):
        """Test that pickling InputDir created from directory works correctly."""
        tmpdir, files = temp_dir_with_multiple_files
        input_dir = InputDir(tmpdir)
        
        # Pickle and unpickle
        pickled = pickle.dumps(input_dir)
        unpickled = pickle.loads(pickled)
        
        # Verify that unpickled object points to the same directory
        assert unpickled.directory == tmpdir
        assert set(unpickled.files) == set(files)

    def test_pickle_preserves_iteration_behavior(self, temp_dir_with_multiple_files):
        """Test that iteration works the same after unpickling."""
        tmpdir, files = temp_dir_with_multiple_files
        input_dir = InputDir(tmpdir)
        original_files = list(input_dir)
        
        # Pickle and unpickle
        pickled = pickle.dumps(input_dir)
        unpickled = pickle.loads(pickled)
        unpickled_files = list(unpickled)
        
        # Verify iteration yields the same files
        assert set(original_files) == set(unpickled_files)

    def test_pickle_preserves_len(self, temp_dir_with_multiple_files):
        """Test that __len__ works the same after unpickling."""
        tmpdir, files = temp_dir_with_multiple_files
        input_dir = InputDir(tmpdir)
        original_len = len(input_dir)
        
        # Pickle and unpickle
        pickled = pickle.dumps(input_dir)
        unpickled = pickle.loads(pickled)
        
        # Verify length is preserved
        assert len(unpickled) == original_len
        assert len(unpickled) == 2

    def test_ten_asc_files_found(self, temp_dir_with_ten_asc_files):
        """Test that InputDir finds all 10 ASCII files in the directory."""
        tmpdir, asc_files = temp_dir_with_ten_asc_files
        input_dir = InputDir(tmpdir)
        
        # Verify 10 files are found
        assert len(input_dir) == 10
        assert len(input_dir.files) == 10
        assert set(input_dir.files) == set(asc_files)

    def test_ten_asc_files_iteration(self, temp_dir_with_ten_asc_files):
        """Test that all 10 ASC files can be iterated over."""
        tmpdir, asc_files = temp_dir_with_ten_asc_files
        input_dir = InputDir(tmpdir)
        
        # Iterate and collect files
        iterated_files = list(input_dir)
        
        # Verify iteration includes all 10 files
        assert len(iterated_files) == 10
        assert set(iterated_files) == set(asc_files)

    def test_ten_asc_files_pickle_and_unpickle(self, temp_dir_with_ten_asc_files):
        """Test that InputDir with 10 ASC files can be pickled and unpickled."""
        tmpdir, asc_files = temp_dir_with_ten_asc_files
        input_dir = InputDir(tmpdir)
        
        # Pickle and unpickle
        pickled = pickle.dumps(input_dir)
        unpickled = pickle.loads(pickled)
        
        # Verify all 10 files are preserved
        assert len(unpickled) == 10
        assert set(unpickled.files) == set(asc_files)
        assert unpickled.directory == tmpdir

    def test_ten_asc_files_pickle_multiple_times(self, temp_dir_with_ten_asc_files):
        """Test that InputDir with 10 ASC files survives multiple pickle cycles."""
        tmpdir, asc_files = temp_dir_with_ten_asc_files
        input_dir = InputDir(tmpdir)
        
        # Pickle and unpickle multiple times
        obj = input_dir
        for _ in range(3):
            pickled = pickle.dumps(obj)
            obj = pickle.loads(pickled)
        
        # Verify all 10 files are still preserved
        assert len(obj) == 10
        assert set(obj.files) == set(asc_files)