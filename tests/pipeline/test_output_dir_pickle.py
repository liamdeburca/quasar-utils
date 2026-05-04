"""Unit tests for OutputDir pickling."""

import pickle
import tempfile
from pathlib import Path

import pytest

from quasar_utils.pipeline.input_dir import InputDir
from quasar_utils.pipeline.sub_dir import SubDir
from quasar_utils.pipeline.output_dir import OutputDir


@pytest.fixture
def temp_dir_with_fits():
    """Create a temporary directory with a FITS file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        fits_file = tmpdir / "test_spectrum.fits"
        fits_file.touch()
        yield tmpdir, fits_file


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


@pytest.fixture
def input_dir_fixture(temp_dir_with_fits):
    """Create an InputDir from temp directory with FITS file."""
    tmpdir, fits_file = temp_dir_with_fits
    return InputDir(fits_file)


@pytest.fixture
def input_dir_with_ten_asc_files(temp_dir_with_ten_asc_files):
    """Create an InputDir with 10 ASC files."""
    tmpdir, asc_files = temp_dir_with_ten_asc_files
    return InputDir(tmpdir)


@pytest.fixture
def temp_output_dir(temp_dir_with_fits):
    """Create a temporary output directory."""
    tmpdir, fits_file = temp_dir_with_fits
    output_path = tmpdir / "output"
    output_path.mkdir()
    return output_path


@pytest.fixture
def temp_output_dir_for_ten_files(temp_dir_with_ten_asc_files):
    """Create a temporary output directory for 10 files."""
    tmpdir, asc_files = temp_dir_with_ten_asc_files
    output_path = tmpdir / "output"
    output_path.mkdir()
    return output_path


class TestOutputDirPickle:
    """Tests for OutputDir pickling functionality."""

    def test_basic_pickle_and_unpickle(self, input_dir_fixture, temp_output_dir):
        """Test that OutputDir can be pickled and unpickled."""
        output_dir = OutputDir(input_dir_fixture, path=temp_output_dir)
        
        # Pickle and unpickle
        pickled = pickle.dumps(output_dir)
        unpickled = pickle.loads(pickled)
        
        # Verify basic attributes
        assert unpickled.input_dir.path == output_dir.input_dir.path
        assert unpickled.path == output_dir.path
        assert unpickled.dangerous == output_dir.dangerous

    def test_pickle_preserves_input_dir(self, input_dir_fixture, temp_output_dir):
        """Test that input_dir is preserved after pickling."""
        output_dir = OutputDir(input_dir_fixture, path=temp_output_dir)
        
        # Pickle and unpickle
        pickled = pickle.dumps(output_dir)
        unpickled = pickle.loads(pickled)
        
        # Verify input_dir is preserved
        assert unpickled.input_dir.path == input_dir_fixture.path
        assert unpickled.input_dir.files == input_dir_fixture.files

    def test_pickle_preserves_output_path(self, input_dir_fixture, temp_output_dir):
        """Test that output path is preserved after pickling."""
        output_dir = OutputDir(input_dir_fixture, path=temp_output_dir)
        
        # Pickle and unpickle
        pickled = pickle.dumps(output_dir)
        unpickled = pickle.loads(pickled)
        
        # Verify output path is preserved
        assert unpickled.path == temp_output_dir

    def test_pickle_preserves_subdirs(self, input_dir_fixture, temp_output_dir):
        """Test that subdirectories are preserved after pickling."""
        output_dir = OutputDir(input_dir_fixture, path=temp_output_dir)
        original_subdirs = list(output_dir)
        
        # Pickle and unpickle
        pickled = pickle.dumps(output_dir)
        unpickled = pickle.loads(pickled)
        unpickled_subdirs = list(unpickled)
        
        # Verify subdirs are preserved
        assert len(unpickled_subdirs) == len(original_subdirs)
        for original, unpickled_subdir in zip(original_subdirs, unpickled_subdirs):
            assert unpickled_subdir == original

    def test_pickle_multiple_times(self, input_dir_fixture, temp_output_dir):
        """Test that OutputDir can be pickled and unpickled multiple times."""
        output_dir = OutputDir(input_dir_fixture, path=temp_output_dir)
        
        # Pickle and unpickle multiple times
        obj = output_dir
        for _ in range(3):
            pickled = pickle.dumps(obj)
            obj = pickle.loads(pickled)
        
        # Verify the object is still correct
        assert obj.path == output_dir.path
        assert obj.dangerous == output_dir.dangerous
        assert set(obj) == set(output_dir)

    def test_getstate_returns_dict_with_expected_keys(self, input_dir_fixture, temp_output_dir):
        """Test that __getstate__ returns a dictionary with expected keys."""
        output_dir = OutputDir(input_dir_fixture, path=temp_output_dir)
        
        state = output_dir.__getstate__()
        
        # Verify structure
        assert isinstance(state, dict)
        assert 'input_dir' in state
        assert 'path' in state
        assert 'dangerous' in state
        assert 'subdirs' in state

    def test_pickle_with_default_output_path(self, input_dir_fixture):
        """Test pickling OutputDir with default output path (None)."""
        output_dir = OutputDir(input_dir_fixture)
        
        # Pickle and unpickle
        pickled = pickle.dumps(output_dir)
        unpickled = pickle.loads(pickled)
        
        # Verify output path equals input_dir.directory
        assert unpickled.path == input_dir_fixture.directory

    def test_pickle_preserves_set_behavior(self, input_dir_fixture, temp_output_dir):
        """Test that OutputDir's set behavior is preserved after pickling."""
        output_dir = OutputDir(input_dir_fixture, path=temp_output_dir)
        original_subdirs = set(output_dir)
        
        # Pickle and unpickle
        pickled = pickle.dumps(output_dir)
        unpickled = pickle.loads(pickled)
        
        # Verify set equality
        assert set(unpickled) == original_subdirs

    def test_pickle_with_dangerous_false(self, input_dir_fixture, temp_output_dir):
        """Test pickling OutputDir with dangerous=False (default)."""
        output_dir = OutputDir(input_dir_fixture, path=temp_output_dir, dangerous=False)
        
        # Pickle and unpickle
        pickled = pickle.dumps(output_dir)
        unpickled = pickle.loads(pickled)
        
        # Verify dangerous flag is False
        assert unpickled.dangerous is False

    def test_pickle_maintains_subdir_attributes(self, input_dir_fixture, temp_output_dir):
        """Test that unpickled OutputDir subdirs have the same attributes."""
        output_dir = OutputDir(input_dir_fixture, path=temp_output_dir)
        original_subdirs = list(output_dir)
        
        # Pickle and unpickle
        pickled = pickle.dumps(output_dir)
        unpickled = pickle.loads(pickled)
        unpickled_subdirs = list(unpickled)
        
        # Verify subdir attributes are maintained
        for original_subdir, unpickled_subdir in zip(original_subdirs, unpickled_subdirs):
            assert unpickled_subdir.in_file == original_subdir.in_file
            assert unpickled_subdir._out_dir == original_subdir._out_dir
            assert unpickled_subdir.current_log == original_subdir.current_log

    def test_pickle_with_unsafe_characters_in_path(self, input_dir_fixture):
        """Test pickling OutputDir with special characters in path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            output_path = tmpdir / "output_with_spaces_and_dashes"
            output_path.mkdir()
            
            output_dir = OutputDir(input_dir_fixture, path=output_path)
            
            # Pickle and unpickle
            pickled = pickle.dumps(output_dir)
            unpickled = pickle.loads(pickled)
            
            # Verify path is preserved correctly
            assert unpickled.path == output_path


class TestOutputDirWithTenAscFiles:
    """Tests for OutputDir with 10 ASCII input files."""

    def test_create_output_dir_with_ten_asc_files(self, input_dir_with_ten_asc_files, temp_output_dir_for_ten_files):
        """Test that OutputDir creates 10 SubDirs for 10 input files."""
        output_dir = OutputDir(input_dir_with_ten_asc_files, path=temp_output_dir_for_ten_files)
        
        # Verify 10 subdirs are created
        assert len(output_dir) == 10
        assert len(list(output_dir)) == 10

    def test_output_dir_subdirs_attribute_size(self, input_dir_with_ten_asc_files, temp_output_dir_for_ten_files):
        """Test that OutputDir's subdirs have correct size."""
        output_dir = OutputDir(input_dir_with_ten_asc_files, path=temp_output_dir_for_ten_files)
        
        # Verify subdirs size
        subdirs = list(output_dir)
        assert len(subdirs) == 10
        assert all(isinstance(subdir, SubDir) for subdir in subdirs)

    def test_output_dir_set_size_is_ten(self, input_dir_with_ten_asc_files, temp_output_dir_for_ten_files):
        """Test that OutputDir (as a set) has size 10."""
        output_dir = OutputDir(input_dir_with_ten_asc_files, path=temp_output_dir_for_ten_files)
        
        # Verify set size
        assert len(output_dir) == 10
        # Convert to set and check again
        as_set = set(output_dir)
        assert len(as_set) == 10

    def test_output_dir_iteration_yields_ten_subdirs(self, input_dir_with_ten_asc_files, temp_output_dir_for_ten_files):
        """Test that iterating OutputDir yields 10 SubDirs."""
        output_dir = OutputDir(input_dir_with_ten_asc_files, path=temp_output_dir_for_ten_files)
        
        # Iterate and collect
        subdirs = list(output_dir)
        assert len(subdirs) == 10
        
        # Verify each is a SubDir with correct input file
        input_files = set(input_dir_with_ten_asc_files.files)
        subdir_input_files = {subdir.in_file for subdir in subdirs}
        assert subdir_input_files == input_files

    def test_output_dir_subdirs_point_to_correct_files(self, input_dir_with_ten_asc_files, temp_output_dir_for_ten_files):
        """Test that each SubDir in OutputDir points to one of the 10 input files."""
        output_dir = OutputDir(input_dir_with_ten_asc_files, path=temp_output_dir_for_ten_files)
        
        input_files = set(input_dir_with_ten_asc_files.files)
        
        # Verify each subdir corresponds to one input file
        for subdir in output_dir:
            assert subdir.in_file in input_files

    def test_output_dir_with_ten_files_no_duplicates(self, input_dir_with_ten_asc_files, temp_output_dir_for_ten_files):
        """Test that OutputDir has no duplicate SubDirs."""
        output_dir = OutputDir(input_dir_with_ten_asc_files, path=temp_output_dir_for_ten_files)
        
        subdirs = list(output_dir)
        subdir_set = set(output_dir)
        
        # Verify no duplicates (list length should equal set length)
        assert len(subdirs) == len(subdir_set) == 10

    def test_output_dir_with_ten_files_all_unique_input_files(self, input_dir_with_ten_asc_files, temp_output_dir_for_ten_files):
        """Test that all 10 SubDirs have unique input files."""
        output_dir = OutputDir(input_dir_with_ten_asc_files, path=temp_output_dir_for_ten_files)
        
        input_files = [subdir.in_file for subdir in output_dir]
        
        # Verify all input files are unique
        assert len(input_files) == len(set(input_files)) == 10

    def test_output_dir_preserves_input_dir_with_ten_files(self, input_dir_with_ten_asc_files, temp_output_dir_for_ten_files):
        """Test that OutputDir preserves InputDir with 10 files."""
        output_dir = OutputDir(input_dir_with_ten_asc_files, path=temp_output_dir_for_ten_files)
        
        # Verify InputDir is preserved
        assert output_dir.input_dir == input_dir_with_ten_asc_files
        assert len(output_dir.input_dir) == 10
        assert len(output_dir.input_dir.files) == 10


class TestOutputDir:
    """Tests for OutputDir instantiation and subdir creation (non-pickling)."""

    def test_output_dir_resets_dangerous(self, input_dir_with_ten_asc_files, temp_output_dir_for_ten_files):
        output_dir = OutputDir(input_dir_with_ten_asc_files, path=temp_output_dir_for_ten_files)
        assert not output_dir.dangerous

    def test_output_dir_creates_ten_subdirs_for_ten_files(self, input_dir_with_ten_asc_files, temp_output_dir_for_ten_files):
        """Test that OutputDir creates 10 SubDirs for 10 input files."""
        output_dir = OutputDir(input_dir_with_ten_asc_files, path=temp_output_dir_for_ten_files)
        
        # Verify 10 subdirs are created
        assert len(output_dir) > 0, "OutputDir should contain subdirs"
        assert len(output_dir) == 10, f"Expected 10 SubDirs, got {len(output_dir)}"

    def test_output_dir_all_subdirs_have_input_files(self, input_dir_with_ten_asc_files, temp_output_dir_for_ten_files):
        """Test that all SubDirs in OutputDir have valid input files."""
        output_dir = OutputDir(input_dir_with_ten_asc_files, path=temp_output_dir_for_ten_files)
        
        input_files = set(input_dir_with_ten_asc_files.files)
        for subdir in output_dir:
            assert hasattr(subdir, 'in_file'), "SubDir should have in_file attribute"
            assert subdir.in_file in input_files, f"SubDir in_file {subdir.in_file} not in input files"

    def test_output_dir_single_file_subdir_attributes(self, input_dir_fixture, temp_output_dir):
        """Test that OutputDir's single SubDir has correct attributes."""
        output_dir = OutputDir(input_dir_fixture, path=temp_output_dir)
        
        assert len(output_dir) == 1, f"Expected 1 SubDir, got {len(output_dir)}"
        subdir = list(output_dir)[0]
        
        assert isinstance(subdir, SubDir)
        assert subdir.in_file == input_dir_fixture.path
        assert subdir._out_dir is not None

    def test_output_dir_does_not_lose_subdirs_on_access(self, input_dir_with_ten_asc_files, temp_output_dir_for_ten_files):
        """Test that OutputDir doesn't lose SubDirs when accessing them multiple times."""
        output_dir = OutputDir(input_dir_with_ten_asc_files, path=temp_output_dir_for_ten_files)
        
        # Access subdirs multiple times
        first_access = len(output_dir)
        second_access = len(list(output_dir))
        third_access = len(output_dir)
        
        assert first_access == 10, f"First access: expected 10 SubDirs, got {first_access}"
        assert second_access == 10, f"Second access: expected 10 SubDirs, got {second_access}"
        assert third_access == 10, f"Third access: expected 10 SubDirs, got {third_access}"

    def test_output_dir_subdirs_match_input_files_count(self, input_dir_with_ten_asc_files, temp_output_dir_for_ten_files):
        """Test that the number of SubDirs matches the number of input files."""
        output_dir = OutputDir(input_dir_with_ten_asc_files, path=temp_output_dir_for_ten_files)
        
        num_subdirs = len(output_dir)
        num_input_files = len(input_dir_with_ten_asc_files.files)
        
        assert num_subdirs == num_input_files, f"SubDir count ({num_subdirs}) should match input file count ({num_input_files})"

    def test_output_dir_as_set_has_correct_size(self, input_dir_with_ten_asc_files, temp_output_dir_for_ten_files):
        """Test that OutputDir's set size is correct."""
        output_dir = OutputDir(input_dir_with_ten_asc_files, path=temp_output_dir_for_ten_files)
        
        # Test as set
        as_set = set(output_dir)
        assert len(as_set) == 10, f"Set size should be 10, got {len(as_set)}"

    def test_output_dir_iteration_does_not_modify_size(self, input_dir_with_ten_asc_files, temp_output_dir_for_ten_files):
        """Test that iterating OutputDir doesn't modify its size."""
        output_dir = OutputDir(input_dir_with_ten_asc_files, path=temp_output_dir_for_ten_files)
        
        initial_size = len(output_dir)
        
        # Iterate
        for _ in output_dir:
            pass
        
        final_size = len(output_dir)
        assert initial_size == final_size == 10, f"Size changed during iteration: {initial_size} -> {final_size}"
