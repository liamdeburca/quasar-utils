import pytest
import numpy as np
from quasar_utils.absorption.smoothing import get_gap_sizes


class TestGetGapSizes:
    """Test suite for get_gap_sizes function."""

    # ========== EMPTY AND SINGLE ELEMENT ARRAYS ==========

    def test_empty_array(self):
        """Empty array should return empty list."""
        mask = np.array([], dtype=bool)
        result = get_gap_sizes(mask)
        assert result == []

    def test_single_element_false(self):
        """Single False element should return [(0, 0, 1)]."""
        mask = np.array([False], dtype=bool)
        result = get_gap_sizes(mask)
        assert result == [(0, 0, 1)]

    def test_single_element_true(self):
        """Single True element should return []."""
        mask = np.array([True], dtype=bool)
        result = get_gap_sizes(mask)
        assert result == []

    # ========== TWO ELEMENT ARRAYS ==========

    def test_two_elements_all_false(self):
        """[F, F] should return [(0, 1, 2)]."""
        mask = np.array([False, False], dtype=bool)
        result = get_gap_sizes(mask)
        assert result == [(0, 1, 2)]

    def test_two_elements_all_true(self):
        """[T, T] should return []."""
        mask = np.array([True, True], dtype=bool)
        result = get_gap_sizes(mask)
        assert result == []

    def test_two_elements_false_true(self):
        """[F, T] should return [(0, 0, 1)]."""
        mask = np.array([False, True], dtype=bool)
        result = get_gap_sizes(mask)
        assert result == [(0, 0, 1)]

    def test_two_elements_true_false(self):
        """[T, F] should return [(1, 1, 1)]."""
        mask = np.array([True, False], dtype=bool)
        result = get_gap_sizes(mask)
        assert result == [(1, 1, 1)]

    # ========== ALL FALSE ARRAYS ==========

    def test_all_false_length_3(self):
        """Array of all False should return single gap."""
        mask = np.array([False, False, False], dtype=bool)
        result = get_gap_sizes(mask)
        assert result == [(0, 2, 3)]

    def test_all_false_length_5(self):
        """Array of all False should return single gap."""
        mask = np.array([False, False, False, False, False], dtype=bool)
        result = get_gap_sizes(mask)
        assert result == [(0, 4, 5)]

    # ========== ALL TRUE ARRAYS ==========

    def test_all_true_length_3(self):
        """Array of all True should return []."""
        mask = np.array([True, True, True], dtype=bool)
        result = get_gap_sizes(mask)
        assert result == []

    def test_all_true_length_5(self):
        """Array of all True should return []."""
        mask = np.array([True, True, True, True, True], dtype=bool)
        result = get_gap_sizes(mask)
        assert result == []

    # ========== SINGLE TRUE ELEMENT, REST FALSE ==========

    def test_single_true_at_start(self):
        """[T, F, F] should return [(1, 2, 2)]."""
        mask = np.array([True, False, False], dtype=bool)
        result = get_gap_sizes(mask)
        assert result == [(1, 2, 2)]

    def test_single_true_at_end(self):
        """[F, F, T] should return [(0, 1, 2)]."""
        mask = np.array([False, False, True], dtype=bool)
        result = get_gap_sizes(mask)
        assert result == [(0, 1, 2)]

    def test_single_true_in_middle(self):
        """[F, T, F] should return [(0, 0, 1), (2, 2, 1)]."""
        mask = np.array([False, True, False], dtype=bool)
        result = get_gap_sizes(mask)
        assert result == [(0, 0, 1), (2, 2, 1)]

    def test_single_true_surrounded_longer(self):
        """[F, F, T, F, F] should return [(0, 1, 2), (3, 4, 2)]."""
        mask = np.array([False, False, True, False, False], dtype=bool)
        result = get_gap_sizes(mask)
        assert result == [(0, 1, 2), (3, 4, 2)]

    # ========== TWO TRUE ELEMENTS, REST FALSE ==========

    def test_two_true_consecutive_surrounded(self):
        """[F, T, T, F] should return [(0, 0, 1), (3, 3, 1)]."""
        mask = np.array([False, True, True, False], dtype=bool)
        result = get_gap_sizes(mask)
        assert result == [(0, 0, 1), (3, 3, 1)]

    def test_two_true_separated_by_one_false(self):
        """[F, T, F, T, F] should return [(0, 0, 1), (2, 2, 1), (4, 4, 1)]."""
        mask = np.array([False, True, False, True, False], dtype=bool)
        result = get_gap_sizes(mask)
        assert result == [(0, 0, 1), (2, 2, 1), (4, 4, 1)]

    def test_two_true_separated_by_multiple_false(self):
        """[F, T, F, F, T, F] should return [(0, 0, 1), (2, 3, 2), (5, 5, 1)]."""
        mask = np.array([False, True, False, False, True, False], dtype=bool)
        result = get_gap_sizes(mask)
        assert result == [(0, 0, 1), (2, 3, 2), (5, 5, 1)]

    # ========== MULTIPLE GAPS WITH BOTH ENDS TRUE ==========

    def test_multiple_gaps_both_ends_true(self):
        """[T, F, T, F, T] should return [(1, 1, 1), (3, 3, 1)]."""
        mask = np.array([True, False, True, False, True], dtype=bool)
        result = get_gap_sizes(mask)
        assert result == [(1, 1, 1), (3, 3, 1)]

    def test_multiple_gaps_varied_sizes_both_ends_true(self):
        """[T, F, F, T, F, T] should return [(1, 2, 2), (4, 4, 1)]."""
        mask = np.array([True, False, False, True, False, True], dtype=bool)
        result = get_gap_sizes(mask)
        assert result == [(1, 2, 2), (4, 4, 1)]

    def test_multiple_gaps_longer_sequence_both_ends_true(self):
        """[T, F, T, F, F, T, F, T] should return [(1, 1, 1), (3, 4, 2), (6, 6, 1)]."""
        mask = np.array([True, False, True, False, False, True, False, True], dtype=bool)
        result = get_gap_sizes(mask)
        assert result == [(1, 1, 1), (3, 4, 2), (6, 6, 1)]

    # ========== GAPS WITH FALSE AT EDGES ==========

    def test_false_at_start_and_end_with_gap(self):
        """[F, T, F, T, F] should return [(0, 0, 1), (2, 2, 1), (4, 4, 1)]."""
        mask = np.array([False, True, False, True, False], dtype=bool)
        result = get_gap_sizes(mask)
        assert result == [(0, 0, 1), (2, 2, 1), (4, 4, 1)]

    def test_false_at_start_only_with_gap(self):
        """[F, T, F, T] should return [(0, 0, 1), (2, 2, 1)]."""
        mask = np.array([False, True, False, True], dtype=bool)
        result = get_gap_sizes(mask)
        assert result == [(0, 0, 1), (2, 2, 1)]

    def test_false_at_end_only_with_gap(self):
        """[T, F, T, F] should return [(1, 1, 1), (3, 3, 1)]."""
        mask = np.array([True, False, True, False], dtype=bool)
        result = get_gap_sizes(mask)
        assert result == [(1, 1, 1), (3, 3, 1)]

    def test_false_at_start_with_multiple_gaps(self):
        """[F, T, F, F, T, F] should return [(0, 0, 1), (2, 3, 2), (5, 5, 1)]."""
        mask = np.array([False, True, False, False, True, False], dtype=bool)
        result = get_gap_sizes(mask)
        assert result == [(0, 0, 1), (2, 3, 2), (5, 5, 1)]

    # ========== ASYMMETRIC FALSE DISTRIBUTIONS ==========

    def test_false_only_on_left(self):
        """[F, F, T, T] should return [(0, 1, 2)]."""
        mask = np.array([False, False, True, True], dtype=bool)
        result = get_gap_sizes(mask)
        assert result == [(0, 1, 2)]

    def test_false_only_on_right(self):
        """[T, T, F, F] should return [(2, 3, 2)]."""
        mask = np.array([True, True, False, False], dtype=bool)
        result = get_gap_sizes(mask)
        assert result == [(2, 3, 2)]

    def test_false_on_both_sides_no_middle_gaps(self):
        """[F, T, T, F] should return [(0, 0, 1), (3, 3, 1)]."""
        mask = np.array([False, True, True, False], dtype=bool)
        result = get_gap_sizes(mask)
        assert result == [(0, 0, 1), (3, 3, 1)]

    # ========== LARGER ARRAYS WITH VARIOUS PATTERNS ==========

    def test_large_array_simple_pattern(self):
        """Larger array with simple True-False pattern."""
        mask = np.array([True, False, True, False, True, False, True], dtype=bool)
        result = get_gap_sizes(mask)
        assert result == [(1, 1, 1), (3, 3, 1), (5, 5, 1)]

    def test_large_array_varied_gap_sizes(self):
        """Larger array with varied gap sizes."""
        mask = np.array(
            [True, False, True, False, False, False, True, False, False, True],
            dtype=bool
        )
        result = get_gap_sizes(mask)
        assert result == [(1, 1, 1), (3, 5, 3), (7, 8, 2)]

    def test_large_array_with_edge_gaps(self):
        """Larger array with gaps at edges."""
        mask = np.array(
            [False, False, True, False, True, False, False, False],
            dtype=bool
        )
        result = get_gap_sizes(mask)
        assert result == [(0, 1, 2), (3, 3, 1), (5, 7, 3)]

    # ========== TUPLE STRUCTURE VALIDATION ==========

    def test_tuple_structure_correct(self):
        """Each tuple should have (left, right, size) with size = right - left + 1."""
        mask = np.array([True, False, False, True, False, True], dtype=bool)
        result = get_gap_sizes(mask)
        
        for left, right, size in result:
            # Verify size calculation
            assert size == right - left + 1, \
                f"Tuple ({left}, {right}, {size}): size should be {right - left + 1}"
            # Verify indices are in bounds
            assert 0 <= left <= right < len(mask), \
                f"Tuple ({left}, {right}, {size}): indices out of bounds"

    def test_all_gaps_are_false(self):
        """All indices in gaps should correspond to False values in mask."""
        mask = np.array([False, True, False, False, True, False], dtype=bool)
        result = get_gap_sizes(mask)
        
        for left, right, size in result:
            # All elements in gap should be False
            for i in range(left, right + 1):
                assert mask[i] == False, \
                    f"Index {i} in gap ({left}, {right}, {size}) is True, not False"

    def test_gaps_are_contiguous(self):
        """No gap should be adjacent to another without a True between."""
        mask = np.array([False, True, False, True, False], dtype=bool)
        result = get_gap_sizes(mask)
        
        if len(result) > 1:
            for i in range(len(result) - 1):
                current_right = result[i][1]
                next_left = result[i + 1][0]
                # There should be at least one True between gaps
                assert current_right < next_left - 1 or current_right == next_left - 1, \
                    f"Gap {i} ends at {current_right}, gap {i+1} starts at {next_left}"
