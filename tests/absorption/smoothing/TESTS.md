Smoothing tests — plan and implementation notes
=============================================

This file documents the smoothing test suite implemented under tests/absorption.
It enumerates the tests, why they exist, and how to run them.

Overview
--------
- Priority A: validation and robustness (test_smoothing_validation.py)
- Priority B: correctness and solver tests (test_smoothing_correctness.py, test_smoothing_solver.py, test_smoothing_mask_interp.py)
- Priority C: randomized/property and stability tests (test_smoothing_randomized.py)

Test details
------------

1. test_smoothing_validation.py
   - Validation checks for window size parity, w > p, and input length.
   - Confirms behavior when no valid indices exist (e.g., dy <= 0).

2. test_smoothing_correctness.py
   - Compares the implementation to scipy.signal.savgol_filter for uniform
     uncertainties. To avoid edge-mode differences, tests compare central
     regions (exclude w//2 elements at both ends).

3. test_smoothing_solver.py
   - Unit-tests the solve_weighted_poly implementation against a per-slide
     numpy.linalg.lstsq reference. Also exercises underdetermined slides.

4. test_smoothing_mask_interp.py
   - Tests interpolate_missing edge cases (no available points, single point)
     and create_slides shape and mask-filtering behaviour.

5. test_smoothing_randomized.py
   - Randomized equivalence checks across several seeds and a stability test
     asserting that tiny perturbations in input produce small output changes.

How to run
----------
Activate the project's conda environment and run pytest from repository root:

  conda activate quasar
  python -m pytest -q tests/absorption

Notes
-----
- Tests rely on running under the project's conda environment (Python 3.12.11)
  so the package imports behave as during normal use.
- Numeric tolerances used are rtol=1e-6 and atol=1e-8 for comparisons with
  SciPy's implementation; stability test uses a looser absolute threshold of
  1e-3 for perturbed inputs.
