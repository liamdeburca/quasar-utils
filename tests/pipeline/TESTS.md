Tests for quasar_utils.pipeline
================================

This document explains each test file and the individual test functions
contained in tests/pipeline. Tests use pytest and the `tmp_path` fixture so
temporary directories are managed and cleaned up by pytest.

Files and motivations
---------------------

1. test_subdir_lazy_and_lifecycle.py
   - test_subdir_lazy_and_lifecycle
     Purpose: Verify the SubDir context manager creates log files and handlers
     lazily on enter, writes records to the log, and restores logger state on
     exit. Ensures no logs or handlers exist before entering the context.

2. test_subdir_exception_cleanup.py
   - test_subdir_exception_cleanup
     Purpose: Ensure that an exception raised inside a SubDir context still
     results in cleanup: handlers are closed and the logger level is restored.

3. test_pickling.py
   - test_pickling_subdir_before_handlers
     Purpose: Confirm that a SubDir pickles/unpickles before handlers are
     created and that the unpickled object can still be used to create
     handlers lazily.

   - test_pickling_subdir_after_handlers
     Purpose: Confirm that pickling a SubDir after handlers have been
     created still yields an object that can be used (via context manager)
     after unpickling.

   - test_pickling_outputdir
     Purpose: Ensure OutputDir (and its subdirs) can be pickled/unpickled and
     that SubDir instances remain usable after unpickling.

4. test_outputdir_lazy.py
   - test_outputdir_iteration_is_lazy
     Purpose: Verify OutputDir iteration does not create SubDir log files on
     mere iteration; files are only created when using the SubDir context
     manager or explicitly creating handlers.

Additional notes
----------------
- The tests assume the project package is importable (pip install -e . in the
  correct environment). They use the `quasar` logger namespace to emit test
  log messages.
- Tests avoid platform-dependent file descriptor checks and instead assert
  observable behavior (file existence, handler presence, logger level
  restoration).

Running tests
-------------
1. Activate the correct conda environment (Python 3.12.11):

   conda activate quasar

2. Install package (editable) and test deps if needed:

   pip install -e .
   pip install pytest

3. Run the pipeline tests:

   pytest -q tests/pipeline

All temporary files and directories used by the tests are created under
pytest-managed temporary directories and are cleaned up automatically by
pytest after the test run.
