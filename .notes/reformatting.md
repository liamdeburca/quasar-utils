# Reformatting Guide for AI Agents

This file is supposed to help users and AI agents understand how to reformat old code to meet current standards and best practices, specifically code relying on Pydantic validation. The old practice often involved using the `__wrapped__` attribute of a validated function to access the original, unwrapped function. This is beneficial for performance and use within functions that are already validated, a.k.a. pre-validated code blocks. This pattern is messy and difficult to understand, and relied on a custom `@validate_call` decorator that correctly typed the `__wrapped__` attribute for static type checkers.

The recommended approach is, for each function, to define two separate versions: a private, unvalidated version that contains the core logic, and a public, validated version that wraps the private version. This makes the code easier to understand, maintain, and test, while still benefiting from Pydantic's validation where necessary.

To ensure that both functions share docstrings, the `functools.wraps` decorator is used, but this requires the following pattern: (1) All public (validated) functions with docstrings are defined at the beginning of the file/module, and (2) all private (unvalidated) functions are defined afterwards, using `functools.wraps` to copy the docstring from the public function. If the module makes use of any other helper functions, that are not expected to be used outside the scope, these are defined at the very end. 

## Examples

An example of this pattern is shown below:

```python
from functools import wraps
from pydantic import validate_call

### Public functions

@validate_call
def some_func(a: int, b: int) -> int:
    """Add two numbers."""
    return _some_func(a, b)

### Private functions

@wraps(some_func)
def _some_func(a: int, b: int) -> int:
    return _helper_add_two_ints(a, b)

### Helper functions

def _helper_add_two_ints(a: int, b: int) -> int:
    """Trivial helper function."""
    return a + b
```

Note the following:

1. I use "Public functions", "Private functions", and "Helper functions" as headings to clearly separate the different types of functions in the module.
2. All imports are placed at the top of the file/module. There are never any local, function-level imports. Imports should be sorted and grouped according to Ruff conventions.
3. Public functions are always validated using Pydantic's `@validate_call` decorator, while private functions are not validated and should contain the core logic.
4. Private functions' names always start with an underscore (`_`) to indicate that they are intended for internal use only and should not be accessed directly from outside the module.
5. Helper functions are also considered private, but they are typically more granular and focused on a single task. They should also start with an underscore (`_`) and be defined at the very end of the module.
6. The `functools.wraps` decorator is used to copy the docstring from the public function to the private function, ensuring that documentation remains consistent and accessible for both developers and static type checkers.

## Suggested Additions

The following short guidelines and checks make the refactor pattern safer and easier to maintain.

1. Forbid use of `.__wrapped__` anywhere in the repository.
   - Rationale: prevents reintroducing the old pattern. Add an automated check (rg in CI or a pre-commit hook) that fails when `\\.__wrapped__` is present.

2. Note `functools.wraps` side-effects.
   - `@wraps(public)` copies the public docstring but also copies `__name__`, `__module__`, and sets `__wrapped__` on the private function. Be aware of these attributes for introspection and some type-checker tooling.

3. Ordering and imports.
   - Public (validated) functions must be defined before the private ones that use `@wraps(public)`.
   - Keep all imports at module top and follow your project's import sorting rules (Ruff/black style).

4. Export surface and `__all__`.
   - Ensure `__all__` and package `__init__` only expose public names. Do not export private `_` names.
   - When replacing an exported symbol, update package-level exports accordingly.

5. Testing and type-checking guidance.
   - Run the module's tests after a refactor (or at least module-specific tests).
   - Run your type checker (mypy/pyright) to confirm overloads and TypeVar changes are sound.

6. Behavior and dtype cautions.
   - Small implementation changes (for example `full_like(y, z_fill)` vs `full_like(..., dtype=float64)`) can change dtypes. Add or update tests that compare dtypes or use `assert_allclose(..., dtype=...)` if dtype matters.

7. Batched / vectorized inputs.
   - If you generalize functions to accept matrices (batched inputs), document intended shapes and add tests for both vector and batched inputs.

8. Performance note.
   - Public wrappers validate inputs once; internal callers should call the private `_` implementation to avoid redundant validation. Add a short comment where appropriate.

9. Transition / deprecation policy.
   - If backward compatibility is required, keep old symbols for one release and add deprecation warnings that tell users to migrate to the new public wrappers.

10. CI / lint rules.
    - Add a lint or CI check for the `.__wrapped__` ban and for running the refactored module's tests. Optionally add a lightweight structural check that public functions are present and private implementations are prefixed with `_`.

(End of file)
