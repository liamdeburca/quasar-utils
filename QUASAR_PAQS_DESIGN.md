# Quasar-PAQS CLI Tool Repository Design

## Overview

This document outlines the design and implementation plan for **quasar-paqs**, a new repository that collects command-line tools for the PAQS (Quasar spectral analysis) project. These tools are invoked via `quasar-paqs [subcommand]` from the command line.

## Vision & Requirements

- **Repository Type**: New, separate repository (not part of quasar-utils)
- **Independence**: CLI tools do NOT depend on quasar-utils; they are standalone
- **Installation**: Supports both `pip install quasar-paqs` (package registry) and direct development install
- **Execution Model**: Hierarchical subcommands using argparse
- **Technology Stack**: Pure Python + argparse (with bash helpers only if needed)
- **Python Version**: 3.12+

## Planned CLI Tools

Based on discussion, the initial tool suite includes:

1. **`quasar-paqs run`** — Fit spectral data in a directory
2. **`quasar-paqs create-config`** — Create a template config file (YAML)
3. **`quasar-paqs create-linelist`** — Create a template linelist file
4. **`quasar-paqs create-csv`** — Create a template CSV lookup table
5. **`quasar-paqs create-workspace`** — Create a complete workspace with all templates
6. **`quasar-paqs validate`** — Validate input files and configurations

## Architecture

### Repository Structure

```
quasar-paqs/
├── src/quasar_paqs/
│   ├── __init__.py
│   ├── __main__.py              # Entry point for python -m quasar_paqs
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── main.py              # Root parser and dispatcher
│   │   └── subcommands/
│   │       ├── __init__.py
│   │       ├── run.py           # quasar-paqs run
│   │       ├── templates.py     # quasar-paqs create-* commands
│   │       └── validate.py      # quasar-paqs validate
│   ├── config.py                # Centralized config system
│   └── data/
│       └── templates/
│           ├── config.yaml      # Template config file
│           ├── linelist.dat     # Template linelist
│           ├── lookup.csv       # Template CSV lookup table
│           └── workspace/       # Template workspace structure
├── tests/
│   ├── __init__.py
│   ├── test_cli.py              # Integration tests
│   └── cli/
│       ├── test_main.py
│       ├── test_run.py
│       ├── test_templates.py
│       └── test_validate.py
├── pyproject.toml
├── README.md
└── LICENSE
```

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Flat subcommand structure** | Simpler UX; tools are: `quasar-paqs create-config`, `quasar-paqs create-linelist`, not nested under `quasar-paqs templates` |
| **Argparse over Click/Typer** | Simple, lightweight, fits "Python + argparse" requirement; can upgrade later if needed |
| **YAML for config format** | Human-readable, composable, supports validation (matches quasar-utils patterns) |
| **Templates as package data** | Embedded in wheel via `include_package_data = true`; no external file dependencies |
| **Centralized config system** | Single `config.yaml` for shared parameters; tool-specific overrides available via CLI flags |
| **Independence from quasar-utils** | CLI tools are standalone; no tight coupling; allows independent versioning and deployment |
| **Pure Python initially** | No bash helpers unless complexity requires; can be added later |

## Detailed Design: `cli/main.py`

### Core Dispatcher Pattern

`cli/main.py` is a thin dispatcher that:
1. Creates a root ArgumentParser for `quasar-paqs`
2. Creates a subparsers container for routing to different tools
3. Dynamically registers each subcommand module
4. Implements `main()` entry point that parses args and dispatches to the appropriate handler
5. Handles error reporting and exit codes

### Skeleton Code

```python
# src/quasar_paqs/cli/main.py

import argparse
import sys
from typing import Optional

from quasar_paqs.cli.subcommands import run, templates, validate

def create_parser() -> argparse.ArgumentParser:
    """Create and configure the root argument parser."""
    parser = argparse.ArgumentParser(
        prog="quasar-paqs",
        description="Command-line tools for PAQS (Quasar spectral analysis)",
        epilog="Use 'quasar-paqs <subcommand> --help' for more info.",
    )
    
    # Global optional arguments (available to all subcommands)
    parser.add_argument(
        "--version",
        action="version",
        version="quasar-paqs 0.1.0"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to global config file (YAML)"
    )
    
    # Create subparsers container for subcommands
    subparsers = parser.add_subparsers(
        dest="subcommand",
        help="Available subcommands",
        required=True
    )
    
    # Register each subcommand module
    # Each module provides a register_subcommand(subparsers) function
    run.register_subcommand(subparsers)
    templates.register_subcommand(subparsers)
    validate.register_subcommand(subparsers)
    
    return parser

def main(argv: Optional[list[str]] = None) -> int:
    """
    Entry point for the CLI.
    
    Args:
        argv: Command-line arguments (defaults to sys.argv[1:])
    
    Returns:
        Exit code (0 = success, non-zero = error)
    """
    parser = create_parser()
    args = parser.parse_args(argv)
    
    try:
        # args.func is set by each subcommand's set_defaults(func=handler)
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

### Subcommand Registration Pattern

Each subcommand module (e.g., `run.py`, `templates.py`) implements the same registration pattern:

```python
# Example: src/quasar_paqs/cli/subcommands/run.py

import argparse
import os
import sys
from quasar_paqs.config import load_config

def register_subcommand(subparsers) -> None:
    """Register the 'run' subcommand and its arguments."""
    parser = subparsers.add_parser(
        "run",
        help="Fit spectral data in a directory"
    )
    
    # Positional arguments
    parser.add_argument(
        "input_dir",
        help="Input directory containing spectral data files"
    )
    
    # Optional arguments specific to this subcommand
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="Path to config file (overrides global --config)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="./output",
        help="Output directory for results"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually running"
    )
    
    # Register the handler function for this subcommand
    parser.set_defaults(func=handle_run)

def handle_run(args: argparse.Namespace) -> int:
    """Execute the 'run' subcommand."""
    # Validate inputs
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}", file=sys.stderr)
        return 1
    
    # Load configuration
    config_file = args.config if args.config else args.config  # Use subcommand config or global
    config = load_config(config_file)
    
    # Run the fitting logic
    try:
        results = fit_spectral_data(
            input_dir=args.input_dir,
            config=config,
            output_dir=args.output,
            dry_run=args.dry_run
        )
        print(f"✓ Successfully processed {results.num_files} files")
        print(f"✓ Results saved to {args.output}")
        return 0
    except Exception as e:
        print(f"✗ Fitting failed: {e}", file=sys.stderr)
        return 1
```

### Template Subcommands (Flat Design)

Using a flat design, each template creation command is a top-level subcommand:

```python
# src/quasar_paqs/cli/subcommands/templates.py

import argparse
import os
import sys
from quasar_paqs.data import write_template, write_workspace

def register_subcommand(subparsers) -> None:
    """Register all template creation commands."""
    
    # create-config subcommand
    config_parser = subparsers.add_parser(
        "create-config",
        help="Create a template config file"
    )
    config_parser.add_argument(
        "-o", "--output",
        type=str,
        default="./config.yaml",
        help="Output path for config file"
    )
    config_parser.set_defaults(func=handle_create_config)
    
    # create-linelist subcommand
    linelist_parser = subparsers.add_parser(
        "create-linelist",
        help="Create a template linelist file"
    )
    linelist_parser.add_argument(
        "-o", "--output",
        type=str,
        default="./linelist.dat",
        help="Output path for linelist file"
    )
    linelist_parser.set_defaults(func=handle_create_linelist)
    
    # create-csv subcommand
    csv_parser = subparsers.add_parser(
        "create-csv",
        help="Create a template CSV lookup table"
    )
    csv_parser.add_argument(
        "-o", "--output",
        type=str,
        default="./lookup.csv",
        help="Output path for CSV file"
    )
    csv_parser.set_defaults(func=handle_create_csv)
    
    # create-workspace subcommand
    workspace_parser = subparsers.add_parser(
        "create-workspace",
        help="Create a complete workspace with all template files"
    )
    workspace_parser.add_argument(
        "-d", "--dir",
        type=str,
        default="./paqs-workspace",
        help="Output directory for workspace"
    )
    workspace_parser.set_defaults(func=handle_create_workspace)

def handle_create_config(args: argparse.Namespace) -> int:
    """Create a template config file."""
    try:
        write_template("config.yaml", args.output)
        print(f"✓ Template config created: {args.output}")
        return 0
    except Exception as e:
        print(f"✗ Failed to create config: {e}", file=sys.stderr)
        return 1

def handle_create_linelist(args: argparse.Namespace) -> int:
    """Create a template linelist file."""
    try:
        write_template("linelist.dat", args.output)
        print(f"✓ Template linelist created: {args.output}")
        return 0
    except Exception as e:
        print(f"✗ Failed to create linelist: {e}", file=sys.stderr)
        return 1

def handle_create_csv(args: argparse.Namespace) -> int:
    """Create a template CSV lookup table."""
    try:
        write_template("lookup.csv", args.output)
        print(f"✓ Template CSV created: {args.output}")
        return 0
    except Exception as e:
        print(f"✗ Failed to create CSV: {e}", file=sys.stderr)
        return 1

def handle_create_workspace(args: argparse.Namespace) -> int:
    """Create a complete workspace with all templates."""
    try:
        write_workspace(args.dir)
        print(f"✓ Workspace created: {args.dir}")
        return 0
    except Exception as e:
        print(f"✗ Failed to create workspace: {e}", file=sys.stderr)
        return 1
```

## Implementation Phases

### Phase 1: Foundation
1. Create repository structure with `__init__.py` files
2. Create `pyproject.toml` with setuptools configuration
3. Add `data/templates/` folder with placeholder template files
4. Define `[project.scripts]` entry point: `quasar-paqs = "quasar_paqs.cli.main:main"`

### Phase 2: Core CLI Framework
4. Implement `cli/main.py` dispatcher and parser
5. Implement `config.py` — centralized config loader (YAML)
6. Implement `data/__init__.py` — template utilities (load/write templates)

### Phase 3: Subcommand Implementations
7. Implement `cli/subcommands/run.py`
8. Implement `cli/subcommands/templates.py`
9. Implement `cli/subcommands/validate.py`

### Phase 4: Testing & Documentation
10. Write integration tests for all subcommands
11. Create README with usage examples and installation instructions

## User Experience

With this design, users interact as follows:

```bash
# Installation
pip install quasar-paqs

# Show help
$ quasar-paqs --help
usage: quasar-paqs [-h] [--version] [--verbose] [--config CONFIG] 
                   {run,create-config,create-linelist,create-csv,create-workspace,validate} ...

positional arguments:
  {run,create-config,create-linelist,create-csv,create-workspace,validate}
                        Available subcommands
    run                 Fit spectral data in a directory
    create-config       Create a template config file
    create-linelist     Create a template linelist file
    create-csv          Create a template CSV lookup table
    create-workspace    Create a complete workspace with all template files
    validate            Validate input files and configurations

optional arguments:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  --verbose, -v         Enable verbose output
  --config CONFIG       Path to global config file (YAML)

# Create templates
$ quasar-paqs create-workspace -d my-project/
✓ Workspace created: my-project/

$ quasar-paqs create-config -o custom-config.yaml
✓ Template config created: custom-config.yaml

# Run fitting
$ quasar-paqs run /path/to/data -c custom-config.yaml -o results/
✓ Successfully processed 42 files
✓ Results saved to results/

# Validate
$ quasar-paqs validate custom-config.yaml
✓ Config is valid
```

## Key Design Principles

1. **Modularity**: Each subcommand is self-contained; `main.py` remains thin
2. **Extensibility**: Adding new tools requires only: create subcommand module + one import line
3. **Error Handling**: All handlers return int (0 = success, non-zero = error)
4. **Separation of Concerns**: Argument parsing ≠ business logic
5. **Reusability**: Config loading, template I/O, fitting logic live in separate modules

## Configuration System

The `config.py` module provides:
- YAML config loading from file or template
- Centralized schema/validation (using pydantic or manual validation)
- Shared parameters (input_dir, output_dir, etc.)
- Tool-specific parameter overrides via CLI flags

Example `config.yaml`:
```yaml
# Global parameters
input_dir: ./data
output_dir: ./results

# Fitting parameters
fitting:
  method: least_squares
  max_iterations: 1000

# Linelist parameters
linelist:
  path: ./linelist.dat
  redshift: 0.0
```

## Template System

The `data/__init__.py` module provides:
- `write_template(template_name: str, output_path: str)` — Write a single template to disk
- `write_workspace(workspace_dir: str)` — Create all templates in a directory
- Uses `importlib.resources` (Python 3.12+) to load templates from package data

Templates are stored as package data in `src/quasar_paqs/data/templates/` and embedded in the wheel distribution.

## Entry Point Configuration

In `pyproject.toml`:

```toml
[project.scripts]
quasar-paqs = "quasar_paqs.cli.main:main"
```

This allows users to invoke the CLI as:
```bash
quasar-paqs run /path/to/data
```

Or via Python module:
```bash
python -m quasar_paqs run /path/to/data
```

## Testing Strategy

- **Unit tests** for config loading, template I/O, and individual handlers
- **Integration tests** for full CLI invocations (e.g., `quasar-paqs run --help`)
- **Mock file I/O** for template creation tests
- Run with `pytest tests/`

## Next Steps for Implementation

1. Create the repository from scratch (or start with cookiecutter if available)
2. Implement Phase 1 (foundation) tasks
3. Implement Phase 2 (CLI framework) tasks
4. Implement Phase 3 (subcommands) with stubs for business logic
5. Add Phase 4 (tests and documentation)
6. Polish and prepare for initial release

---

**Last Updated**: 2026-09-16  
**Decision**: Flat subcommand structure preferred over nested
