# quasar-typing

A small Python project containing utilities for the Quasar project. 

## Features

## Installation

### Development Installation

```bash
pip install -e ".[dev]"
```

### With Documentation Tools

```bash
pip install -e ".[docs]"
```

## Development

### Running Tests

```bash
pytest
```

### Code Quality

```bash
black src/quasar_utils tests/
isort src/quasar_utils tests/
flake8 src/quasar_utils tests/
mypy src/quasar_utils
```

## Project Structure

```
quasar-utils/
├── src/quasar_utils/          # Main package
├── tests/                     # Unit tests
├── docs/                      # Documentation
├── pyproject.toml             # Project configuration
└── README.md                  # This file
```

## License

MIT License - see LICENSE file for details