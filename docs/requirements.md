# Requirements & Setup

Environment setup and dependency management.

## Dependency Files

The project uses three separate requirements files for different purposes:

### requirements-core.txt
Core ML stack dependencies that are stable and rarely change across projects:
- **PyTorch** (torch, torchvision) - Deep learning framework
- **PyTorch Lightning** - Training framework for cleaner code
- **Hydra & OmegaConf** - Configuration management
- **NumPy, Pandas, scikit-learn** - Data & scientific computing
- **Weights & Biases** - Experiment tracking & logging
- **python-dotenv, PyYAML** - Utilities

Install with: `pip install -r requirements-core.txt`

### requirements.txt
Project-specific dependencies (empty by default).
Add dataset libraries, domain-specific packages, or other project-only dependencies here.

Install with: `pip install -r requirements.txt`

### requirements-dev.txt
Development and testing tools (optional, only needed for development):
- **pytest** - Testing framework
- **ruff, black, isort, flake8** - Code linting & formatting
- **mypy** - Type checking
- **sphinx** - Documentation generation
- **pre-commit** - Git hooks for automation

Install with: `pip install -r requirements-dev.txt`

## Installation

To be filled: Step-by-step installation instructions.

## Environment Variables

To be filled: Document required and optional environment variables (.env file).

## Development Setup

To be filled: Development environment configuration and tool setup.

## Troubleshooting

To be filled: Common issues and solutions.
