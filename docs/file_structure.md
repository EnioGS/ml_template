# File Structure

This document describes the directory layout and purpose of each component in the ML project template.

## Root Level

```
ml-project/
├── configs/              # Hydra configuration files
├── src/                  # Main Python package
├── data/                 # Datasets (local, gitignored)
├── logs/                 # Training logs and checkpoints (gitignored)
├── tests/                # Unit and integration tests
├── scripts/              # Helper scripts (bash, slurm, etc.)
├── docs/                 # Project documentation
├── .env                      # Environment variables (gitignored)
├── .gitignore                # Git ignore rules
├── pyproject.toml            # Project metadata and dependencies
├── requirements-core.txt     # Core ML stack (stable, rarely changes)
├── requirements.txt          # Project-specific dependencies (empty by default)
├── requirements-dev.txt      # Development tools (testing, linting, formatting)
└── README.md                 # Project overview
```

## configs/ - Hydra Configuration

All training configuration is managed through Hydra's modular config system. Each subdirectory represents a configuration group.

```
configs/
├── config.yaml           # Root composition (defaults, metadata)
├── paths/
│   └── default.yaml      # I/O paths (data_dir, log_dir, etc.)
├── trainer/
│   └── default.yaml      # Trainer behavior & hardware
├── model/
│   ├── resnet.yaml       # ResNet architecture example
│   └── ...               # Other model variants
├── datamodule/
│   ├── cifar10.yaml      # CIFAR-10 dataset example
│   └── ...               # Other datasets
├── optimizer/
│   ├── adam.yaml         # AdamW optimizer example
│   └── ...               # Other optimizers
├── scheduler/
│   ├── cosine.yaml       # Cosine annealing scheduler example
│   └── ...               # Other schedulers
├── callbacks/
│   └── default.yaml      # Training callbacks (checkpoint, early stop, etc.)
├── logger/
│   ├── wandb.yaml        # Weights & Biases logger example
│   └── ...               # Other loggers
├── experiment/
│   ├── exp1.yaml         # Experiment override example
│   └── ...               # Other experiments
└── hydra/
    └── default.yaml      # Hydra runtime & output directories
```

### Config Ownership Guide

**configs/config.yaml** (Root composition)
- defaults composition (which model/datamodule/trainer to use)
- global run metadata (seed, task_name, tags, ckpt_path, resume)
- shared toggles (debug, compile, float32_matmul_precision)

**configs/paths/** (I/O paths only)
- data_dir, log_dir, output_dir, ckpt_dir, cache_dir
- Environment variable references with defaults

**configs/trainer/** (Trainer behavior + hardware)
- compute/runtime: accelerator, devices, strategy, precision
- training loop: max_epochs, max_steps, val_check_interval
- performance: accumulate_grad_batches, gradient_clip_val, deterministic
- logging: log_every_n_steps, enable_progress_bar, num_sanity_val_steps

**configs/model/** (LightningModule + architecture)
- _target_ for your LightningModule
- architecture hyperparams (backbone, depth, hidden sizes, dropout, pretrained)
- loss/metrics (label_smoothing, etc.)
- model-level regularization

**configs/datamodule/** (Dataset + loading + transforms)
- _target_ for your LightningDataModule
- dataset identity & splits (val_split, test_split)
- data paths (data_dir from paths config)
- batch & loader (batch_size, num_workers, pin_memory)
- transforms/augmentations

**configs/optimizer/** (Optimizer only)
- optimizer _target_
- hyperparams (lr, weight_decay, betas, momentum, etc.)

**configs/scheduler/** (Scheduler only)
- scheduler _target_
- hyperparams (T_max, eta_min, warmup_steps)
- interval & frequency

**configs/callbacks/** (Training-time "extras")
- checkpointing policy (monitor, save_top_k, mode, filename)
- early stopping (patience, min_delta)
- lr monitor, progress bar, etc.

**configs/logger/** (Logging backend)
- logger _target_
- project/run identifiers, tags
- online/offline toggles

**configs/hydra/** (Hydra runtime)
- output directory patterns
- sweep directory patterns
- job naming & chdir settings

**configs/experiment/** (Experiment overrides)
- Composition of which configs to override
- Value overrides relative to defaults

## src/ - Main Python Package

Contains all project code organized by responsibility.

```
src/
├── __init__.py           # Package initialization
├── train.py              # Main training entrypoint
├── eval.py               # Evaluation script
├── predict.py            # Inference script
├── export.py             # Model export script
├── data/
│   ├── __init__.py
│   └── cifar10.py        # LightningDataModule implementations
├── models/
│   ├── __init__.py
│   └── resnet_module.py  # LightningModule implementations
├── modules/              # Pure torch nn.Modules (no Lightning)
│   └── __init__.py
├── losses/               # Custom loss functions
│   └── __init__.py
├── metrics/              # Custom metrics
│   └── __init__.py
├── callbacks/            # Custom Lightning callbacks
│   └── __init__.py
├── utils/                # Utilities
│   ├── __init__.py
│   ├── instantiators.py  # Hydra instantiate helpers
│   ├── logging.py        # Logging utilities
│   └── seed.py           # Seed & reproducibility
└── eval/                 # Evaluation utilities
    └── __init__.py
```

### Key Distinction

- **src/models/**: LightningModule classes that implement training/validation logic
- **src/modules/**: Pure PyTorch nn.Module classes for architectures (stateless, no Lightning)
- **src/data/**: LightningDataModule classes that handle data loading
- **src/callbacks/**: Custom Lightning callbacks (e.g., visualization, custom logging)
- **src/losses/**: Custom loss implementations
- **src/metrics/**: Custom metric implementations
- **src/utils/instantiators.py**: Helpers to instantiate Hydra configs into objects

## data/ - Datasets

Local dataset storage (gitignored to prevent bloat).

```
data/
├── raw/                  # Original, immutable data
├── processed/            # Processed/transformed data ready for training
└── external/             # External datasets or resources
```

## logs/ - Training Outputs

Training logs, checkpoints, and artifacts (gitignored).

```
logs/
├── train/                # Training runs
│   └── YYYY-MM-DD_HH-MM-SS/  # Timestamp-based directories per run
│       ├── checkpoints/  # Model checkpoints
│       └── ...           # Hydra outputs, metrics, etc.
└── multirun/             # Hyperparameter sweeps
    └── YYYY-MM-DD_HH-MM-SS/  # Sweep runs
        └── 0,1,2.../     # Individual job directories
```

## tests/

Unit and integration tests.

```
tests/
├── test_datamodule.py    # Tests for datamodule logic
├── test_model.py         # Tests for model logic
└── ...                   # Other tests
```

## scripts/

Standalone helper scripts for training, evaluation, slurm job submission, etc.

```
scripts/
├── train.sh              # Example training script
└── ...                   # Other helper scripts
```

## docs/

Documentation files.

```
docs/
├── architecture.md       # System architecture & design decisions
├── requirements.md       # Dependency & environment setup
├── file_structure.md     # This file - directory structure explanation
└── adrs/                 # Architecture Decision Records
    └── ...               # ADR files explaining key decisions
```

## Configuration Files

**pyproject.toml**
- Project metadata (name, version, author, license)
- Dependency groups (core, dev, optional)
- Build system configuration
- Tool configurations (pytest, black, mypy, etc.)

**requirements-core.txt**
- Stable core dependencies (torch, lightning, hydra, numpy, wandb, etc.)
- Rarely changed across projects
- Install: `pip install -r requirements-core.txt`

**requirements.txt**
- Project-specific dependencies (empty by default)
- Dataset libraries, domain-specific packages, etc.
- Install: `pip install -r requirements.txt`

**requirements-dev.txt**
- Development tools (pytest, ruff, black, mypy, sphinx, pre-commit, etc.)
- Optional, only needed for development
- Install: `pip install -r requirements-dev.txt`

**.env**
- Environment variables (DATA_DIR, LOG_DIR, CACHE_DIR, WANDB_API_KEY, etc.)
- Gitignored - copy .env.example and populate locally
- Loaded via python-dotenv in training scripts

**.gitignore**
- Ignores: data/, logs/, .cache/, *.pyc, __pycache__, .env, etc.
- Prevents large files and secrets from version control

## Typical Usage

```bash
# Install dependencies
pip install -r requirements-core.txt
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Load environment variables
source .env

# Train with default config
python src/train.py

# Train with specific experiment
python src/train.py --config-name=config experiment=exp1

# Evaluate
python src/eval.py model=resnet

# Run inference
python src/predict.py

# Export model
python src/export.py

# Run tests
pytest tests/
```

## Quick Ownership Checklist

Use this to decide where to put your code:

- **configs/**: Hydra config overrides & experiments
- **src/models/**: Lightning modules with training logic
- **src/modules/**: Reusable PyTorch architecture blocks
- **src/data/**: Data loading & preprocessing (Lightning)
- **src/losses/**: Custom loss functions
- **src/metrics/**: Custom metric functions
- **src/callbacks/**: Training-time side effects (visualization, logging)
- **src/utils/**: Shared helpers & utilities
- **src/eval/**: Evaluation-time logic
- **tests/**: Unit & integration tests
- **scripts/**: Standalone runnable scripts
- **docs/**: Design decisions & setup instructions
