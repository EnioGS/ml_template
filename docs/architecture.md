# Architecture

System architecture and high-level design decisions.

## Overview

To be filled: Describe the overall system architecture, design patterns, and key components.

## Core Concepts

To be filled: Explain fundamental concepts and how components interact.

## Data Flow

To be filled: Describe how data flows through the system during training and evaluation.

## Extension Points

To be filled: Explain where and how to extend the system for new models, datasets, or training procedures.

## Transformer Architecture

### Overview

The template supports three Transformer variants following standard NLP architectures:

1. **Encoder-only** (BERT-like) - For classification, sequence labeling, masked language modeling
2. **Decoder-only** (GPT-like) - For causal language modeling, text generation
3. **Encoder-Decoder** (T5-like) - For seq2seq tasks like translation, summarization

### Code Organization

The template separates concerns across three directories (implementation left to user):

**src/modules/** - Pure PyTorch implementations (no Lightning)
- Transformer encoder, decoder, seq2seq architectures
- Attention mechanisms, positional encodings, feed-forward networks
- Stateless `nn.Module` classes with no training logic

**src/models/** - Lightning module wrappers
- LightningModule classes that wrap the pure PyTorch modules
- Training/validation/test logic, loss computation, metrics
- Optimizer and scheduler configuration via `configure_optimizers()`

**src/data/** - Data loading
- LightningDataModule for tokenization and data loading
- Uses HuggingFace tokenizers for text processing
- Handles padding, truncation, dynamic batching

### Config Pattern (Nested Backbone)

Transformer configs use a nested backbone pattern for clean separation:

```yaml
# configs/model/transformer_encoder.yaml
_target_: src.models.transformer_module.TransformerEncoderLitModule

# Task-level concerns (LightningModule)
task: classification
num_classes: 2
label_smoothing: 0.0
pooling_strategy: "cls"

# Architecture concerns (pure PyTorch module)
backbone:
  _target_: src.modules.transformer.TransformerEncoder
  vocab_size: 30522
  d_model: 768
  n_layers: 12
  # ... architecture hyperparams
```

When implementing the LightningModule, instantiate the backbone via:
```python
self.backbone = hydra.utils.instantiate(cfg.backbone)
```

And configure the scheduler manually in `configure_optimizers()`:
```python
from transformers import get_cosine_schedule_with_warmup

def configure_optimizers(self):
    optimizer = # ... instantiate optimizer

    if self.hparams.scheduler.name == "cosine_warmup":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.scheduler.num_warmup_steps,
            num_training_steps=self.hparams.scheduler.num_training_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.hparams.scheduler.interval,
                "frequency": self.hparams.scheduler.frequency,
            }
        }
```

### Critical Separation: Tokenization vs Architecture

**Tokenization (data concern) → configs/datamodule/text.yaml:**
- tokenizer_name: "bert-base-uncased"
- max_seq_len: 512 (for truncation)
- padding, truncation, special tokens

**Architecture (model concern) → configs/model/transformer_encoder.yaml:**
- vocab_size: 30522 (embedding table size)
- d_model: 768 (hidden dimension)
- max_seq_len: 512 (positional encoding limit)
- attention_dropout, activation

The same `max_seq_len` value appears in both configs but serves different purposes:
- **Datamodule**: Maximum tokenized sequence length (truncates longer inputs)
- **Model**: Maximum positional encoding range (defines learned position embeddings)

### Gradient Clipping

Transformers typically require gradient clipping for stable training. This is configured in `configs/trainer/transformer.yaml`:

```yaml
gradient_clip_val: 1.0
gradient_clip_algorithm: "norm"
```

Vision models (CNNs) rarely need clipping, so `configs/trainer/default.yaml` sets it to 0.0.

### Warmup Scheduling

Transformer training typically uses learning rate warmup followed by cosine decay. The scheduler config stores parameters; the LightningModule calls `get_cosine_schedule_with_warmup()` in `configure_optimizers()`:

```yaml
# configs/scheduler/warmup_cosine.yaml
name: cosine_warmup
num_warmup_steps: 1000
num_training_steps: ${trainer.max_steps}
interval: "step"
frequency: 1
```

This pattern avoids Hydra trying to instantiate a function that requires the optimizer instance as an argument.
