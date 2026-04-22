# SMB Recommendation System

A GPU-accelerated sequential recommendation engine for SmartBuy merchant transactions. The system uses NVTabular for large-scale data preprocessing and Transformers4Rec for training transformer-based next-item prediction models on temporal customer behavior sequences.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [nvt\_preprocessing](#nvt_preprocessing)
  - [Folder Structure](#nvt-folder-structure)
  - [Pipeline Workflow](#pipeline-workflow)
  - [Custom NVTabular Operators](#custom-nvtabular-operators)
  - [Configuration](#nvt-configuration)
  - [Utilities](#nvt-utilities)
- [model\_training](#model_training)
  - [Folder Structure](#model-folder-structure)
  - [Training Workflow](#training-workflow)
  - [Model Architecture](#model-architecture)
  - [Custom Loss & Metrics](#custom-loss--metrics)
  - [Configuration](#model-configuration)
- [End-to-End Workflow](#end-to-end-workflow)

---

## Overview

The system processes e-commerce merchant transaction data to build sequential recommendation models. It operates in two stages:

1. **NVT Preprocessing** -- Transforms raw transaction-level data into grouped, normalized customer behavior sequences using GPU-accelerated NVTabular pipelines on a Dask-CUDA cluster.
2. **Model Training** -- Trains transformer models (XLNet, ALBERT, BERT, GPT-2) on the preprocessed sequences using a day-wise temporal training strategy with custom weighted loss functions.

**Data domain**: Customer interactions with products -- clicks, activations, offers, page views -- enriched with payment method features, merchant segments, geographic info, and financial metrics (balances, transaction counts/sums, etc.).

---

## Architecture

```
Raw Parquet Data
       |
       v
 nvt_preprocessing/
       |
  [Pre-GroupBy Pipeline]
  - Filter events, deduplicate, categorify
  - Normalize continuous features (min-max)
  - Cyclic-encode time features
  - Create loss masks
       |
  [GroupBy Pipeline]
  - Group by customer_id + month
  - Aggregate columns into sequences (lists)
  - Slice to max sequence length
  - Filter by min sequence length
       |
       v
 Processed Parquet + Merlin Schema (.pbtxt)
       |
       v
 model_training/
       |
  - Load day-partitioned data
  - Train transformer (XLNet/ALBERT/etc.)
  - Day-wise: train on day N, evaluate on day N+1
  - Custom weighted cross-entropy loss
  - Checkpoint per day, early stopping
       |
       v
 Trained Model + Evaluation Metrics
```

---

## nvt_preprocessing

GPU-accelerated data preprocessing built on [NVTabular](https://github.com/NVIDIA-Merlin/NVTabular) and [Dask-CUDA](https://github.com/rapidsai/dask-cuda).

### NVT Folder Structure

```
nvt_preprocessing/
├── config.py                        # Central configuration (paths, params)
├── pipeline.py                      # Main preprocessing pipeline
├── columns_workflow.py              # Column group definitions (~300 features)
├── nvt_custom_operations.py         # Custom GPU operators
├── util_functions.py                # Cluster setup, schema validation, utilities
├── env.yml                          # Environment variables (UCX, RMM, Dask)
├── part_size_Benchmarking.py        # Parquet partition benchmarking
├── main.ipynb                       # Main execution notebook
├── nvt-cust.ipynb                   # Custom operations notebook
├── test2.parquet                    # Test data
└── preprocessing/                   # Specialized pipeline variants
    ├── training_pipeline.py         # Training-specific pipeline
    ├── inference_pipeline.py        # Inference pipeline (no month grouping)
    ├── pipeline_test.py             # Pipeline tests
    ├── columns_workflow.py          # Column definitions (copy)
    ├── nvt_custom_operations.py     # Custom operators (copy)
    ├── util_functions.py            # Utilities (copy)
    ├── env.yml                      # Environment config (copy)
    ├── nvt-cust.ipynb               # Custom operations notebook
    └── nvt-cust_triton.ipynb        # Triton inference variant
```

### Pipeline Workflow

The pipeline is defined in `pipeline.py` and executes in two stages:

#### Stage 1: Pre-GroupBy Pipeline (`pre_groupby_pipeline`)

Row-level transformations applied before aggregation:

| Step | Description |
|------|-------------|
| Event Filtering | Removes "Other/Context_Event" products; optionally keeps clicks only |
| GPU Deduplication | Removes consecutive duplicate product views per session |
| Categorical Encoding | Categorifies categorical columns, casts to `int64` |
| Date Normalization | Converts dates to unix epoch, normalizes to `[0, 1]` as `float32` |
| Continuous Normalization | Min-max scaling to `[0, 1]` as `float32` |
| Time Feature Engineering | Extracts hour from timestamps, applies cyclic (sine) encoding |
| Month Index | Computes relative month index from dataset minimum |
| Loss Mask | Combines `is_activation` + `is_click` into a training signal mask |
| Customer ID Cast | Casts `customer_id` to `int64` |

#### Stage 2: GroupBy Pipeline (`groupby_pipeline_cust`)

Aggregates row-level data into per-customer, per-month sequences:

| Step | Description |
|------|-------------|
| GroupBy | Groups by `customer_id` + `evnt_ts_month_index` |
| List Aggregation | Aggregates most columns into ordered lists (sequences) |
| Special Aggregations | `product_name`: list + count; `is_activation`: list + max; `month_index`: min |
| Item ID Tagging | Tags product sequences as item IDs for the recommendation model |
| Label Creation | `is_activation_max` serves as the binary target |
| Sequence Slicing | Keeps the last N items (configurable, default 512) |
| Length Filtering | Removes sequences shorter than `min_length` (default 5) |

### Custom NVTabular Operators

Defined in `nvt_custom_operations.py`:

| Operator | Purpose |
|----------|---------|
| `GPUConsecutiveDedupOp` | Removes consecutive duplicate items in sessions (GPU-accelerated via cuDF) |
| `NormalizeMinMaxcust` | Custom min-max normalization with fit/transform phases; handles zero-range edge cases |
| `MonthIndexOp` | Converts timestamps to relative month indices from dataset minimum (Dask-aware) |
| `DayindexOp` | Converts timestamps to relative day indices |
| `GetcyclicOp` | Sine encoding for hour-of-day features (period=24) |
| `CyclicEncodingOp` | General-purpose sine/cosine encoding with configurable period maps |
| `ExtractHourOp` | Extracts hour component from timestamps |
| `LossmaskOp` | Combines activation + click columns into a single loss mask |
| `CustidOp` | Casts customer ID to int64 |

### NVT Configuration

**`config.py`** -- Central parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `gby` | `customer_id` | Groupby dimension (or `sessn_id`) |
| `min_length` | 5 | Minimum sequence length |
| `max_length` | 128 | Maximum sequence length |
| `cl_only` | `False` | Filter to click events only |
| `PART_SIZE_FINAL` | `100GB` | Final parquet partition size |
| `ROW_GROUP_SIZE_FINAL` | `1e6` | Final parquet row group size |
| `dask_workdir` | `/projects/merlin/dask_wdir` | Dask scratch space |

**`env.yml`** -- Cluster environment:

| Variable | Value | Description |
|----------|-------|-------------|
| `ucx_tls` | `tcp,cuda_copy,cuda_ipc` | UCX transport layers |
| `rmm_pool_size` | `3GB` | GPU memory pool per worker |
| `dashboard_port` | `8787` | Dask dashboard port |

**`columns_workflow.py`** -- Defines ~300+ features across groups:
- **Categorical (~42)**: product_name, event_name, payment methods, user segments, flags
- **Continuous (~250+)**: transaction counts, balances, reversals, withdrawals, volume metrics across time windows
- **Date (8)**: account creation, signup, last seen, etc.

### NVT Utilities

`util_functions.py` provides:

- **Cluster Setup**: `setup_cluster()` creates a Dask `LocalCUDACluster` with UCX protocol, NVLink, and RMM pool allocation
- **Schema Validation**: `validate_schema()` checks DataFrame columns against expected types
- **Data Splitting**: `save_time_based_splits_dask()` creates train/valid/test splits (70/10/20) by time partition
- **Memory Tuning**: `tune_gc_for_high_memory()` optimizes garbage collection for 512GB+ systems
- **Schema Extraction**: `extract_features_from_pbtxt()` parses Merlin protobuf schema files to CSV

---

## model_training

Sequential recommendation model training built on [Transformers4Rec](https://github.com/NVIDIA-Merlin/Transformers4Rec) and PyTorch.

### Model Folder Structure

```
model_training/
├── main.py                          # Entry point
├── config.yml                       # Smoke test configuration
├── config-Copy1.yml                 # Production configuration
├── logs/                            # Training logs & checkpoints
│   └── YYYYMMDD_HHMMSS/
│       ├── main_<experiment>.log
│       ├── tensorboard/
│       └── checkpoint_day_N/
├── model/                           # Core model code
│   ├── model_definition.py          # Model architecture builder
│   ├── model_trainer.py             # Custom trainer with weighted loss
│   ├── model_training.py            # Day-wise training orchestration
│   ├── helper.py                    # Feature building, schema, utilities
│   └── registry.py                  # Task, metric & transformer registries
└── utils/                           # Config & logging
    ├── config_loader.py             # YAML config loading with dot-path access
    └── logger.py                    # Distributed-aware logging setup
```

### Training Workflow

The system uses a **day-wise temporal training strategy**:

```
For each day_index in [start_day ... end_day]:
    1. Load train data from day_index partition (parquet)
    2. Load eval data from day_index+1 partition
    3. Train model on day_index data (multiple epochs)
    4. Save checkpoint: checkpoint_day_{day_index}/
    5. Evaluate on day_index+1 data
    6. Clear GPU memory
    7. Accumulate metrics (NDCG, Recall, Precision @ 1,3,5)

Final: Save final model, compute averaged metrics across all days
```

Orchestrated by `model/model_training.py` (`CustomTrainer._train_day_wise()`).

### Model Architecture

Defined in `model/model_definition.py` (`RecommenderModel` class):

```
Input Parquet → Merlin Schema
       |
CustomTabularSequenceFeatures
  - Categorical embeddings (dim=64)
  - Continuous feature projection (MLP)
  - Feature aggregation (concat)
  - Causal Language Modeling (CLM) masking
       |
MLPBlock (optional projection)
       |
TransformerBlock
  - Architecture: XLNet / ALBERT / BERT / GPT-2
  - d_model=512, n_head=4, n_layer=2
       |
Prediction Head(s)
  - NextItemPredictionTask (primary)
  - BinaryClassificationTask (optional, for is_activation)
```

**Supported transformer architectures** (registered in `model/registry.py`):

| Architecture | Config Class | Typical Use |
|---|---|---|
| XLNet | `XLNetConfig` | Production (autoregressive with permutation) |
| ALBERT | `AlbertConfig` | Smoke tests / lighter models |
| BERT | `BertConfig` | Bidirectional experiments |
| GPT-2 | `GPT2Config` | Autoregressive experiments |

### Custom Loss & Metrics

**Custom Loss** (`model/model_trainer.py` -- `T4RCustomTrainer.compute_loss()`):

```
total_loss = nip_weight * next_item_loss + (1 - nip_weight) * binary_loss

Where:
  - next_item_loss = weighted cross-entropy on item predictions
    Weights from loss_mask_list: categories map to [0.0, 1.0, 2.0]
  - binary_loss = BCE on is_activation prediction (optional)
  - nip_weight = 0.6 (configurable)
```

The loss weighting allows the model to emphasize certain interaction types (e.g., activations weighted at 2x vs. regular clicks at 1x, and noise/padding at 0x).

**Evaluation Metrics** (computed per day, then averaged):

| Metric | Top-K Values |
|--------|-------------|
| NDCG | @1, @3, @5 |
| Recall | @1, @3, @5 |
| Precision | @1, @3, @5 |

For optional binary classification task: Precision, Recall, Accuracy, F1Score.

### Model Configuration

Key parameters from the YAML config files:

| Parameter | Smoke Test (`config.yml`) | Production (`config-Copy1.yml`) |
|-----------|--------------------------|--------------------------------|
| Architecture | ALBERT | XLNet |
| d_model | 512 | 512 |
| n_head | 4 | 4 |
| n_layer | 4 | 2 |
| max_sequence_length | 50 | 128 |
| embedding_dim | 64 | 64 |
| d_output | 64 | 64 |
| learning_rate | 5e-6 | 5e-5 |
| batch_size | 1 | 4096 |
| num_epochs | 1 | 10 |
| early_stopping | -- | Enabled (patience=3) |
| loss_weights | [0, 1, 10] | [0, 1, 2] |
| nip_weight | 0.6 | 0.6 |
| Input features | minimal | 30+ features |

### Output Structure

```
logs/
└── YYYYMMDD_HHMMSS/
    ├── main_<experiment_name>.log         # Training log
    ├── tensorboard/                       # TensorBoard events
    ├── <experiment_name>/
    │   ├── checkpoint_day_1/
    │   │   └── pytorch_model.bin
    │   ├── checkpoint_day_2/
    │   │   └── pytorch_model.bin
    │   └── <experiment>_final_model/
    │       └── pytorch_model.bin
    └── evaluation_summary_HHMMSS.csv      # Metrics & hyperparameters
```

---

## End-to-End Workflow

```
1. Configure nvt_preprocessing/config.py
   - Set data paths, groupby dimension, sequence length bounds

2. Run NVT preprocessing (via main.ipynb or training_pipeline.py)
   - Initializes Dask-CUDA cluster with UCX + RMM
   - Executes pre-groupby pipeline (filtering, encoding, normalization)
   - Executes groupby pipeline (sequence aggregation, slicing)
   - Writes processed parquet + Merlin schema (.pbtxt)

3. Configure model_training/config-Copy1.yml
   - Set schema path pointing to NVT output
   - Choose transformer architecture, hyperparameters
   - Define input features, loss weights, training day range

4. Run model training (python main.py)
   - Loads config and builds model from Merlin schema
   - Trains day-wise with checkpointing and early stopping
   - Evaluates with ranking metrics (NDCG, Recall, Precision)
   - Saves final model and metrics summary
```

---

## Key Dependencies

| Package | Purpose |
|---------|---------|
| [NVTabular](https://github.com/NVIDIA-Merlin/NVTabular) | GPU-accelerated feature engineering |
| [Transformers4Rec](https://github.com/NVIDIA-Merlin/Transformers4Rec) | Sequential recommendation models |
| [Merlin Core](https://github.com/NVIDIA-Merlin/core) | Schema, dataset, and I/O utilities |
| [RAPIDS cuDF](https://github.com/rapidsai/cudf) | GPU DataFrames |
| [Dask-CUDA](https://github.com/rapidsai/dask-cuda) | Multi-GPU distributed computing |
| [PyTorch](https://pytorch.org/) | Deep learning framework |
| [HuggingFace Transformers](https://github.com/huggingface/transformers) | Transformer architectures (XLNet, ALBERT, etc.) |
