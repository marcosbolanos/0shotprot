# 0shotProt

This repository contains the reproducible implementation and evaluation pipeline for 0shotProt, together with ProSpero and ProSST as pinned submodules.

## Experimental protocol

0shotProt starts from a known wild-type sequence and its measured fitness. These form the experimental baseline. No other task-specific sequence or fitness value from the benchmark initialization set is available to optimization.

Each round:

1. masks four positions using positional feedback, middle-entropy exploration, and anti-collapse sampling;
2. decodes candidates sequentially with a masked protein language model;
3. ranks candidates by sequential conditional log-odds relative to the current incumbent;
4. queries the oracle for the top `K` candidates;
5. optionally adapts the model with signed advantage-weighted masked reconstruction and KL regularization;
6. uses the best observed sequence, including the WT baseline, as the next incumbent.

The supported 0shotProt configurations are:

- fine-tuned ProSST with a charge-restricted decoding vocabulary;
- fine-tuned EvoDiff with the same optimization protocol;
- no fine-tuning, as the primary ablation;
- fine-tuned ProSST with an unrestricted 20-amino-acid vocabulary.

## Setup

```bash
git clone --recurse-submodules <repository-url>
cd ProSpero
uv sync
```

Download the benchmark oracles if they are not already present:

```bash
./bash/download_oracles.sh
```

ProSST structure tokens are expected under `outputs/prosst_structure_tokens/`.

## Reproduction

Inspect every command without running it:

```bash
uv run python scripts/reproduce.py --dry-run
```

Run the complete recipe on a GPU pinned by UUID:

```bash
uv run python scripts/reproduce.py \
  --gpu GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

Useful filters:

```bash
uv run python scripts/reproduce.py \
  --stages prosst_online_adaptation prosst_without_adaptation \
  --tasks AAV LGK \
  --budgets 8 \
  --seeds 1 2
```

Every run is written to `outputs/reproduction/<timestamp>/` with configuration, commands, logs, results, completion markers, traces, and generated plots.

## Validation

```bash
uv run pytest -q
uv run ruff check scripts src/prospero tests
uv run pyright
```

## Repository layout

- `scripts/reproduce.py`: human-readable paper recipe.
- `src/prospero/reproduction/`: command construction and orchestration.
- `src/prospero/runners/`: experiment and plotting entry points.
- `src/prospero/`: pinned ProSpero fork and 0shotProt implementation.
- `src/prosst/`: pinned ProSST fork.
- `datasets/`, `oracles/`: benchmark inputs.
- `outputs/reproduction/`: timestamped experiment artifacts.
