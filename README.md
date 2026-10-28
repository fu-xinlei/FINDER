# FINDER

Code for **FINDER: A Function-Centered Relational Graph Neural Network for Interpretable Drug Repositioning and Mechanism Discovery**.

## Setup

Linux, Bash, and a CUDA 12.8-compatible NVIDIA driver are required.

```bash
conda env create -f environment.yml
conda activate finder-paper
```

Dependency details: [ENVIRONMENT.md](ENVIRONMENT.md).

## Data

Place these files in `../data/inputs/finder/`, relative to this code repository:

- `hetero_graph.pt`
- `all_data_pos_neg_post_replacement_8.pt`

Set `FINDER_DATA_DIR` to use another input directory. Relative paths are resolved from the code repository root.

## Run

```bash
bash scripts/run_finder_baseline_comparison.sh
```

Ten folds, split seed 42, training seed 93, and up to 50 epochs. Folds 1-4 and 5-10 run in separate processes, with training seed 93 reset before fold 5. Checkpoint resume is not supported.

Outputs: `runs/baseline_comparison/folds1to4/` and `runs/baseline_comparison/folds5to10/`.

## Results and other code

Reference metrics: [results/table2_finder_fold_metrics.csv](results/table2_finder_fold_metrics.csv). Use `val_auc`, `val_aupr`, and classification columns ending in `_prob_0_5` (probability threshold 0.5).

Preprocessing scripts: `finder/preprocessing/`.

Tests:

```bash
python -m unittest discover -s tests -v
```
