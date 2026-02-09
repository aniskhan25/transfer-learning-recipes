# Supplemental Transfer: comprehensive transfer-learning recipes

This folder mirrors the `supplemental_1` philosophy: config-driven experiments, reusable methods, and notebooks that combine hypothesis, ablations, and stability instrumentation.

The corresponding blog post keeps intuition simple. The notebooks here intentionally go deeper.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Main entrypoint

```bash
python scripts/run_transfer.py --config configs/transfer_core_related.yaml --use-progress
```

Outputs land in:

- `outputs/logs/*.csv`
- `outputs/figures/*.png`

## Notebook order

- `01_baseline_from_scratch.ipynb`
- `02_transfer_core_loop.ipynb`
- `03_safe_transfer_staged_adaptation.ipynb`
- `04_naive_transfer_and_forgetting.ipynb`
- `05_related_vs_unrelated_tasks.ipynb`
- `06_stability_mismatch_proxy.ipynb`
- `07_modern_methods_and_label_budget.ipynb`
- `08_design_pattern_and_checklist.ipynb`

Each notebook follows the same runtime toggle pattern used in `recursive-training-recipes`:

- `FAST_DEV_RUN = False` for full experiments
- set `FAST_DEV_RUN = True` for quick sanity runs (smaller subsets and fewer epochs)

## What is instrumented

- target accuracy (`target_test_acc`)
- source retention accuracy (`source_retention_acc`) for transfer methods
- feature drift proxy (`feature_drift`)
- gradient norm proxy (`grad_norm`)

These signals are used to connect the implementation back to the blog's stability narrative.

## Available transfer configs

- `configs/transfer_core_related.yaml`
- `configs/transfer_safe_vs_naive.yaml`
- `configs/transfer_unrelated_failure.yaml`

You can duplicate these configs to run additional source-target class pairings and target label budgets.
