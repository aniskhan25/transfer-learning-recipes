# Source Layout

`src/` contains all reusable implementation code for the recipes.

## Package map

- `data/`: dataset loading, transforms, and sampling splits.
- `methods/`: algorithm implementations and adaptation loops.
- `models/`: model definitions used by experiments.
- `utils/`: logging, plotting, progress, seeding, notebook runtime helpers.
- `train/`: legacy compatibility namespace (`trainer.py` delegates to `methods.supervised`).
- `eval/`: small evaluation helpers.

## Organization rules

- Put recipe logic in `methods/`, not notebooks.
- Keep `scripts/` as orchestration entrypoints only.
- Reuse `utils/` for shared behavior; avoid duplicating helper code across notebooks.
- Prefer extending existing modules over creating one-off files.

