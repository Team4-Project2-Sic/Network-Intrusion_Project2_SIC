# Network-Intrusion_Project2_SIC

## Leakage-safe evaluation updates

The notebook now evaluates models with a group-aware workflow to reduce hidden leakage risk:

- Replaced random `train_test_split` evaluation with `GroupKFold` (5 folds).
- Built group keys from session/burst columns so related flows stay in the same fold.
- Moved preprocessing into reusable sklearn/imblearn pipelines:
  - Zero-variance feature removal
  - High-correlation feature removal
  - `Port_Category` feature drop
  - Mutual-information top-k feature selection
  - Standard scaling
- Kept imbalance handling inside the pipeline path (`none`, `class_weight`, `smote`, `undersample`).
- Updated model comparison to report out-of-fold metrics from grouped cross-validation.

If no session/burst identifier column exists in the dataset, the notebook now raises a clear error to prevent unsafe evaluation.
