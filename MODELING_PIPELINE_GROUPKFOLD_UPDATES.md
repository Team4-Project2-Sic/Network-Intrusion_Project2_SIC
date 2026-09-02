# Modeling and Validation Updates

## What was changed

1. **Group-aware validation with `GroupKFold`**
   - Replaced row-based splitting with `GroupKFold(n_splits=5)`.
   - Added a `build_flow_groups` function that builds groups from session/burst identifiers when available.
   - If explicit session/burst columns are not present, grouping falls back to flow-level identifiers (5-tuple style fields) and timestamp buckets.
   - Added a fold preview table that confirms **zero group overlap** between train and test partitions.

2. **Unified preprocessing pipeline**
   - Introduced `PREPROCESSING_PIPELINE` using scikit-learn `Pipeline`.
   - Moved preprocessing steps into pipeline transformers so they are fit only on training folds:
     - Zero-variance feature removal
     - High-correlation feature filtering
     - Optional drop of `Port_Category` columns
     - Mutual-information top-k feature selection
     - Standard scaling

3. **Pipeline-based model training**
   - Reworked model training to build a full pipeline per model (`preprocess` + optional sampler + estimator).
   - Added imbalance handling directly in pipeline flow:
     - `class_weight` strategy applies `class_weight='balanced'` where supported
     - `smote` and `undersample` strategies run inside cross-validation folds through `imblearn` pipeline

4. **Group-aware evaluation utilities**
   - Updated `evaluate_model` to run `cross_validate` and `cross_val_predict` using `GroupKFold` and groups.
   - Reports out-of-fold classification metrics and confusion matrix.
   - Keeps a fitted final pipeline per model for later inference.

5. **Pipeline-compatible tuning helper**
   - Updated `tune_model` so parameter grids target estimator parameters via `model__...` keys.
   - Ensures hyperparameter search is also group-aware by passing `groups` into `GridSearchCV.fit`.

## Why these changes reduce leakage risk

- All train/test boundaries are now based on groups, so flows from the same session/burst are not split across folds.
- Feature selection and scaling are done inside fold-local pipelines, preventing information from test folds from leaking into preprocessing.
- Any resampling strategy runs only on each training fold, not on the full dataset.

## How to use the updated workflow

- Select imbalance strategy in `IMBALANCE_STRATEGY`:
  - `'none'`, `'class_weight'`, `'smote'`, or `'undersample'`
- Run the model comparison cell; it now evaluates all models with GroupKFold automatically.
- Use `tune_model` with estimator-only parameter names; the function maps them to pipeline keys.
