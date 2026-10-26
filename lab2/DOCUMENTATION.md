# Lab 2 — Documentation

What we did for Lab 2, the decisions we made and why. Update this file whenever something changes.

- Requirements: [L2_Classification_with_XGBoost.md](L2_Classification_with_XGBoost.md)
- File overview and run order: [README.md](README.md)

---

## 1. Task

Predict the price class of a house (Ames Housing dataset, 2930 houses) with **XGBoost as a multiclass classifier**:

| Class | Price range | Houses | Share |
|---|---|---|---|
| 1 | < 100,000 | 237 | 8.1% |
| 2 | 100,000 – 130,000 | 509 | 17.4% |
| 3 | 130,000 – 180,000 | 1037 | 35.4% |
| 4 | 180,000 – 250,000 | 686 | 23.4% |
| 5 | 250,000 – 400,000 | 397 | 13.5% |
| 6 | ≥ 400,000 | 64 | 2.2% |

Each class includes its lower bound (a house at exactly 130,000 is class 3). The classes are **imbalanced**: class 6 has only
64 houses, so it is expected to be the hardest class and stratification matters most for it.

---

## 2. Overall workflow

```
01_prep_A / B / C   →   02_compare_preprocessing   →   03_xgboost_tuning   →   04_bonus_lightgbm
 3 datasets              pick the winning approach       tune the winner          same features,
                         (fixed hyperparameters)         + final evaluation       another algorithm
```

**Two stages, not "tune everything":**

1. **Compare the preprocessing approaches with the same XGBoost hyperparameters.** Since the model is identical, any difference
   in the results comes from the data preparation alone.
2. **Tune only the winning approach** (manual experiments, then Optuna).

Tuning every approach separately would cost far more and would mix two effects: we could no longer tell whether an improvement
came from the preprocessing or from the tuning. If two approaches end up very close in stage 1, a short Optuna run on both is an option.

**Choosing the winner:** by the **mean macro-F1 across the CV folds**, not by a single test split (one split can favour an approach
by chance). The spread across folds is also used for the bias-variance discussion.

**Test set:** used **once**, at the very end, for the final tuned model. Optuna, the manual experiments and the macro-F1 threshold
search are all evaluated with CV on the training data — otherwise the final scores would be too optimistic.

---

## 3. Repository structure

```
LabML/
├── data/AmesHousing.csv          raw data, shared by all labs (not in git)
├── archive/                      old notebooks (Lab 1, first Lab 2 attempt), kept unchanged except for the CSV path
└── lab2/
    ├── common.py                 shared code: paths, price classes, loading/saving
    ├── processed_data/           one CSV per preprocessing approach (not in git, generated)
    ├── results/                  cached experiment results + best_setup.json (CSVs not in git, generated)
    ├── 01_prep_A_baseline.ipynb
    ├── 01_prep_B_native_categorical.ipynb
    ├── 01_prep_C_feature_engineering.ipynb
    ├── 02_compare_preprocessing.ipynb
    ├── 03_xgboost_tuning.ipynb
    └── 04_bonus_lightgbm.ipynb
```

**Why one notebook per approach:** each approach can be presented on its own with the same structure (*Idea → Steps → Output saved*),
and adding a new approach means adding one notebook and one name in `02`.

**Why the approaches can be saved as CSV files:** they contain only steps that learn nothing from the data (dropping columns,
filling structural NaNs, fixed mappings). XGBoost handles missing values itself, so no median imputation is needed, and no
scaling is needed for trees. Steps that *do* learn from the data (outlier removal, SMOTE) happen inside each training split in `02`.
One-hot encoding on the full dataset only reads the list of categories, which has a negligible effect.

### `common.py`

| Name | Purpose |
|---|---|
| `RAW_DATA_PATH`, `PROCESSED_DIR` | paths, resolved from the file's location, so they work from any notebook |
| `PRICE_BINS`, `CLASS_LABELS`, `TARGET` | the 6 classes from the requirements; target column `PriceClass` |
| `load_raw()` | reads the raw file — it is **tab-separated** (`sep="\t"`) |
| `add_price_class(df)` | adds `PriceClass` (1–6) and **drops `SalePrice`** (keeping it would give the model the answer) |
| `save_processed(df, name)` | writes `processed_data/<name>.csv` |
| `load_processed(name)` | reads a processed CSV and turns text columns back into `category` (a CSV cannot store that type) |
| `split_features_target(df)` | returns X and y, with y shifted to 0–5 (XGBoost expects labels starting at 0) |
| `drop_outliers(X, y)` | removes houses with `Gr Liv Area > 4000` — training data only |
| `add_smote_samples(X, y)` | SMOTE / SMOTENC oversampling; original rows keep their NaNs |
| `fit_model(model, X, y, remove_outliers, balance)` | fits a model, applying outlier removal and class balancing to the training data only |
| `compute_scores(y_true, y_pred)` | accuracy and macro precision, recall, F1 |
| `cross_validate(make_model, X, y, cv, ...)` | k-fold CV with a fresh model per fold; train and validation scores per fold |

---

## 4. Preprocessing approaches

All approaches keep **all 2930 houses** and the same target. Their outputs are compared in `02`.

| | A – baseline | B – native categorical | C – feature engineering |
|---|---|---|---|
| Features | 201 | 79 (35 numeric, 44 categorical) | 208 |
| Columns dropped | 12 (IDs, mostly empty, near-constant, redundant) | only the 2 IDs | same as A |
| Missing values | structural filled, 497 truly missing left as NaN | all kept as NaN | same as A |
| Text columns | ordinal → integers, nominal → one-hot | pandas `category`, `enable_categorical=True` | same as A |
| Extra | — | — | 7 engineered features |
| Question it answers | — | Does manual cleaning/encoding help? | Do new features help? |

### 4.1 Approach A — Baseline (`01_prep_A_baseline.ipynb`)

**Columns dropped**

| Reason | Columns |
|---|---|
| Identifiers | `Order`, `PID` |
| > 80% missing | `Pool QC`, `Misc Feature`, `Alley`, `Fence` |
| Redundant | `Garage Yr Blt` (correlation 0.835 with `Year Built`, undefined without a garage) |
| Near-constant (one value ≥ 99%) | `Street`, `Utilities`, `Pool Area` |

`Condition 2` is just below the 99% threshold (with all 2930 houses), so it is kept.

**Missing values**

- **Structural** NaNs (the data documentation says NaN = the house does not have the feature): no basement, no garage,
  no fireplace, no masonry veneer → `"None"` for text columns, `0` for the related areas and counts.
  - Only filled when the house really lacks the feature. A house that has a basement but one blank descriptor keeps NaN.
  - One house is listed as `Detchd` garage but has no area, cars or finish → treated as having no garage.
- **Truly missing** values (497, of which 490 in `Lot Frontage`) stay **NaN**. XGBoost learns, at every split, which branch
  missing values go to. Reasons: no leakage (no median computed over the test rows), and the comparison between approaches
  stays about encoding and features, not about imputation.

**Encoding**

- **Ordinal** (natural order) → integers, `"None"` = 0: quality/condition ratings (`Po < Fa < TA < Gd < Ex`), `Bsmt Exposure`,
  `BsmtFin Type 1/2`, `Garage Finish`, `Lot Shape`, `Land Slope`, `Paved Drive`, `Functional`, `Central Air`.
  A check stops the notebook if a value is not in its mapping.
- **Nominal** (20 columns, no order) → one-hot. Categories with fewer than 10 houses are grouped as `Other`.
  `MS SubClass` is a category code stored as a number, so it is treated as text.

### 4.2 Approach B — Native categorical (`01_prep_B_native_categorical.ipynb`)

Minimal cleaning, XGBoost does the rest:

- Only `Order` and `PID` are dropped; mostly-empty and near-constant columns are kept.
- Nothing is filled: structural NaNs ("no garage") stay NaN too.
- The 44 text columns stay as categories. With `enable_categorical=True`, one split can send any group of categories left and
  the rest right, instead of testing one dummy column at a time.
- `MS SubClass` gets an `SC` prefix (`20` → `SC20`) so it is read back from the CSV as text, not as a number.
- Trade-off: the order of the quality ratings is not given to the model.

### 4.3 Approach C — Feature engineering (`01_prep_C_feature_engineering.ipynb`)

Loads A's CSV and adds 7 features, so everything else is identical to A (**run A before C**). Original columns are kept.

| Feature | Formula | Spearman with `PriceClass` | Compared with its parts |
|---|---|---|---|
| `Quality x SF` | overall quality × total area | **0.866** | stronger than `Overall Qual` (0.780) |
| `Total SF` | basement + 1st + 2nd floor area | **0.793** | stronger than `Gr Liv Area` (0.705) |
| `Total Bathrooms` | full + ½ half baths, incl. basement | **0.684** | stronger than `Full Bath` (0.605) |
| `House Age` | year sold − year built | 0.653 | ≈ `Year Built` (0.654) |
| `Years Since Remod` | year sold − remodel year | 0.589 | ≈ `Year Remod/Add` (0.584) |
| `Total Porch SF` | 5 porch / deck areas summed | 0.416 | **weaker** than `Open Porch SF` (0.470) |
| `Was Remodeled` | remodel year ≠ build year | 0.080 | weak on its own |

- All houses were sold in 2006–2010, so the age features are almost the construction year reversed; they mainly make trees easier to read.
- Summing porch types with different value makes the feature weaker — not every engineered feature helps.
- 3 houses were sold before construction/remodelling was finished (negative age) → ages clipped at 0.
- Correlation looks at one feature at a time; the real test is the model comparison in `02`.

---

## 5. Comparison of the approaches (`02_compare_preprocessing.ipynb`)

### Setup

- **Fixed model:** XGBoost with its **default hyperparameters** (100 trees, depth 6, learning rate 0.3) for every approach, so no
  approach is favoured by settings chosen for it. `enable_categorical=True` (needed for B, no effect on A and C).
  XGBoost needs labels 0–5, so `PriceClass − 1` is used (`split_features_target()`).
- **Evaluation grid (48 experiments):** approach (A/B/C) × test size (10% / 20%) × stratification (yes / no, for the split *and*
  the CV folds) × k-fold CV (5 / 10) × outlier removal (yes / no). For each: split → k-fold CV on the training part (train and
  validation scores per fold) → fit on the whole training part → score on the test part.
- **Outliers:** houses with `Gr Liv Area > 4000` (5 houses, recommended by the dataset's author) are removed from the **training
  data only**. Validation folds and the test part keep all houses, as real data would.
- **Metrics:** accuracy and macro-averaged precision, recall and F1 (`compute_scores()`). **Macro-F1** is the main metric because it
  gives every class the same weight; with accuracy, the large class 3 would dominate.
- The grid takes ≈ 7 minutes; results are cached in `results/compare_preprocessing.csv` (+ `_folds.csv`). Set `RERUN = True` to recompute.

### Results

| Approach | CV macro-F1 | Fold std | Train macro-F1 | Test macro-F1 | Settings won (of 16) |
|---|---|---|---|---|---|
| A – baseline | 0.734 | 0.036 | 0.998 | 0.741 | 5 |
| B – native categorical | 0.728 | 0.033 | 0.998 | 0.730 | 1 |
| **C – feature engineering** | **0.740** | 0.035 | 0.998 | **0.754** | **10** |

- **C wins**, A beats B: the new features help, and the manual cleaning/encoding helps compared with the raw data.
- **The differences are small** (≈ 0.006) compared with the fold spread (≈ 0.035). The **consistent ranking** C > A > B across
  settings is what makes the result convincing.

**Evaluation settings**

| Setting | Finding |
|---|---|
| Stratification | clearly better: higher macro-F1 and less fold variation (std ≈ 0.030 vs. 0.040); without it, a fold can contain very few class-6 houses |
| k = 5 vs. 10 | same mean (0.734), but fold scores vary more with k = 10 (std 0.044 vs. 0.026) because each validation fold is half the size |
| Outlier removal | small improvement for all approaches (+0.001 to +0.003) |
| Test size 10% vs. 20% | test scores vary more with 10% (std 0.021 vs. 0.013): only ≈ 6 class-6 houses in a 10% test part |

CV and test scores agree within ≈ ±0.02, so CV is a reliable guide.

**Bias-variance:** training macro-F1 ≈ 1.0 vs. CV ≈ 0.73–0.74 (gap ≈ 0.26) for every approach. The default model almost memorises the
training data → **high variance**, low bias. It does not change the ranking, but it is the main target for tuning in `03`
(shallower trees, lower learning rate, `min_child_weight`, subsampling, regularisation).

### Imbalanced classes (approach C, 20% test, stratified, k = 10, outliers removed)

Implemented in `fit_model(..., balance=...)` and applied to the training folds only:
- **Class weights:** each house weighted inversely to its class size (`compute_sample_weight("balanced")`).
- **SMOTE:** synthetic houses for the smaller classes until all classes are equal. SMOTE cannot handle NaN, so it runs on an
  imputed copy and only the new synthetic rows are appended; the original rows keep their NaNs (`add_smote_samples()`).
  For categorical data (approach B) SMOTENC is used.

| Method | CV macro-F1 | Recall class 1 | Recall class 3 | Recall class 6 |
|---|---|---|---|---|
| none | 0.740 | 0.67 | 0.82 | 0.71 |
| **class weights** | **0.749** | 0.71 | 0.78 | **0.80** |
| SMOTE | 0.748 | 0.68 | 0.79 | 0.78 |

Both methods help the small classes at the cost of class 3. **Class weights** were chosen: practically equal to SMOTE but simpler,
no synthetic data, and faster.

### Chosen setup (saved to `results/best_setup.json`, read by `03`)

**Approach C, 20% test part, stratified, k = 10, outliers removed from training data, class weights** — baseline CV macro-F1 **0.749**.

## 6. Tuning and final evaluation (`03_xgboost_tuning.ipynb`)

Reads `results/best_setup.json` → same split and folds as in `02`. Default model: train macro-F1 0.998, **CV macro-F1 0.749 ± 0.032**.

### 6.1 Manual hyperparameter experiments

One hyperparameter changed at a time (others at default), 10-fold CV, training vs. CV macro-F1 for each value.
Results cached in `results/manual_tuning.csv` and `results/manual_tuning_lr_trees.csv`.

| Hyperparameter | Values tried | Best value | CV macro-F1 | vs. default | Observation |
|---|---|---|---|---|---|
| `max_depth` | 2–10 | 4 | 0.756 | +0.007 | depth 2 underfits (train 0.93) |
| `learning_rate` | 0.01–0.6 | 0.3 (default) | 0.749 | 0 | ≤ 0.03 underfits with only 100 trees |
| `n_estimators` | 50–800 | 200 | 0.757 | +0.008 | flat after 200 |
| `min_child_weight` | 1–20 | 20 | 0.750 | +0.001 | no clear trend |
| `subsample` | 0.5–1.0 | 0.7 | 0.753 | +0.004 | |
| `colsample_bytree` | 0.3–1.0 | 0.7 | 0.755 | +0.006 | |
| `reg_lambda` | 0–50 | 20 | **0.760** | **+0.011** | best single change |
| `gamma` | 0–5 | 0 (default) | 0.749 | 0 | higher → train *and* CV drop (bias) |

**Interpretation**
- The training macro-F1 stays ≈ 1.0 for almost all values; it only drops for settings that make the model clearly weaker
  (`gamma` ≥ 1, small `learning_rate`, `max_depth` = 2), and then the CV score drops too → reducing variance turns into **high bias**.
- The CV score stays at ≈ 0.74–0.76 whatever we change. Part of the train–CV gap cannot be removed by tuning: the classes cut a
  continuous price into bins, so houses just below and above a boundary (e.g. 178k vs. 182k) are nearly identical.
- The individual gains (≤ 0.011) are about the size of the uncertainty of a 10-fold mean (0.03 / √10 ≈ 0.01).
- **Learning rate × trees** (0.03 / 0.1 / 0.3 × 100 / 300 / 900): a small learning rate needs more trees (0.03 with 100 trees
  underfits, with 900 trees it reaches 0.751); all combinations end at CV ≈ 0.75. The learning rate mainly changes *how many trees
  are needed*, not how good the model can become.
- **Combining the individual best values** (`max_depth` 4, 200 trees, `subsample` 0.7, `colsample_bytree` 0.7, `reg_lambda` 20)
  gives only **0.751** — less than `reg_lambda` = 20 alone. Hyperparameters interact and some single gains were noise.
  One-at-a-time experiments explain *what each hyperparameter does*, but cannot find the best *combination* → Optuna.

### 6.2 Optuna (`TPESampler`)

- **TPE** (Tree-structured Parzen Estimator), a form of Bayesian optimisation: after 10 random start-up trials, the finished
  trials are split into "good" and "bad"; TPE models the hyperparameter distribution of each group (l(x), g(x)) and proposes the
  values where l(x) / g(x) is highest.
- **Objective:** the same 10-fold CV macro-F1 (outliers removed, class weights); the test part is not used.
  Each trial also stores its training macro-F1 and fold std (`trial.set_user_attr`).
- **Search space:** `max_depth` 2–10, `learning_rate` 0.01–0.3 (log), `n_estimators` 100–800 (step 50), `min_child_weight` 1–20 (log),
  `subsample` and `colsample_bytree` 0.5–1.0, `reg_lambda` and `reg_alpha` 0.001–50 (log), `gamma` 0–2.
- **Storage:** SQLite, `results/optuna_xgboost.db` (study `xgboost_approach_C`). Every trial is saved immediately → an interrupted
  run can be resumed; the notebook only runs the trials missing from `N_TRIALS`. The file is **not** excluded by `.gitignore`, so
  teammates get the finished study without re-running it.
- **100 trials, ≈ 44 minutes.** Best hyperparameters saved to `results/best_params.json`.

| Model | Train macro-F1 | CV macro-F1 | Fold std |
|---|---|---|---|
| Default | 0.998 | 0.749 | 0.032 |
| Manual combination | 0.995 | 0.751 | 0.040 |
| Best manual single change (`reg_lambda` = 20) | 0.997 | 0.760 | 0.036 |
| **Optuna best (trial 64)** | 0.997 | **0.773** | 0.034 |

**Best hyperparameters:** `max_depth` 7, `learning_rate` 0.082, `n_estimators` 400, `min_child_weight` 1.71, `subsample` 0.68,
`colsample_bytree` 0.58, `reg_lambda` 0.002, `reg_alpha` 0.078, `gamma` 0.21.

**Visualizations (plotly, interactive):** optimization history, hyperparameter importances (fANOVA), slice plot, parallel coordinates,
contour plot of the two most important hyperparameters, and a matplotlib scatter of train vs. CV macro-F1 for every trial.
Plotly output is shown in Jupyter / VS Code but not on GitHub.

**Interpretation**
- **+0.024 over the default**, +0.013 over the best manual change: searching jointly beats one-at-a-time.
- **TPE learns:** random start-up trials average 0.706, the last 50 trials 0.763. Best value 0.768 by trial 30, 0.773 at trial 64,
  unchanged until 100 → more trials would bring little.
- **Stable region:** the 10 best trials are within 0.005 and share `max_depth` 7–9, `learning_rate` ≈ 0.07–0.1, 350–450 trees,
  `min_child_weight` ≈ 2, subsampling ≈ 0.6–0.7.
- **Importances:** `learning_rate` (0.25), `colsample_bytree` (0.16), `n_estimators` (0.15), `subsample` (0.11), `max_depth` (0.10).
- **Interaction:** manual tuning favoured `reg_lambda` = 20, Optuna an almost zero L2 penalty — row/column subsampling and a small
  `gamma` already regularise, so strong L2 is no longer needed.
- **Bias-variance:** early trials fit the training data poorly and score low on CV (high bias); the best trials fit the training data
  almost perfectly *and* score highest on CV. The remaining gap is mostly the irreducible noise at the class boundaries.
- **Caveat:** the best of 100 trials on the same folds is slightly optimistic; the test part gives the unbiased estimate.

### 6.3 Final evaluation (test part, used once)

Final model = best Optuna hyperparameters, trained on the whole training part (outliers removed, class weights).
Metrics saved to `results/final_metrics.json`.

| Model | Accuracy | Macro-precision | Macro-recall | Macro-F1 |
|---|---|---|---|---|
| Default (test) | 0.778 | 0.765 | 0.769 | 0.766 |
| **Tuned (test)** | **0.775** | **0.758** | **0.769** | **0.763** |
| Tuned (CV estimate) | – | – | – | 0.773 |

Per class (tuned, test): F1 0.72 / 0.71 / 0.80 / 0.78 / 0.83 / 0.74 for classes 1–6; class 6 recall 0.77 (10 of 13 houses).

- **Test score ≈ CV estimate** → CV was a reliable guide.
- **Tuned ≈ default on the test part.** Paired bootstrap (2000 resamples): difference −0.004, 95% interval [−0.022, +0.014] → the
  586 test houses cannot separate models that differ by ≈ 0.02. The CV comparison (+0.024) is the more reliable evidence, minus some
  optimism from picking the best of 100 trials. We do **not** switch models based on the test score (that would make it a validation set).

**Confusion matrix:** all errors are between neighbouring classes; only 6 of 586 houses are off by more than one class →
the model learned the price order and struggles at the class boundaries.

**ROC / precision-recall (one-vs-rest):** macro ROC-AUC 0.962 (per class 0.93–0.995), macro average precision 0.831.
Class 6 has the highest ROC-AUC (0.995) but the lowest AP (0.72): with only 13 positives, a few false positives cost much precision,
and the many easy negatives that make the ROC curve look good do not count → PR curves are the better view for small classes.

**Thresholds for the best macro-F1** (chosen on out-of-fold training predictions, never on the test part):

| Decision rule | Out-of-fold macro-F1 | Test macro-F1 |
|---|---|---|
| **Argmax (kept)** | 0.772 | **0.763** |
| Best-F1 threshold per class (from the PR curves) | 0.768 | 0.760 |
| Thresholds tuned jointly for macro-F1 (coordinate ascent) | 0.777 | 0.762 |

Multiclass decision with thresholds: predict the class with the highest probability / threshold. The joint tuning looks better
only on the predictions it was tuned on; a nested CV gives −0.001 ± 0.012 → overfitting. The class weights already shift the
probabilities towards the small classes, so argmax is already (close to) optimal.

**Trees** (6 trees per round, one per class → 2400 trees; tree *i* = round *i* // 6 + 1, class *i* mod 6 + 1):
- **First tree** = tree 0 (round 1, class 1), 37 splits, starts with `Quality x SF` < 8830 and `Garage Qual`.
- **Best tree** = defined as the tree with the **largest total gain** → tree 5 (round 1, class 6), only 10 splits on size and quality
  (`Gr Liv Area`, `Total Bsmt SF`, `Exter Qual`). Its gain is amplified by the class weight of class 6 (7.8 vs. 0.47 for class 3).
- Average gain per tree: 561 in round 1, 8 in round 100, 1 in round 400 — the 10 best trees all come from rounds 1–3. Early trees learn
  the big structure, later trees make small corrections.
- Trees are rendered as SVG with graphviz (wide; scroll in VS Code).

**Feature importance:**
- `Quality x SF` is #1 by gain, split and total gain; `Total SF` #2 by gain → the engineered features of approach C matter most.
- **Split** favours continuous features with many split points (`Lot Area`, `Lot Frontage`, `Garage Area`); **gain** favours quality
  ratings (`Garage Qual`, `Overall Qual`).
- Average gain overrates rare features (`Electrical_FuseF`: 3 splits, `Exterior 2nd_Stucco`: 1 split, both in the gain top 10);
  **total gain** (gain × split) is the more robust summary.

### 6.4 Conclusion

| Step | Macro-F1 |
|---|---|
| Default model (CV) | 0.749 |
| Best manual single change (CV) | 0.760 |
| Optuna, 100 trials (CV) | 0.773 |
| **Final model (test)** | **0.763** (accuracy 0.775, ROC-AUC 0.962) |

The default model has low bias but high variance; tuning reduced the variance mainly with a lower learning rate, more trees and
row/column subsampling. The remaining gap is largely irreducible: the classes cut a continuous price into bins. A test score is
accurate to about ±0.02 with 586 houses.

## 7. Bonus: LightGBM vs. XGBoost (`04_bonus_lightgbm.ipynb`)

Same features (approach C), split, folds, outlier removal and class weights as XGBoost; tuned the same way (Optuna, TPE, 100 trials,
study `lightgbm_approach_C` in the same SQLite file). Results saved to `results/lightgbm_results.json`.

**Practical note:** LightGBM is ≈ 5× faster with 1 thread than with 8 on this small dataset (thread overhead), so each model uses
1 thread and Optuna runs 4 trials in parallel. LightGBM replaces spaces in feature names with `_`; importances are mapped back.

### How the trees are built

| | XGBoost | LightGBM |
|---|---|---|
| Growth | level-wise: every node of a level is split → balanced trees | leaf-wise (best-first): always the leaf with the largest loss reduction → deep, unbalanced trees |
| Size limit | `max_depth` | `num_leaves` (+ optional `max_depth`) |
| Split finding | histograms | histograms; Exclusive Feature Bundling for sparse features |
| Multiclass | one tree per class per round | the same |

**Demonstration with the same leaf budget (16 leaves, learning rate 0.1, 100 rounds):**

| | Mean depth (max) | Mean leaves | Train macro-F1 | CV macro-F1 |
|---|---|---|---|---|
| XGBoost `max_depth` = 4 | 4.0 (4) | 11.8 | 0.957 | 0.736 |
| LightGBM `num_leaves` = 16 | 8.5 (14) | 16.0 | 0.998 | 0.751 |

- XGBoost trees are always exactly depth 4 and often do not use all 16 leaves; LightGBM always uses 16 leaves, at depth 5–14.
- Drawn tree (round 1, class 3): LightGBM's first split sends 717 expensive houses straight to a leaf and spends the other leaves on a
  9-level branch of mid-sized, mid-quality houses — where class 3 must be separated from classes 2 and 4.
- Same budget → LightGBM fits faster (better CV here, but also more variance per leaf).

### Results

| | XGBoost | LightGBM |
|---|---|---|
| Default CV macro-F1 | 0.749 | 0.748 |
| Tuned CV macro-F1 | **0.773** | 0.772 |
| Test macro-F1 | **0.763** | 0.752 |
| Test accuracy | **0.775** | 0.770 |
| Test log-loss | 0.587 | **0.562** |
| Time per boosting round (1 thread) | 4.0 ms | **1.9 ms** |
| Tuned trees | mean depth ≈ 3.1, ≈ 8 leaves | mean depth ≈ 3.9, ≈ 10 leaves |

**Best LightGBM hyperparameters:** `max_depth` 4, `num_leaves` 39 (never reached: depth 4 allows at most 16), `learning_rate` 0.015,
`n_estimators` 750, `min_child_samples` 21, `subsample` 0.68 (`subsample_freq` 1), `colsample_bytree` 0.60, `reg_lambda` 0.038,
`reg_alpha` 0.125, `min_split_gain` 0.042.

**Interpretation**
- **Optuna limited leaf-wise growth:** with `max_depth` = 4 LightGBM's trees become shallow; deep leaf-wise trees overfit 2300 houses.
  Both tuned models converge to small trees (≈ 8–10 leaves) — tuning makes the two algorithms more alike.
- **Learning process:** XGBoost (lr 0.082) reaches its lowest validation log-loss after ≈ 100 rounds, then the loss rises slightly
  (over-confident probabilities) while macro-F1 stays flat. LightGBM (lr 0.015) learns slowly and smoothly, lowest validation loss at
  ≈ round 600. Both end at validation macro-F1 ≈ 0.77–0.78.
- **No significant difference in macro-F1:** test difference −0.011, bootstrap 95% interval ≈ [−0.037, +0.010]; CV almost identical.
  Per class within ≈ 0.01 except class 6 (0.74 vs. 0.69 — about one of 13 houses).
- **LightGBM has better-calibrated probabilities** (lower log-loss), matching the learning curves.
- **Speed:** ≈ 2× faster per round. EFB is not the reason (switching it off changes nothing here). The ≈ 6× faster Optuna search
  (7.6 vs. 44 min) is mainly due to parallel single-threaded trials — not a fair library comparison.
- **Feature importance** is very similar: `Quality x SF`, `Total SF`, `Gr Liv Area` are the top 3 by gain in both.
- **Conclusion:** different tree construction, same performance after tuning — the limit is the data (class boundaries in a
  continuous price), not the algorithm. XGBoost remains the final model (slight, non-significant edge in macro-F1).

---

## 8. Requirements checklist

| Requirement | Where | Status |
|---|---|---|
| Cleaned dataset with categorical encoding | `01_prep_*` | ✅ |
| 6 price classes, XGBoost multiclass | `common.py`, `02`, `03` | ✅ |
| With / without stratification | `02` | ✅ |
| Test size 10% and 20% | `02` | ✅ |
| k-fold CV, k = 5 and 10 | `02` | ✅ |
| Bias-variance interpretation | `02`, `03` | ✅ |
| First and best tree | `03` | ✅ |
| Feature importance (gain and split) | `03` | ✅ |
| Precision, accuracy, recall, F1 | `02`, `03` | ✅ |
| Confusion matrix, ROC, precision-recall curve | `03` | ✅ |
| Thresholds for the best macro-F1 | `03` | ✅ |
| Compare data preparation strategies | `01_prep_*`, `02` | ✅ |
| Experiment further with features | `01_prep_C` | ✅ |
| Methods for imbalanced classes | `02` | ✅ |
| Manual hyperparameter experiments | `03` | ✅ |
| Optuna `TPESampler` + optimization plots | `03` | ✅ |
| Results described in Markdown | every notebook | ongoing |
| Bonus: LightGBM / AdaBoost / CatBoost | `04` | ✅ LightGBM |

---

## 9. Environment setup

```bash
poetry install            # Python packages, incl. xgboost, optuna, plotly, anywidget, graphviz, imbalanced-learn, lightgbm
brew install graphviz     # system program needed to plot trees (not a Python package — every teammate must install it)
```

- Put `AmesHousing.csv` in `data/` (CSV files are excluded from git), then run the notebooks in the order from the README.
- In VS Code, select the project's `.venv` as interpreter and notebook kernel; otherwise imports show as unresolved.
- `.venv/bin/jupyter` still points to an old folder location; recreate the environment (`poetry env remove --all`, then
  `poetry install`) if you need the `jupyter` command in the terminal.

---

## 10. Decision log

| Decision | Reason |
|---|---|
| Compare preprocessing with fixed hyperparameters, then tune only the winner | isolates the effect of preprocessing; much cheaper |
| Pick the winner by mean CV macro-F1 | a single split can favour one approach by chance; macro-F1 weights the small classes equally |
| Test set used once, at the end | tuning on it would make final scores too optimistic |
| One notebook + one CSV per approach | readable, easy to present, easy to add an approach |
| Truly missing values left as NaN | XGBoost handles them natively; no leakage; keeps the comparison about encoding/features |
| Structural NaNs filled in A and C | they are not unknown — the documentation defines them as "feature not present" |
| Outlier removal as a comparison option, training data only | real test data contains unusual houses too |
| `SalePrice` dropped from the features | the target is built from it |
| B saved as CSV + `load_processed()` helper | all approaches stay readable CSV files; no extra dependency (Parquet would need `pyarrow`) |
| C built on top of A's CSV | no duplicated cleaning code; differences come only from the new features |
| Old notebooks moved to `archive/` | keeps the Lab 2 presentation focused |
| Fixed model in `02` = XGBoost defaults | neutral: no approach is favoured by hand-picked settings |
| Macro-F1 as main metric | every class counts equally; accuracy would be dominated by class 3 |
| Approach C chosen | highest CV macro-F1, best in 10/16 settings, highest test score |
| 20% test, stratified, k = 10 | less noisy test estimate, more stable folds, more training data per fold |
| Outliers removed from training data | small but consistent improvement in the grid |
| Class weights for the imbalance | +0.009 macro-F1, class-6 recall 0.71 → 0.80; equal to SMOTE but simpler and faster |
| Grid results cached in `results/` | the grid takes ≈ 7 min; the notebook can be shown without waiting |
| Setup passed to `03` via `best_setup.json` | `03` uses exactly the choices from `02`, nothing copied by hand |
| Manual tuning: one hyperparameter at a time, plus learning rate × trees | shows what each hyperparameter does (bias vs. variance); the interaction shows why single changes are not enough |
| Move on to Optuna after manual tuning | combining the individual best values gave only 0.751 — the best combination must be searched jointly |
| Optuna: 100 trials, 10-fold CV objective, SQLite storage | same evaluation as before; resumable; the study can be shared via git |
| Log scale for learning rate, regularisation, `min_child_weight` | the ratio matters more than the difference for these parameters |
| Tuned model kept although the default scored 0.003 higher on test | bootstrap interval includes 0; choosing on the test part would bias it |
| Argmax kept instead of tuned thresholds | thresholds gave no gain on unseen data (nested CV −0.001); class weights already balance the classes |
| "Best tree" = tree with the largest total gain | clear and works with the final model; no early-stopping split needed |
| Paired bootstrap on the test part | quantifies how precise a 586-house test score is |
| Bonus with LightGBM | leaf-wise growth is the clearest contrast to XGBoost's level-wise trees |
| LightGBM: 1 thread per model, 4 parallel Optuna trials | 5× faster per model on this small dataset; search in ≈ 8 min |
| XGBoost stays the final model | LightGBM is not significantly different; XGBoost has the slightly higher CV and test macro-F1 |
