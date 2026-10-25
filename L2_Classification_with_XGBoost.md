# Fundamentals of Machine Learning – L2

## Classification with XGBoost

**Received in:** L2
**Due in:** L3

---

### Installation

Install the `xgboost` and `optuna` libraries. To make sure everything works in the notebook:

```bash
pip install xgboost plotly optuna anywidget
```

### Data and Task

Use the cleaned **Ames Housing Dataset**, with appropriate encoding for categorical features (Assignment 1). Implement **xgboost as a multiclass classifier** to predict the prices of the houses. Use the following six classes:

| Class | Price range |
|---|---|
| 1 | < 100,000 |
| 2 | 100,000 – 130,000 |
| 3 | 130,000 – 180,000 |
| 4 | 180,000 – 250,000 |
| 5 | 250,000 – 400,000 |
| 6 | ≥ 400,000 |

### Experimental Setup

For all experiments:

- Keep **stratification** in mind and try it both with and without.
- Use a **train-test split** of 10% and 20% and compare the results.
- Use **k-fold cross-validation** with k = 5 and k = 10.
- Interpret the results. Also take the **bias-variance tradeoff** into account.

### Visualization and Metrics

- Plot the **first** and the **best tree**, as well as the **feature importance** (*gain* and *split*). More details in the documentation.
- Report the metrics **precision, accuracy, recall, and F1**.
- Also plot the **confusion matrix**, the **ROC curve**, and the **precision-recall curve**.
- Find the precision-recall thresholds that correspond to the best **macro-F1**.
- Interpret the results.

### Data Preparation

Compare the results for different data preparation strategies. If needed, experiment further with the features. If necessary, use one or more of the methods for **imbalanced classes** discussed in the lecture.

### Hyperparameters

1. Experiment with the model's hyperparameters. **Manually** change some hyperparameter values and examine how the results change. Explain the results.
2. Then use **Optuna** for Bayesian optimization. Use **`TPESampler`** – it corresponds most closely to the optimization discussed in the lecture (documentation and example).
3. Also create **visualizations of the optimization process**. Use this to save intermediate steps. See an alternative complete example here.

### Documentation

Describe all results in the notebook. Try to find explanations. Markdown (examples) makes it easy to include formatted text in Jupyter notebooks, including equations, headings, tables, bullet points, etc.

---

### Bonus

- **× 1.5, × 1.0, × 0.5** for the three teams with the best **macro-F1**.
- **× 1.5:** For identically constructed features, implement and optimize another boosted trees algorithm: **LightGBM**, **AdaBoost**, or **CatBoost**. Explain, in comparison, how the trees are built (with plots), the learning process, and the results.
