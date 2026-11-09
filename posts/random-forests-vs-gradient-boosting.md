---
title: Random forests vs gradient boosting, in practice
date: 2026-01-19
summary: How bagging and boosting differ, which hyperparameters matter, a runnable scikit-learn comparison, and why impurity-based feature importance misleads
tags: Machine learning, Python
---

Random forests and gradient boosting are both ensembles of decision trees, and on tabular data they are usually the first two models I reach for. They get to good predictions from opposite directions, though, and that difference decides how you tune them, how they fail and when each is the better default.

This post covers how each one works, the hyperparameters worth touching, a comparison you can run in under a minute, and a feature importance trap that catches a lot of people.

## Bagging and boosting attack different errors

A single deep decision tree has low bias and high variance. It fits the training data closely, and a slightly different sample gives you a noticeably different tree.

A **random forest** keeps the deep trees and reduces the variance. Each tree trains on a bootstrap sample of the rows, and each split only considers a random subset of the features. The trees end up partly decorrelated, and averaging many of them cancels out much of their individual noise. This is bagging (bootstrap aggregating) plus feature subsampling.

**Gradient boosting** goes the other way. It starts with weak, shallow trees that have high bias and builds them in sequence. Each new tree fits the gradient of the loss with respect to the current predictions (for squared error, that is simply the residuals), and its output is scaled down by a learning rate before it is added. Bias falls with every round. Run it too long and variance climbs instead, which is why boosting needs a stopping rule and a forest does not.

That gives the practical summary:

- Adding more trees to a random forest rarely hurts. It just costs time.
- Adding more rounds to gradient boosting eventually overfits, so the number of rounds is a hyperparameter you tune, ideally with early stopping.

## The hyperparameters that matter

For `RandomForestClassifier`:

- `n_estimators`: more is better until the gains flatten. A few hundred is a common starting point.
- `max_features`: the number of features each split considers. This is the main decorrelation control. The classifier defaults to `"sqrt"`.
- `max_depth`, `min_samples_leaf`: limit tree size if the forest overfits noisy data or the model is too large to serve.
- `n_jobs=-1`: the trees are independent, so they train in parallel.

For gradient boosting, the learning rate and the number of rounds trade off against each other. A smaller learning rate needs more rounds but usually generalises better. In scikit-learn's `HistGradientBoostingClassifier`, the main settings are:

- `learning_rate` (default 0.1) and `max_iter` (default 100), the number of boosting rounds.
- `max_leaf_nodes` (default 31), `max_depth`, `min_samples_leaf` (default 20): tree complexity.
- `l2_regularization`: shrinks leaf values.
- `early_stopping`: the default is `"auto"`, which turns early stopping on only when the training set has more than 10,000 samples or when you pass `X_val` and `y_val` to `fit`. On a small dataset you have to set it to `True` yourself. It then holds out `validation_fraction` of the training data and stops when the score has not improved for `n_iter_no_change` rounds.
- `categorical_features`: since 1.6 it defaults to `"from_dtype"`, so pandas or polars categorical columns are handled natively.

In current XGBoost, early stopping is a constructor argument. You set `early_stopping_rounds` on `XGBClassifier` and pass an `eval_set` to `fit`. Older tutorials pass `early_stopping_rounds` to `fit()`, but in recent versions `fit()` has no such parameter. After training, `best_iteration` tells you where the validation metric peaked.

## A runnable comparison

The script below uses scikit-learn's bundled breast cancer dataset (569 rows, 30 numeric features), so it needs no download. I add one column of pure random noise to use later in the feature importance section.

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
rng = np.random.RandomState(42)
X = X.assign(random_noise=rng.normal(size=len(X)))  # a column with no signal

models = {
    "random_forest": RandomForestClassifier(
        n_estimators=500, max_features="sqrt", n_jobs=-1, random_state=42
    ),
    "hist_gradient_boosting": HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=500, max_leaf_nodes=15,
        early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=20, random_state=42,
    ),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, model in models.items():
    scores = cross_validate(model, X, y, cv=cv,
                            scoring=["accuracy", "roc_auc"], return_estimator=True)
    acc, auc = scores["test_accuracy"], scores["test_roc_auc"]
    print(f"{name:24s} acc {acc.mean():.3f} ± {acc.std():.3f}   "
          f"roc_auc {auc.mean():.3f} ± {auc.std():.3f}")
    if name == "hist_gradient_boosting":
        print("  rounds per fold:", [est.n_iter_ for est in scores["estimator"]])

# XGBoost with early stopping on a separate validation split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
X_fit, X_val, y_fit, y_val = train_test_split(X_tr, y_tr, test_size=0.2,
                                              stratify=y_tr, random_state=42)
xgb = XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=4,
                    subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                    early_stopping_rounds=50, tree_method="hist", random_state=42)
xgb.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
print("xgboost best_iteration", xgb.best_iteration,
      "test accuracy", round(xgb.score(X_te, y_te), 3))
```

Here is what one run printed (Python 3, recent scikit-learn and XGBoost, a 2-core machine):

```text
random_forest            acc 0.956 ± 0.010   roc_auc 0.992 ± 0.005
hist_gradient_boosting   acc 0.965 ± 0.020   roc_auc 0.991 ± 0.007
  rounds per fold: [134, 109, 229, 215, 98]
xgboost best_iteration 501, test accuracy 0.972
```

How I read it:

- **The two models are effectively tied.** The gap in mean accuracy is smaller than the fold-to-fold standard deviation, and ROC AUC is the same to two decimal places. On a dataset this small and clean, I would not claim either one is better.
- **Early stopping chose very different round counts per fold**, from 98 to 229. That spread is the reason to let a validation set pick the number of rounds rather than hard-coding it.
- **The XGBoost number is a single train/test split**, so it is not directly comparable with the cross-validated scores above. It shows the API, not a winner.

For a fair comparison of your own, keep the folds fixed (`random_state` on the splitter), report the spread as well as the mean, and do any early stopping inside each training fold so the test fold is never used to choose the number of rounds. `HistGradientBoostingClassifier` does this for you when it carves `validation_fraction` out of each training fold.

## Feature importance: impurity vs permutation

Tree ensembles expose `feature_importances_`, which is the mean decrease in impurity (MDI). The scikit-learn documentation warns about two problems with it: it can inflate the importance of high-cardinality numerical features, and because it is computed from training-set statistics, "the importances can be high even for features that are not predictive of the target variable, as long as the model has the capacity to use them to overfit."

The alternative is permutation importance. Shuffle one column in held-out data and measure how much the score drops.

```python
rf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42).fit(X_tr, y_tr)
perm = permutation_importance(rf, X_te, y_te, n_repeats=20, random_state=42, n_jobs=-1)
```

My run surfaced a different trap from the one I set up. This is condensed from the script's output:

```text
MDI: random_noise rank 30 of 31
MDI top 5: worst concave points 0.122, worst perimeter 0.120, worst radius 0.118,
           worst area 0.117, mean concave points 0.100
Permutation top 5: mean concave points 0.0024, worst texture 0.0021, ...
Features with permutation importance <= 0: 27 of 31
```

Here the noise column did not get inflated by MDI. It ranked near the bottom. The bigger surprise was permutation importance: 27 of the 31 features scored zero or below. This dataset is full of strongly correlated features (radius, perimeter and area measure almost the same thing). When you shuffle one of them, the forest falls back on its correlated twins and accuracy barely moves. Permutation importance answers "how much does the model depend on this exact column?", and with redundant features the answer is "very little", even for features that clearly carry signal.

The practical rules I take from this:

- Do not report MDI importances as if they were facts about the data. They describe how the trees split on the training set.
- Compute permutation importance on held-out data, and repeat the shuffle (`n_repeats`) so you can see the spread.
- When features are correlated, group or cluster them first and interpret importance per group. The scikit-learn example on multicollinear features shows one way to do this.

## When to pick which

I pick a **random forest** when I want a strong baseline with almost no tuning, when the data is small or noisy and I want something hard to overfit, or when I need out-of-bag estimates and easy parallel training.

I pick **gradient boosting** (`HistGradientBoostingClassifier`, XGBoost or LightGBM) when I have enough data and time for a validation set to tune the learning rate and the number of rounds. It is also my choice when I want native handling of missing values and categorical features, when model size and inference speed matter (shallow boosted trees are often smaller than a forest of deep trees), or when I need monotonic constraints.

On many real tabular problems a tuned boosting model comes out slightly ahead. As the run above shows, though, "slightly" can fall inside the noise. Cross-validate both, look at the spread, and let the data decide.

## Sources

- [scikit-learn: HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)
- [scikit-learn: Ensembles user guide](https://scikit-learn.org/stable/modules/ensemble.html)
- [scikit-learn: Permutation importance vs random forest feature importance (MDI)](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html)
- [scikit-learn: Permutation importance with multicollinear or correlated features](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html)
- [XGBoost: Python API reference (scikit-learn interface)](https://xgboost.readthedocs.io/en/stable/python/python_api.html)
