# the confusion matrix

Everything comes from:

- TP (true positives)
- FP (false positives)
- TN (true negatives)
- FN (false negatives)

If you don’t have an intuition for how these shift when you move your threshold, you don’t understand any of the metrics below. Period.

## 2. Core “single-number” metrics (threshold-dependent)

### Accuracy

`(TP + TN) / total`

- What it measures: overall correctness
- When it works: balanced classes
- When it fails: imbalanced data (most real problems)

👉 Brutal truth: accuracy is often a vanity metric.

![Confusion Matrix Diagram](https://miro.medium.com/v2/format:webp/0*-oGC3SE8sPCPdmxs.jpg)

*Source: [Medium - Confusion Matrix for Machine Learning](https://miro.medium.com)*

> This diagram visually summarizes the layout of the confusion matrix, with predicted vs. actual classes and the positions of TP, FP, TN, and FN. Having a concrete visual often helps make sense of where each metric "looks" in the matrix.


### Precision (Positive Predictive Value)

`TP / (TP + FP)`

- “When I predict positive, how often am I right?”
- Sensitive to false positives

Use when:

- False alarms are expensive (fraud flags, spam filters)

### Recall (Sensitivity, True Positive Rate)

`TP / (TP + FN)`

- “Of all real positives, how many did I catch?”
- Sensitive to false negatives

Use when:

- Missing positives is costly (disease detection, fraud)

### Specificity (True Negative Rate, TNR)

`TN / (TN + FP)`

“Of all real negatives, how many did I correctly reject?”

This is the mirror of recall, but for the negative class.

### False Positive Rate (FPR)

`FP / (FP + TN)` — same denominator as **specificity** (all actual negatives).

**Complement:** `FPR = 1 − specificity` (and `specificity = 1 − FPR`). Don’t confuse this with sensitivity: FPR is **not** `1 − recall`.

“How often do I falsely trigger among negatives?”

Important because:

👉 ROC curves are built from this (x-axis is FPR).

### False Negative Rate (FNR)

`FN / (FN + TP)` — same denominator as **recall / sensitivity / TPR** (all actual positives).

**Complement:** `FNR = 1 − recall` (i.e. `1 − sensitivity` / `1 − TPR`). Don’t confuse this with specificity: FNR is **not** `1 − specificity`.

“How often do I miss real positives?”

## 3. Combined metrics (where people start getting sloppy)

### F1 Score

`2 × (Precision × Recall) / (Precision + Recall)`

- Balances precision and recall equally
- Ignores TN completely

👉 Hidden danger:

If TN matters in your problem, F1 is blind to it.

### Fβ Score (like F2, F0.5)

General form:

- F1 → equal weight
- F2 → recall-heavy
- F0.5 → precision-heavy

Use when:

- You explicitly care more about one type of error

### Balanced Accuracy

`(Recall + Specificity) / 2`

Fixes accuracy for imbalanced datasets

### G-Mean

`√(Recall × Specificity)`

Penalizes models that do well on one class but fail on the other

## 4. Threshold-free metrics (this is where real evaluation happens)

These evaluate your model across all possible thresholds.

### ROC Curve and ROC-AUC

ROC = plot of:

- x-axis: FPR
- y-axis: TPR (recall)

ROC-AUC = probability model ranks a random positive above a random negative

When it works:

- Balanced or moderately imbalanced data

When it lies to you:

- Highly imbalanced datasets

👉 Why?

Because FPR barely moves when negatives dominate.

### Precision-Recall Curve and PR-AUC (AUPRC)

PR curve = Precision vs Recall

When it matters:

- Imbalanced data (THIS IS YOUR CASE most of the time)

👉 This is often more honest than ROC-AUC.

## 5. Calibration metrics (almost everyone ignores these)

### Log Loss (Cross-Entropy Loss)

Penalizes wrong confidence, not just wrong predictions

👉 A model that says “99% sure” and is wrong gets crushed

### Brier Score

Mean squared error of predicted probabilities

👉 Measures how well probabilities match reality

### Calibration curves

Do predicted probabilities reflect actual outcomes?

Example:

If model says “70% chance” → does it happen ~70% of the time?

## 6. Ranking & top-k metrics (critical for recommendation systems)

Since you’ve worked on rec systems, this is where you should be sharper than most:

- Precision@K
- Recall@K
- MAP (Mean Average Precision)
- NDCG (Normalized Discounted Cumulative Gain)

👉 These care about ordering, not just classification

## 7. Multi-class extensions (don’t screw this up)

For multi-class:

- Macro average → treat all classes equally
- Micro average → weight by frequency
- Weighted average → compromise

👉 Common mistake:

Using micro when you think you’re being fair, but you’re actually biasing toward dominant classes.

## 8. What you’re probably missing (and shouldn’t)

### Matthews Correlation Coefficient (MCC)

- Uses all 4 confusion matrix values
- Robust to imbalance

👉 This is criminally underused and often better than F1

### Cohen’s Kappa

Adjusts for chance agreement

### Lift / Gain

Used in marketing, ranking, risk models

## 9. The real problem: you’re probably thinking about this wrong

Here’s the part most people avoid:

You don’t pick a metric because it’s “standard.”

You pick it because it aligns with real-world cost.

Ask yourself:

- What’s worse: false positives or false negatives?
- By how much?
- Does ranking matter more than classification?
- Do I care about probabilities or just decisions?

If you can’t answer those, you’re not evaluating—you’re decorating.
