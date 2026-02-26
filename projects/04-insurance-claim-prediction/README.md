# Project 4: Insurance Claim Frequency Prediction

## Summary
Predict claim frequency on a real-world motor third party liability dataset and compare classical actuarial baselines with ML approaches under an **exposure-weighted Poisson deviance** objective.

## Key concepts (GLM and Poisson deviance)

### Poisson GLM (what “GLM” means here)
A **Generalized Linear Model (GLM)** extends linear regression to non-Gaussian targets by combining:
1) a distribution for the target (here **Poisson** for counts),
2) a **linear predictor**: $\eta_i = \langle \theta, x_i \rangle + \theta_0$,
3) a **link function** that maps $\eta_i$ to the mean.

For claim frequency modeling, we assume the number of claims for policy $i$ satisfies:
$$
\mathrm{ClaimNb}_i \sim \mathrm{Poisson}(\lambda_i \cdot \mathrm{Exposure}_i),
\quad\text{equivalently}\quad
y_i \cdot \mathrm{Exposure}_i \sim \mathrm{Poisson}(\lambda_i \cdot \mathrm{Exposure}_i),
$$
where $y_i = \mathrm{ClaimNb}_i / \mathrm{Exposure}_i$ is the **claim frequency**.

The Poisson GLM uses a **log-link**, meaning the intensity (expected frequency) is:
$$
\lambda_i = \exp(\langle \theta, x_i \rangle + \theta_0).
$$
This guarantees $\lambda_i > 0$ and makes the model linear in the log-scale.

### Exposure-weighted Poisson deviance (what it measures)
The main objective is the **exposure-weighted Poisson deviance**, which is a likelihood-based loss for Poisson models (lower is better). It penalizes errors in predicted frequency while accounting for different policy durations via exposure weights.

The loss used in this project is:
$$
L(\mathcal{D}, \hat{\theta})
=
\frac{1}{\sum_{i=1}^m \mathrm{Exposure}_i}
\sum_{i=1}^m \mathrm{Exposure}_i \, \ell(\hat{\lambda}_i, y_i),
$$
with
$$
\hat{\lambda}_i = \exp(\langle \hat{\theta}, x_i \rangle + \hat{\theta}_0),
\qquad
\ell(\hat{\lambda}, y) = 2\big(\hat{\lambda} - y - y\log \hat{\lambda} + y\log y\big),
$$
and the convention $x\log x = 0$ when $x=0$.

Intuition:
- The deviance compares the fitted model to a “perfect” model that would predict $y$ exactly.
- Weighting by **Exposure** gives more influence to policies observed for longer (more reliable frequency estimates).

## What this notebook covers
- Train-test split (90% train, 10% test)
- Preprocessing
  - Standardize numeric features and one-hot encode categorical variables
  - Use exposure as sample weights
- Poisson GLM baseline
  - Fit an unregularized Poisson regressor
  - Report MAE, MSE, and exposure-weighted Poisson deviance
- GLM feature engineering
  - Marginal log-frequency plots
  - Transformations to improve approximate linearity under the log-link
  - Refit GLM and compare performance
- Poisson neural network
  - Exponential output to model intensities
  - Exposure-weighted deviance training
  - Cross-validation of regularization
- Tree-based methods (Poisson deviance)
  - Regression tree
  - Random forest
  - Gradient boosting
  - Cross-validation for hyperparameters

## Conclusion

| Model | **L_train** | **Δ_train (%)** | **L_test** | **Δ_test (%)** |
|---|---:|---:|---:|---:|
| Poisson GLM (baseline) | 0.459110 | +0.00 | 0.450700 | +0.00 |
| Poisson GLM + feat. eng. | 0.458230 | +0.19 | 0.449240 | +0.32 |
| Neural net #1 | 0.447387 | +2.55 | 0.462990 | -2.73 |
| Neural net #2 (best) | 0.454183 | +1.07 | 0.444963 | +1.27 |
| Decision tree | 0.453908 | +1.13 | 0.442557 | +1.81 |
| Random forest | 0.419068 | +8.72 | 0.440117 | +2.35 |
| XGBoost (Poisson) | 0.445334 | +3.00 | 0.436277 | +3.20 |

Overall, all models improve on the baseline Poisson GLM in test Poisson deviance (except NN#1), with **XGBoost** achieving the best test loss (≈ **3.2%** improvement), followed by **Random Forest** (≈ **2.35%**), while feature engineering gives only a modest gain (≈ **0.32%**).