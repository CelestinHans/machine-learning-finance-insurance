# Machine Learning in Finance & Insurance (ETH Zurich, Fall 2025)

This repository contains four course projects from **Machine Learning in Finance & Insurance (ETHZ, Fall 2025)**.  
Each project is implemented as a single Jupyter notebook, with a short project-level README describing the objective, methods, and outputs.

## Projects

### 1) Linear Regression and Regularization (Housing Prices)
- Goal: explain and predict house prices using OLS and regularized linear models.
- Topics: preprocessing, target transformation, OLS (sklearn + matrix algebra + statsmodels), standard errors, Ridge/Lasso/Elastic Net, truncated pseudoinverse, cross-validation, leakage discussion.
- Project README: [projects/01-linear-regression-regularization/README.md](projects/01-linear-regression-regularization/README.md)
- Notebook: [projects/01-linear-regression-regularization/mlfi_project_1.ipynb](projects/01-linear-regression-regularization/mlfi_project_1.ipynb)

### 2) Credit Analytics (Consumer Loans)
- Goal: estimate borrower risk profiles and compare probabilistic classifiers.
- Topics: logistic regression, kernel SVM with probability calibration (Platt scaling), ROC/AUC evaluation, and lending strategy simulation with P&L and VaR.
- Project README: [projects/02-credit-analytics/README.md](projects/02-credit-analytics/README.md)
- Notebook: [projects/02-credit-analytics/mlfi_project_2.ipynb](projects/02-credit-analytics/mlfi_project_2.ipynb)

### 3) Deep Hedging (Black–Scholes and Heston)
- Goal: implement deep hedging and evaluate hedging losses on simulated paths.
- Topics: Black–Scholes delta hedge benchmark, deep hedging networks over time steps, Heston hedging with CVaR objective (alpha = 0.5 and 0.99).
- Project README: [projects/03-deep-hedging/README.md](projects/03-deep-hedging/README.md)
- Notebook: [projects/03-deep-hedging/mlfi_project_3.ipynb](projects/03-deep-hedging/mlfi_project_3.ipynb)

### 4) Insurance Claim Frequency Prediction
- Goal: predict claim frequency and compare classical actuarial models with ML methods.
- Topics: Poisson GLM with exposure-weighted deviance, feature engineering, Poisson neural network, regression trees, random forest, gradient boosting with cross-validation.
- Project README: [projects/04-insurance-claim-prediction/README.md](projects/04-insurance-claim-prediction/README.md)
- Notebook: [projects/04-insurance-claim-prediction/mlfi_project_4.ipynb](projects/04-insurance-claim-prediction/mlfi_project_4.ipynb)
