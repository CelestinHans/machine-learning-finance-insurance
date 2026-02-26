# Project 1: Linear Regression and Regularization (Housing Prices)

## Summary
Predict house prices using the Ames Housing dataset and compare ordinary least squares against several regularization techniques, with an emphasis on both predictive performance and interpretability.

## What this notebook covers
- Data split (70% train, 30% test)
- Target distribution analysis and logarithm transformation
- Missing value imputation using training statistics
  - Numerical: mean imputation
  - Categorical: most frequent category (handling true missingness vs valid "None"/"NA" levels)
- Standardization of numerical features and one-hot encoding of categorical variables
- OLS with numerical features only (sklearn), evaluated with MSE and R2
- OLS via matrix algebra (numpy)
  - Coefficients via solve
  - Standard errors from estimated residual variance and $(AᵀA)^{-1}$
  - Pseudoinverse comparison and when it differs
  - Cross-check with statsmodels
- Regularization on the full feature set (numerical + dummies)
  - Truncated pseudoinverse, Ridge, Lasso, Elastic Net
  - 8-fold cross-validation for hyperparameter tuning
- Discussion: preprocessing leakage and why the intercept is not penalized

## Conclusion
Based on the results, we recommend using the **Elastic Net** model for predicting house prices. It achieved the best overall balance between bias and variance, with the highest test $R^2$ and the lowest out-of-sample MSE among all models. Compared to OLS, which clearly overfit when using the full feature set (or was less accurate when restricted to numerical features), and Ridge, which retained all coefficients, Elastic Net provided a more robust and interpretable model while maintaining strong predictive accuracy.

Conceptually, this choice aligns well with the structure of the housing price prediction problem. The dataset contains a relatively large number of features (including both numerical and one-hot encoded categorical variables) that are often correlated and potentially redundant. The Elastic Net’s combination of **L1** and **L2** penalties is particularly effective in this context: it can perform variable selection like Lasso while stabilizing coefficients among correlated predictors like Ridge.
