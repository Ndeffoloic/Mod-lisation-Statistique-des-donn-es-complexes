import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Simulate the model for X and R
np.random.seed(0)
n = 100
X1 = np.random.normal(0, 1, n)
print("Voici X1 :",X1)

X2 = 0.5 * X1 + np.random.normal(0, 1, n)
R = np.random.binomial(1, 0.7, n)  # Missingness indicator

# Introduce missing values in X2 based on R
X2_obs = np.where(R == 1, X2, np.nan)

# Compute the mean of X2
mean_X2 = np.nanmean(X2)
print(f"Mean of X2: {mean_X2}")

# Compute the mean of (X2)obs
mean_X2_obs = np.nanmean(X2_obs)
print(f"Mean of observed X2: {mean_X2_obs}")

# Impute data by the mean of the observed values
X2_mean_imputed = np.where(np.isnan(X2_obs), mean_X2_obs, X2_obs)
mean_X2_mean_imputed = np.mean(X2_mean_imputed)
print(f"Mean of mean-imputed X2: {mean_X2_mean_imputed}")

# Impute data by a regression method
# Prepare the data for regression
observed_mask = ~np.isnan(X2_obs)
X1_obs = X1[observed_mask].reshape(-1, 1)
X2_obs_nonan = X2_obs[observed_mask]

# Fit the regression model
reg = LinearRegression().fit(X1_obs, X2_obs_nonan)

# Predict the missing values
X1_missing = X1[~observed_mask].reshape(-1, 1)
X2_pred = reg.predict(X1_missing)

# Imputing the missing values
X2_regression_imputed = X2_obs.copy()
X2_regression_imputed[~observed_mask] = X2_pred
mean_X2_regression_imputed = np.mean(X2_regression_imputed)
print(f"Mean of regression-imputed X2: {mean_X2_regression_imputed}")