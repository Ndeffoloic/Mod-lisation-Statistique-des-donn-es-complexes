import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV, cross_val_score

# Nom du fichier à ouvrir
file_name = 'prostate.txt'

# Obtenir le chemin complet du fichier
file_path = os.path.join(os.path.dirname(__file__), file_name)

# Lire le fichier dans un DataFrame avec les bons séparateurs
data = pd.read_csv(file_path, sep='\t', header=0, decimal='.')

print(data.columns)
print(data.head())

# Préparer les données pour la régression
X = data.iloc[:, 1:8].values
Y = data.iloc[:, 9].values

# Tracer la matrice de corrélation entre les différentes variables (colonnes 1 à 8)
correlation_matrix = data.iloc[:, 1:8].corr()
plt.figure(figsize=(10, 8))
plt.title('Matrice de Corrélation')
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.show()

# Définir une plage de valeurs pour le paramètre de régularisation alpha (lambda)
alphas = np.arange(0, 25, 0.05)
alphasLasso = np.arange(0, 1, 0.05)

# Initialiser les modèles de régression Ridge et Lasso
ridge = Ridge()
lasso = Lasso()

# Utiliser GridSearchCV pour trouver la meilleure valeur de alpha pour Ridge
ridge_cv = GridSearchCV(ridge, param_grid={'alpha': alphas}, cv=5)
ridge_cv.fit(X, Y)
best_alpha_ridge = ridge_cv.best_params_['alpha']
print(f"Best alpha for Ridge: {best_alpha_ridge}")

# Utiliser GridSearchCV pour trouver la meilleure valeur de alpha pour Lasso
lasso_cv = GridSearchCV(lasso, param_grid={'alpha': alphasLasso}, cv=5)
lasso_cv.fit(X, Y)
best_alpha_lasso = lasso_cv.best_params_['alpha']
print(f"Best alpha for Lasso: {best_alpha_lasso}")

# Ajuster les modèles avec les meilleurs alpha
ridge.set_params(alpha=best_alpha_ridge)
ridge.fit(X, Y)
lasso.set_params(alpha=best_alpha_lasso)
lasso.fit(X, Y)

# Tracer les coefficients en fonction de lambda
plt.figure(figsize=(12, 6))

# Tracé des coefficients Ridge
plt.subplot(1, 2, 1)
for i in range(ridge.coef_.shape[0]):
    plt.plot(alphas, [ridge_cv.best_estimator_.coef_[i]] * len(alphas), label=data.columns[i+1])
plt.xlabel('Lambda')
plt.ylabel('Coefficients')
plt.title('Ridge Regression Coefficients')
plt.axhline(0, color='black', linestyle='--')
plt.legend(loc='upper right')

# Tracé des coefficients Lasso
plt.subplot(1, 2, 2)
for i in range(lasso.coef_.shape[0]):
    plt.plot(alphasLasso, [lasso_cv.best_estimator_.coef_[i]] * len(alphasLasso), label=data.columns[i+1])
plt.xlabel('Lambda')
plt.ylabel('Coefficients')
plt.title('Lasso Regression Coefficients')
plt.axhline(0, color='black', linestyle='--')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

# Calculer l'erreur de validation croisée pour les deux modèles
ridge_cv_error = -cross_val_score(ridge, X, Y, cv=5, scoring='neg_mean_squared_error').mean()
lasso_cv_error = -cross_val_score(lasso, X, Y, cv=5, scoring='neg_mean_squared_error').mean()

print(f"Cross-validation error for Ridge: {ridge_cv_error}")
print(f"Cross-validation error for Lasso: {lasso_cv_error}")

# Refaire le modèle avec les 3 variables les plus importantes pour Lasso
important_features = np.argsort(np.abs(lasso.coef_))[-3:]
X_important = X[:, important_features]

lasso_important = Lasso(alpha=best_alpha_lasso)
lasso_important.fit(X_important, Y)

# Tracer les valeurs ajustées
plt.figure(figsize=(8, 6))
plt.plot(Y, label='True Values')
plt.plot(lasso_important.predict(X_important), label='Fitted Values')
plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Fitted Values vs True Values')
plt.legend()
plt.show()