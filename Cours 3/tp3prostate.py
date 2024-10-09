import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import cross_val_score

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

# Définir une plage de valeurs pour le paramètre de régularisation alpha (lambda)
alphas = np.arange(0, 25, 0.05)

# Initialiser les modèles de régression Ridge et Lasso
ridge = Ridge()
lasso = Lasso()

# Effectuer la validation croisée pour trouver le meilleur alpha pour Ridge

ridge_scores = [cross_val_score(Ridge(alpha=a), X, Y, cv=5).mean() for a in alphas]
best_alpha_ridge = alphas[np.argmax(ridge_scores)]
print(f"Best alpha for Ridge: {best_alpha_ridge}")

# Effectuer la validation croisée pour trouver le meilleur alpha pour Lasso
lasso_scores = [cross_val_score(Lasso(alpha=a), X, Y, cv=5).mean() for a in alphas]
best_alpha_lasso = alphas[np.argmax(lasso_scores)]
print(f"Best alpha for Lasso: {best_alpha_lasso}")

# Ajuster les modèles avec les meilleurs alpha
ridge.set_params(alpha=best_alpha_ridge)
lasso.set_params(alpha=best_alpha_lasso)

ridge.fit(X, Y)
lasso.fit(X, Y)

# Extraire les coefficients
ridge_coefs = ridge.coef_
lasso_coefs = lasso.coef_

# Créer un DataFrame pour les coefficients
df_coefs = pd.DataFrame({
    'Variable': data.columns[1:8],
    'Ridge': ridge_coefs,
    'Lasso': lasso_coefs
})

# Tracer les coefficients
df_coefs.set_index('Variable').plot(kind='bar')
plt.title('Coefficients des variables pour Ridge et Lasso')
plt.xlabel('Variables')
plt.ylabel('Coefficients')
plt.show()
