import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.api import OLS, add_constant
from statsmodels.tools.eval_measures import aic, bic

# Nom du fichier à ouvrir
file_name = 'poumon.txt'

# Obtenir le chemin complet du fichier
file_path = os.path.join(os.path.dirname(__file__), file_name)

# Lire le fichier dans un DataFrame avec les bons séparateurs
data = pd.read_csv(file_path, sep='\t', decimal=',')

# Sélectionner les variables d'intérêt
variables = ['origine', 'Sexe', 'AGE', 'TAILLE_EN_M', 'POIDS', 'BMI', 'TLCO']
data_selected = data[variables]

# Afficher les premières lignes pour vérifier les données
pd.set_option('display.max_columns', None)  # Afficher toutes les colonnes
pd.set_option('display.width', None)  # Ajuster la largeur d'affichage
print(data_selected.head())

# Calculer les valeurs de n et p
n = data_selected.shape[0]  # nombre d'observations
p = data_selected.shape[1] - 1  # nombre de variables explicatives (excluant TLCO)

print(f"Nombre d'observations (n) : {n}")
print(f"Nombre de variables explicatives (p) : {p}")

# Calculer le nombre de possibilités pour le sous-ensemble de données dans le modèle
# Le nombre de sous-ensembles possibles est 2^p
num_possibilities = 2 ** p
print(f"Nombre de possibilités pour le sous-ensemble de données dans le modèle : {num_possibilities}")

# Préparer les données pour la régression
X = data_selected[['origine', 'Sexe', 'AGE', 'TAILLE_EN_M', 'POIDS', 'BMI']]
y = data_selected['TLCO']

# Convertir les variables catégorielles en variables numériques
X = pd.get_dummies(X, drop_first=True)

# Vérifier les types de données pour s'assurer qu'ils sont tous numériques
print(X.dtypes)

# Ajouter une constante pour le modèle OLS
X = add_constant(X)

# Créer et ajuster le modèle de régression linéaire
model = OLS(y, X).fit()

# Afficher les coefficients du modèle
print(model.summary())

# Calculer les résidus
residuals = model.resid
print("Premiers résidus :")
print(residuals.head())

# Calculer RSS
RSS = np.sum(residuals ** 2)
print(f"RSS : {RSS}")

# Prédictions
predictions = model.predict(X)
RSS2 = np.sum((predictions - y) ** 2)
print(f"RSS2 : {RSS2}")

# Calculer sigma^2
sigma_squared = model.mse_resid
print(f"Sigma^2 : {sigma_squared}")

# R-squared et Adjusted R-squared
r_squared = model.rsquared
adj_r_squared = model.rsquared_adj
print(f"R-squared : {r_squared}")
print(f"Adjusted R-squared : {adj_r_squared}")

# Calculer AIC et BIC
AIC = model.aic
BIC = model.bic
print(f"AIC : {AIC}")
print(f"BIC : {BIC}")

# Fonction pour calculer Cp
def calculate_cp(model, X, y):
    n = len(y)
    p = X.shape[1]
    RSS = np.sum(model.resid ** 2)
    sigma_squared = model.mse_resid
    Cp = RSS / sigma_squared - n + 2 * p
    return Cp

Cp = calculate_cp(model, X, y)
print(f"Cp : {Cp}")

# Sélection backward stepwise
def backward_stepwise_selection(X, y, significance_level=0.05):
    initial_features = X.columns.tolist()
    best_features = initial_features.copy()
    while len(best_features) > 0:
        model = OLS(y, X[best_features]).fit()
        p_values = model.pvalues
        max_p_value = p_values.max()
        if max_p_value > significance_level:
            excluded_feature = p_values.idxmax()
            best_features.remove(excluded_feature)
            print(f"Excluding {excluded_feature} with p-value {max_p_value}")
        else:
            break
    return best_features

selected_features = backward_stepwise_selection(X, y)
print(f"Selected features: {selected_features}")

# Ajuster le modèle final avec les variables sélectionnées
final_model = OLS(y, X[selected_features]).fit()
print(final_model.summary())