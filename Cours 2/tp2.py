import os

import pandas as pd
from sklearn.linear_model import LinearRegression

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

# Créer et ajuster le modèle de régression linéaire
model = LinearRegression()
model.fit(X, y)

# Afficher les coefficients du modèle
print("Coefficients du modèle :")
print(model.coef_)
print("Intercept du modèle :")
print(model.intercept_)