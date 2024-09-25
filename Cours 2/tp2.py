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

# Fonction pour convertir les lettres en indices alphabétiques
def letter_to_index(letter):
    return ord(letter.upper()) - ord('A') + 1

# Appliquer la fonction aux colonnes 'Sexe' et 'Origine'
data['Sexe'] = data['Sexe'].apply(letter_to_index)
data['origine'] = data['origine'].apply(letter_to_index)

# Sélectionner les variables d'intérêt
variables = ['origine', 'Sexe', 'AGE', 'TAILLE_EN_M', 'POIDS', 'BMI', 'TLCO']
data_selected = data[variables]

# Rediriger la sortie vers un fichier
with open('results.txt', 'w') as out:
    # Afficher les premières lignes pour vérifier les données
    pd.set_option('display.max_columns', None)  # Afficher toutes les colonnes
    pd.set_option('display.width', None)  # Ajuster la largeur d'affichage
    out.write(data_selected.head().to_string() + '\n')

    # Calculer les valeurs de n et p
    n = data_selected.shape[0]  # nombre d'observations
    p = data_selected.shape[1] - 1  # nombre de variables explicatives (excluant TLCO)

    out.write(f"Nombre d'observations (n) : {n}\n")
    out.write(f"Nombre de variables explicatives (p) : {p}\n")

    # Calculer le nombre de possibilités pour le sous-ensemble de données dans le modèle
    # Le nombre de sous-ensembles possibles est 2^p
    num_possibilities = 2 ** p
    out.write(f"Nombre de possibilités pour le sous-ensemble de données dans le modèle : {num_possibilities}\n")

    # Préparer les données pour la régression
    X = data_selected[['origine', 'Sexe', 'AGE', 'TAILLE_EN_M', 'POIDS', 'BMI']]
    y = data_selected['TLCO']

    # Convertir les variables catégorielles en variables numériques
    X = pd.get_dummies(X, drop_first=True)

    # Vérifier les types de données pour s'assurer qu'ils sont tous numériques
    out.write(X.dtypes.to_string() + '\n')

    # Ajouter une constante pour le modèle OLS
    X = add_constant(X)

    # Créer et ajuster le modèle de régression linéaire
    model = OLS(y, X).fit()

    # Afficher les coefficients du modèle
    out.write(model.summary().as_text() + '\n')

    # Calculer les résidus
    residuals = model.resid
    out.write("Premiers résidus :\n")
    out.write(residuals.head().to_string() + '\n')

    # Calculer RSS
    RSS = np.sum(residuals ** 2)
    out.write(f"RSS : {RSS}\n")

    # Prédictions
    predictions = model.predict(X)
    RSS2 = np.sum((predictions - y) ** 2)
    out.write(f"RSS2 : {RSS2}\n")

    # Calculer sigma^2
    sigma_squared = model.mse_resid
    out.write(f"Sigma^2 : {sigma_squared}\n")

    # R-squared et Adjusted R-squared
    r_squared = model.rsquared
    adj_r_squared = model.rsquared_adj
    out.write(f"R-squared : {r_squared}\n")
    out.write(f"Adjusted R-squared : {adj_r_squared}\n")

    # Calculer AIC et BIC
    AIC = model.aic
    BIC = model.bic
    out.write(f"AIC : {AIC}\n")
    out.write(f"BIC : {BIC}\n")

    # Fonction pour calculer Cp
    def calculate_cp(model, X, y):
        n = len(y)
        p = X.shape[1]
        RSS = np.sum(model.resid ** 2)
        sigma_squared = model.mse_resid
        Cp = RSS / sigma_squared - n + 2 * p
        return Cp

    Cp = calculate_cp(model, X, y)
    out.write(f"Cp : {Cp}\n")

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
                out.write(f"Excluding {excluded_feature} with p-value {max_p_value}\n")
            else:
                break
        return best_features

    selected_features = backward_stepwise_selection(X, y)
    out.write(f"Selected features: {selected_features}\n")

    # Ajuster le modèle final avec les variables sélectionnées
    final_model = OLS(y, X[selected_features]).fit()
    out.write(final_model.summary().as_text() + '\n')
