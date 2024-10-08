import os
from contextlib import redirect_stdout
from math import log

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

# Rediriger les sorties vers un fichier
with open('resultat.txt', 'w') as f:
    with redirect_stdout(f):
        # Afficher les premières lignes pour vérifier les données
        pd.set_option('display.max_columns', None)  # Afficher toutes les colonnes
        pd.set_option('display.width', None)  # Ajuster la largeur d'affichage
        print("Premières lignes des données sélectionnées :")
        print(data_selected.head())
        print("\n")

        # Calculer les valeurs de n et p
        n = data_selected.shape[0]  # nombre d'observations
        p = data_selected.shape[1] - 1  # nombre de variables explicatives (excluant TLCO)

        print(f"Nombre d'observations (n) : {n}")
        print(f"Nombre de variables explicatives (p) : {p}")
        print("\n")

        # Calculer le nombre de possibilités pour le sous-ensemble de données dans le modèle
        # Le nombre de sous-ensembles possibles est 2^p
        num_possibilities = 2 ** p
        print(f"Nombre de possibilités pour le sous-ensemble de données dans le modèle : {num_possibilities}")
        print("\n")

        # Préparer les données pour la régression
        X = data_selected[['origine', 'Sexe', 'AGE', 'TAILLE_EN_M', 'POIDS', 'BMI']]
        y = data_selected['TLCO']

        # Convertir les variables catégorielles en variables numériques
        X = pd.get_dummies(X, drop_first=True)

        # Vérifier les types de données pour s'assurer qu'ils sont tous numériques
        print("Types de données des variables explicatives :")
        print(X.dtypes)
        print("\n")

        # Ajouter une constante pour le modèle OLS
        X = add_constant(X)

        # Créer et ajuster le modèle de régression linéaire
        model = OLS(y, X).fit()

        # Afficher les coefficients du modèle
        print("Résumé du modèle de régression linéaire :")
        print(model.summary())
        print("\n")

        # Calculer les résidus
        residuals = model.resid
        print("Premiers résidus :")
        print(residuals.head())
        print("\n")

        # Calculer RSS
        RSS = np.sum(residuals ** 2)
        print(f"RSS : {RSS}")
        print("\n")

        # Prédictions
        predictions = model.predict(X)
        RSS2 = np.sum((predictions - y) ** 2)
        print(f"RSS2 : {RSS2}")
        print("\n")

        # Calculer sigma^2
        sigma_squared = model.mse_resid
        print(f"Sigma^2 : {sigma_squared}")
        print("\n")

        # R-squared et Adjusted R-squared
        r_squared = model.rsquared
        adj_r_squared = model.rsquared_adj
        print(f"R-squared : {r_squared}")
        print(f"Adjusted R-squared : {adj_r_squared}")
        print("\n")

        # Calculer AIC et BIC
        AIC = model.aic
        BIC = model.bic
        print(f"AIC : {AIC}")
        print(f"BIC : {BIC}")
        print("\n")

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
        print("\n")

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
        print("\n")

        # Ajuster le modèle final avec les variables sélectionnées
        final_model = OLS(y, X[selected_features]).fit()
        print("Résumé du modèle final après sélection backward stepwise :")
        print(final_model.summary())
        print("\n")

        # Sélection forward stepwise avec AIC
        def forward_stepwise_selection(X, y, criterion='aic'):
            initial_features = []
            best_features = initial_features.copy()
            remaining_features = list(X.columns)
            current_score, best_new_score = float('inf'), float('inf')
            while remaining_features and current_score == best_new_score:
                scores_with_candidates = []
                for candidate in remaining_features:
                    features = best_features + [candidate]
                    model = OLS(y, X[features]).fit()
                    if criterion == 'aic':
                        score = model.aic
                    elif criterion == 'bic':
                        score = model.bic
                    scores_with_candidates.append((score, candidate))
                scores_with_candidates.sort()
                best_new_score, best_candidate = scores_with_candidates[0]
                if current_score > best_new_score:
                    remaining_features.remove(best_candidate)
                    best_features.append(best_candidate)
                    current_score = best_new_score
            return best_features

        selected_features_aic = forward_stepwise_selection(X, y, criterion='aic')
        print(f"Selected features with AIC: {selected_features_aic}")
        print("\n")

        selected_features_bic = forward_stepwise_selection(X, y, criterion='bic')
        print(f"Selected features with BIC: {selected_features_bic}")
        print("\n")

        # Ajuster le modèle final avec les variables sélectionnées par AIC
        final_model_aic = OLS(y, X[selected_features_aic]).fit()
        print("Résumé du modèle final après sélection forward stepwise avec AIC :")
        print(final_model_aic.summary())
        print("\n")

        # Ajuster le modèle final avec les variables sélectionnées par BIC
        final_model_bic = OLS(y, X[selected_features_bic]).fit()
        print("Résumé du modèle final après sélection forward stepwise avec BIC :")
        print(final_model_bic.summary())