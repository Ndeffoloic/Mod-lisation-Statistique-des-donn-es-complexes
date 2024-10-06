import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.api import OLS, add_constant

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

# Préparer les données pour la régression
X = data_selected[['origine', 'Sexe', 'AGE', 'TAILLE_EN_M', 'POIDS', 'BMI']]
y = data_selected['TLCO']

# Convertir les variables catégorielles en variables numériques
X = pd.get_dummies(X, drop_first=True)

# Ajouter une constante pour le modèle OLS
X = add_constant(X)

# Fonction pour la sélection forward stepwise
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

# Sélectionner les trois meilleurs modèles parmi les six évalués
best_models = []
for num_features in range(1, 7):
    selected_features = forward_stepwise_selection(X, y, criterion='aic')
    best_models.append(selected_features)

# Fonction pour calculer le RMSE
def calculate_rmse(model, X, y):
    predictions = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    return rmse

# Fonction pour effectuer le bootstrap
def bootstrap(X, y, selected_features, B=1000):
    n = len(y)
    rmse_list = []
    
    for _ in range(B):
        sample_indices = np.random.choice(n, size=n, replace=True)
        X_sample = X.iloc[sample_indices]
        y_sample = y.iloc[sample_indices]
        
        model = OLS(y_sample, X_sample[selected_features]).fit()
        rmse = calculate_rmse(model, X_sample[selected_features], y_sample)
        rmse_list.append(rmse)
    
    return rmse_list

# Calculer l'erreur Bootstrap pour les trois meilleurs modèles
bootstrap_errors = {i: [] for i in range(3)}

for i in range(3):
    selected_features = best_models[i]
    bootstrap_errors[i] = bootstrap(X, y, selected_features, B=1000)

# Exporter les résultats dans un fichier
with open('bootstrap_results.txt', 'w') as f:
    for i in range(3):
        f.write(f"Bootstrap errors for Model {i+1}:\n")
        f.write(f"Mean RMSE: {np.mean(bootstrap_errors[i])}\n")
        f.write(f"Standard Deviation of RMSE: {np.std(bootstrap_errors[i])}\n")
        f.write("\n")

# Tracer le boxplot des erreurs Bootstrap
plt.figure(figsize=(10, 6))
plt.boxplot([bootstrap_errors[i] for i in range(3)], tick_labels=[f'Model {i+1}' for i in range(3)])
plt.xlabel('Models')
plt.ylabel('RMSE')
plt.title('Boxplot of Bootstrap RMSE for Different Models')
plt.show()

# Comparer les résultats des différentes méthodes et proposer un modèle final
# (Cette partie est subjective et dépend de l'analyse des résultats obtenus)