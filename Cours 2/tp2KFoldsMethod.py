import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
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

# Fonction pour effectuer la validation croisée K-fold
def k_fold_validation(X, y, selected_features, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=1)
    rmse_list = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model = OLS(y_train, X_train[selected_features]).fit()
        rmse = calculate_rmse(model, X_test[selected_features], y_test)
        rmse_list.append(rmse)
    
    return rmse_list

# 1. 5-Fold method
rmse_5_fold = {i: [] for i in range(3)}

for i in range(3):
    selected_features = best_models[i]
    rmse_5_fold[i] = k_fold_validation(X, y, selected_features, k=5)

# 2. Répéter la méthode 100 fois et tracer le boxplot des erreurs
rmse_5_fold_repeated = {i: [] for i in range(3)}

for _ in range(100):
    for i in range(3):
        selected_features = best_models[i]
        rmse_list = k_fold_validation(X, y, selected_features, k=5)
        rmse_5_fold_repeated[i].extend(rmse_list)

# Tracer le boxplot des RMSE pour 5-Fold
plt.figure(figsize=(10, 6))
plt.boxplot([rmse_5_fold_repeated[i] for i in range(3)], labels=[f'Model {i+1}' for i in range(3)])
plt.xlabel('Models')
plt.ylabel('RMSE')
plt.title('Boxplot of RMSE for 5-Fold Cross-Validation')
plt.show()

# 3. 10-Fold method
rmse_10_fold = {i: [] for i in range(3)}

for i in range(3):
    selected_features = best_models[i]
    rmse_10_fold[i] = k_fold_validation(X, y, selected_features, k=10)

# Répéter la méthode 100 fois et tracer le boxplot des erreurs pour 10-Fold
rmse_10_fold_repeated = {i: [] for i in range(3)}

for _ in range(100):
    for i in range(3):
        selected_features = best_models[i]
        rmse_list = k_fold_validation(X, y, selected_features, k=10)
        rmse_10_fold_repeated[i].extend(rmse_list)

# Tracer le boxplot des RMSE pour 10-Fold
plt.figure(figsize=(10, 6))
plt.boxplot([rmse_10_fold_repeated[i] for i in range(3)], labels=[f'Model {i+1}' for i in range(3)])
plt.xlabel('Models')
plt.ylabel('RMSE')
plt.title('Boxplot of RMSE for 10-Fold Cross-Validation')
plt.show()