import os
from contextlib import redirect_stdout

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score
from statsmodels.api import Logit, add_constant
from statsmodels.stats.multitest import multipletests

# Charger le fichier datacancer.csv
file_name = 'datacancer.csv'
file_path = os.path.join(os.path.dirname(__file__), file_name)
df = pd.read_csv(file_path)

# Supprimer les individus ayant des données manquantes
ind = np.where(df.isna().sum(axis=1) == 0)[0]
df = df.iloc[ind]

# Sélectionner les covariables qualitatives et quantitatives
qualitative_vars = df.loc[:, 'TREATMENT':'Anti_TPO_antibodies_class'].columns
quantitative_vars = df.loc[:, 'NBCYCLE_CT':].columns

# Variable d'intérêt
surv12 = df['Surv12']

# Fonction pour effectuer les tests sur les covariables qualitatives
def test_qualitative_covariates(df, qualitative_vars, surv12):
    p_values = []
    for var in qualitative_vars:
        contingency_table = pd.crosstab(df[var], surv12)
        _, p_value, _, _ = chi2_contingency(contingency_table)
        p_values.append(p_value)
    return p_values

# Fonction pour effectuer les tests sur les covariables quantitatives
def test_quantitative_covariates(df, quantitative_vars, surv12):
    p_values = []
    for var in quantitative_vars:
        group1 = df[df['Surv12'] == 0][var]
        group2 = df[df['Surv12'] == 1][var]
        _, p_value = ttest_ind(group1, group2)
        p_values.append(p_value)
    return p_values

# Effectuer les tests et sauvegarder les valeurs-p
p_values_qualitative = test_qualitative_covariates(df, qualitative_vars, surv12)
p_values_quantitative = test_quantitative_covariates(df, quantitative_vars, surv12)

# Agréger les vecteurs de valeurs-p
p_values_all = p_values_qualitative + p_values_quantitative

# Ajuster les valeurs-p avec la méthode de Benjamini-Hochberg
_, pvals_corrected, _, _ = multipletests(p_values_all, alpha=0.05, method='fdr_bh')

# Identifier les covariables significatives
significant_vars = [var for var, pval in zip(qualitative_vars.tolist() + quantitative_vars.tolist(), pvals_corrected) if pval < 0.05]

# Construire le DataFrame avec la variable Surv12 et les covariables sélectionnées
frame = df[['Surv12'] + significant_vars]

# Appliquer les méthodes forward et backward pour sélectionner les variables dans le modèle logistique
def forward_selection(data, response):
    initial_features = []
    best_features = initial_features.copy()
    remaining_features = list(data.columns)
    remaining_features.remove(response)
    current_score, best_new_score = float('inf'), float('inf')
    
    while remaining_features and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining_features:
            features = best_features + [candidate]
            model = Logit(data[response], add_constant(data[features])).fit(disp=0)
            score = model.aic
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates[0]
        if current_score > best_new_score:
            remaining_features.remove(best_candidate)
            best_features.append(best_candidate)
            current_score = best_new_score
    return best_features

def backward_selection(data, response, significance_level=0.05):
    initial_features = list(data.columns)
    initial_features.remove(response)
    best_features = initial_features.copy()
    
    while len(best_features) > 0:
        model = Logit(data[response], add_constant(data[best_features])).fit(disp=0)
        p_values = model.pvalues[1:]  # Exclure la constante
        max_p_value = p_values.max()
        if max_p_value > significance_level:
            excluded_feature = p_values.idxmax()
            best_features.remove(excluded_feature)
        else:
            break
    return best_features

selected_features_forward = forward_selection(frame, 'Surv12')
selected_features_backward = backward_selection(frame, 'Surv12')

# Ajuster les modèles logistiques avec les variables sélectionnées
model_forward = Logit(frame['Surv12'], add_constant(frame[selected_features_forward])).fit(disp=0)
model_backward = Logit(frame['Surv12'], add_constant(frame[selected_features_backward])).fit(disp=0)

# Construire la matrice de confusion et les courbes ROC des deux modèles
def plot_roc_curve(model, X, y, label):
    y_pred_prob = model.predict(add_constant(X))
    fpr, tpr, _ = roc_curve(y, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')
    return roc_auc

plt.figure(figsize=(10, 6))
auc_forward = plot_roc_curve(model_forward, frame[selected_features_forward], frame['Surv12'], 'Forward Selection')
auc_backward = plot_roc_curve(model_backward, frame[selected_features_backward], frame['Surv12'], 'Backward Selection')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.savefig('roc_curve.png')
plt.close()

# Comparer les modèles par validation croisée
def cross_val_score_auc(model, X, y, cv=5):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=1)
    aucs = []
    accuracies = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model_fit = Logit(y_train, add_constant(X_train)).fit(disp=0)
        y_pred_prob = model_fit.predict(add_constant(X_test))
        y_pred = (y_pred_prob >= 0.5).astype(int)
        
        # Trier les valeurs de y_test et y_pred_prob
        sorted_indices = np.argsort(y_test)
        y_test_sorted = y_test.iloc[sorted_indices]
        y_pred_prob_sorted = y_pred_prob[sorted_indices]
        
        aucs.append(auc(y_test_sorted, y_pred_prob_sorted))
        accuracies.append(accuracy_score(y_test, y_pred))
    return np.mean(aucs), np.std(aucs), np.mean(accuracies), np.std(accuracies)

mean_auc_forward, std_auc_forward, mean_acc_forward, std_acc_forward = cross_val_score_auc(model_forward, frame[selected_features_forward], frame['Surv12'])
mean_auc_backward, std_auc_backward, mean_acc_backward, std_acc_backward = cross_val_score_auc(model_backward, frame[selected_features_backward], frame['Surv12'])

# Exporter les résultats dans un fichier
with open('classification_results.txt', 'w') as f:
    with redirect_stdout(f):
        print("Selected features with forward selection:", selected_features_forward)
        print("Selected features with backward selection:", selected_features_backward)
        print("\n")
        
        print("Model summary for forward selection:")
        print(model_forward.summary())
        print("\n")
        
        print("Model summary for backward selection:")
        print(model_backward.summary())
        print("\n")
        
        print(f"AUC for forward selection model: {auc_forward:.2f}")
        print(f"AUC for backward selection model: {auc_backward:.2f}")
        print("\n")
        
        print(f"Cross-validated AUC for forward selection model: {mean_auc_forward:.2f} ± {std_auc_forward:.2f}")
        print(f"Cross-validated AUC for backward selection model: {mean_auc_backward:.2f} ± {std_auc_backward:.2f}")
        print("\n")
        
        print(f"Cross-validated accuracy for forward selection model: {mean_acc_forward:.2f} ± {std_acc_forward:.2f}")
        print(f"Cross-validated accuracy for backward selection model: {mean_acc_backward:.2f} ± {std_acc_backward:.2f}")
        print("\n")
        
        # Choisir le meilleur modèle pour prédire la survie
        if mean_auc_forward > mean_auc_backward:
            print("The forward selection model is chosen as the best model for predicting survival.")
        else:
            print("The backward selection model is chosen as the best model for predicting survival.")