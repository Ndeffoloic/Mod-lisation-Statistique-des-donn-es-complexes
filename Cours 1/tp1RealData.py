import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind
from statsmodels.stats.multitest import multipletests

# Charger le fichier datacancer.csv
file_name = 'datacancer.csv'
file_path = os.path.join(os.path.dirname(__file__), file_name)
df = pd.read_csv(file_path)

# Afficher les noms des colonnes pour vérifier leur existence
print("Noms des colonnes dans le DataFrame:")
print(df.columns)

# Sélectionner les covariables qualitatives et quantitatives
qualitative_vars = df.loc[:, 'TREATMENT':'Anti_TPO_antibodies_class'].columns  # Colonnes I à BH (indices 8 à 59)
quantitative_vars = df.loc[:, 'NBCYCLE_CT':].columns  # Colonnes 60 à la fin

print("Noms des colonnes dans le qualitative_vars:")
print(qualitative_vars)

# Variable d'intérêt
surv12 = df['Surv12']
surv6 = df['Surv6'] # j'ajoute ça au cas où. 

# Remplacement des valeurs NaN
# Remplacer les NaN dans les colonnes numériques par la médiane
df[quantitative_vars] = df[quantitative_vars].apply(lambda x: x.fillna(x.median()), axis=0)

# Remplacer les NaN dans les colonnes non numériques par le mode
df[qualitative_vars] = df[qualitative_vars].apply(lambda x: x.fillna(x.mode()[0]), axis=0)

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

# Ajuster les valeurs-p avec différentes méthodes
methods = ['bonferroni', 'sidak', 'holm', 'fdr_bh']
adjusted_p_values = {}
for method in methods:
    _, pvals_corrected, _, _ = multipletests(p_values_all, alpha=0.05, method=method)
    adjusted_p_values[method] = pvals_corrected

# Identifier les covariables qui peuvent expliquer la survie
significant_vars = {}
for method in methods:
    significant_vars[method] = [var for var, pval in zip(qualitative_vars.tolist() + quantitative_vars.tolist(), adjusted_p_values[method]) if pval < 0.05]

# Sauvegarder les résultats dans un fichier texte
with open('results_cancer.txt', 'w') as file:
    file.write("P-values for qualitative covariates:\n")
    for var, pval in zip(qualitative_vars, p_values_qualitative):
        file.write(f"{var}: {pval}\n")
    
    file.write("\nP-values for quantitative covariates:\n")
    for var, pval in zip(quantitative_vars, p_values_quantitative):
        file.write(f"{var}: {pval}\n")
    
    for method in methods:
        file.write(f"\nAdjusted p-values using {method} method:\n")
        for var, pval in zip(qualitative_vars.tolist() + quantitative_vars.tolist(), adjusted_p_values[method]):
            file.write(f"{var}: {pval}\n")
        
        file.write(f"\nSignificant variables using {method} method:\n")
        for var in significant_vars[method]:
            file.write(f"{var}\n")

# Tracer les histogrammes des valeurs-p ajustées
for method in methods:
    plt.hist(adjusted_p_values[method], bins=10, edgecolor='k')
    plt.xlabel('Adjusted p-values')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of adjusted p-values ({method})')
    plt.savefig(f'histogram_adjusted_{method}.png')
    plt.close()