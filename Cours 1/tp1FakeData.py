import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind

# Étape 1: Simuler un vecteur de variable binaire Y de taille n = 50
n = 50
Y = np.random.choice([0, 1], size=n)

# Fonction pour simuler X, effectuer les tests et calculer les erreurs
def simulation_et_tests(p):
    # Étape 2: Simuler une matrice de covariables quantitatives X
    X = np.random.normal(size=(n, p))
    
    # Étape 3: Effectuer le test t de Student pour chaque covariable
    p_values = []
    for i in range(p):
        X_when_Y_is_0 = X[Y == 0, i]
        X_when_Y_is_1 = X[Y == 1, i]
        _, p_value = ttest_ind(X_when_Y_is_0, X_when_Y_is_1)
        p_values.append(p_value)
    
    # Étape 4: Trouver les valeurs min et max des p-values et tracer l'histogramme
    min_p_value = min(p_values)
    max_p_value = max(p_values)
    print(f"Min p-value pour p={p}: {min_p_value}")
    print(f"Max p-value pour p={p}: {max_p_value}")
    
    plt.hist(p_values, bins=10, edgecolor='k')
    plt.xlabel('p-values')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of p-values for p={p}')
    plt.show()
    
    # Étape 5: Calculer le nombre de faux positifs pour un niveau alpha
    alpha = 0.05
    false_positives = sum(p_value < alpha for p_value in p_values)
    print(f"Nombre de faux positifs pour p={p}: {false_positives}")
    
    # Étape 6: Calculer l'erreur de type I empirique et la comparer à alpha
    empirical_type_I_error = false_positives / p
    comparison = "less than" if empirical_type_I_error < alpha else "greater than or equal to"
    print(f"L'erreur de type I empirique est {empirical_type_I_error:.2f}, ce qui est {comparison} alpha.")
    
    return p_values

# Étape 7: Répéter pour p = 100, p = 1000 et p = 10000
for p in [100, 1000, 10000]:
    simulation_et_tests(p)

# Étape 8: Simuler des covariables liées à Y selon différentes distributions
def simulate_linked_covariates(n, p1, Y, mean_factor):
    X = np.zeros((n, p1))
    for j in range(p1):
        for i in range(n):
            if np.random.rand() < 0.7:
                X[i, j] = np.random.normal(mean_factor * Y[i], 1)
            else:
                X[i, j] = np.random.normal(0, 1)
    return X

# Fonction pour effectuer les tests et calculer la puissance empirique
def test_linked_covariates(X, Y, alpha=0.05):
    p_values = []
    for i in range(X.shape[1]):
        X_when_Y_is_0 = X[Y == 0, i]
        X_when_Y_is_1 = X[Y == 1, i]
        _, p_value = ttest_ind(X_when_Y_is_0, X_when_Y_is_1)
        p_values.append(p_value)
    
    min_p_value = min(p_values)
    max_p_value = max(p_values)
    print(f"Min p-value: {min_p_value}")
    print(f"Max p-value: {max_p_value}")
    
    plt.hist(p_values, bins=10, edgecolor='k')
    plt.xlabel('p-values')
    plt.ylabel('Frequency')
    plt.title('Histogram of p-values')
    plt.show()
    
    significant_tests = sum(p_value < alpha for p_value in p_values)
    empirical_power = significant_tests / X.shape[1]
    print(f"Nombre de tests significatifs: {significant_tests}")
    print(f"La puissance empirique du test est: {empirical_power:.2f}")
    
    return p_values

# Étape 8a: Simuler et tester les covariables liées avec Xj ∼ 0.7N(3Y,1) + 0.3N(0,1)
p1 = 100
X_linked_3Y = simulate_linked_covariates(n, p1, Y, 3)
p_values_3Y = test_linked_covariates(X_linked_3Y, Y)

# Étape 8b: Simuler et tester les covariables liées avec Xj ∼ 0.7N(2Y,1) + 0.3N(0,1)
X_linked_2Y = simulate_linked_covariates(n, p1, Y, 2)
p_values_2Y = test_linked_covariates(X_linked_2Y, Y)

# Étape 8c: Simuler et tester les covariables liées avec Xj ∼ 0.7N(Y,1) + 0.3N(0,1)
X_linked_Y = simulate_linked_covariates(n, p1, Y, 1)
p_values_Y = test_linked_covariates(X_linked_Y, Y)

# Étape 8d: Simuler et tester les covariables liées avec Xj ∼ 0.3N(3Y,1) + 0.7N(0,1)
X_linked_3Y_03 = simulate_linked_covariates(n, p1, Y, 3)
p_values_3Y_03 = test_linked_covariates(X_linked_3Y_03, Y)

# Étape 8e: Simuler et tester les covariables liées avec Xj ∼ 0.3N(2Y,1) + 0.7N(0,1)
X_linked_2Y_03 = simulate_linked_covariates(n, p1, Y, 2)
p_values_2Y_03 = test_linked_covariates(X_linked_2Y_03, Y)

# Étape 8f: Simuler et tester les covariables liées avec Xj ∼ 0.3N(Y,1) + 0.7N(0,1)
X_linked_Y_03 = simulate_linked_covariates(n, p1, Y, 1)
p_values_Y_03 = test_linked_covariates(X_linked_Y_03, Y)

# Étape 9: Mélanger les covariables liées et non liées
X_combined = np.hstack((X_linked_3Y, X_linked_2Y, X_linked_Y, X_linked_3Y_03, X_linked_2Y_03, X_linked_Y_03))
X_unlinked = np.random.normal(size=(n, 6 * p1))
X_final = np.hstack((X_combined, X_unlinked))

# Étape 10: Calculer les valeurs-p pour les covariables liées et non liées
p_values_combined = test_linked_covariates(X_combined, Y)
p_values_unlinked = test_linked_covariates(X_unlinked, Y)

# Étape 11: Agréger les valeurs-p et analyser les résultats
p_values_all = p_values_combined + p_values_unlinked
min_p_value_all = min(p_values_all)
max_p_value_all = max(p_values_all)
print(f"Min p-value (all): {min_p_value_all}")
print(f"Max p-value (all): {max_p_value_all}")

plt.hist(p_values_all, bins=10, edgecolor='k')
plt.xlabel('p-values')
plt.ylabel('Frequency')
plt.title('Histogram of all p-values')
plt.show()

# Étape 12: Ajuster les valeurs-p selon différentes méthodes et comparer les résultats
from statsmodels.stats.multitest import multipletests

methods = ['bonferroni', 'sidak', 'holm', 'fdr_bh']
for method in methods:
    _, pvals_corrected, _, _ = multipletests(p_values_all, alpha=0.05, method=method)
    min_pval_corr = min(pvals_corrected)
    max_pval_corr = max(pvals_corrected)
    print(f"Method: {method}, Min p-value: {min_pval_corr}, Max p-value: {max_pval_corr}")
    
    plt.hist(pvals_corrected, bins=10, edgecolor='k')
    plt.xlabel('Adjusted p-values')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of adjusted p-values ({method})')
    plt.show()