import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import Lasso, Ridge

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
print(X)

# Tracer la matrice de corrélation entre les différentes variables (colonnes 1 à 8)
correlation_matrix = data.iloc[:, 1:8].corr()
plt.figure(figsize=(10, 8))
plt.title('Matrice de Corrélation')
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.show()
Y = data.iloc[:, 9].values
print(Y)
# Ridge regression
# Définir une plage de valeurs pour le paramètre de régularisation alpha (lambda)
alphas = np.arange(0, 25, 0.05)
alphasLasso = np.arange(0, 1, 0.05)

# Initialiser les modèles de régression Ridge et Lasso
ridge = Ridge()
lasso = Lasso()

# Listes pour stocker les coefficients pour chaque valeur de alpha
coefsRidge = []
coefsLasso = []

# Boucle sur chaque valeur de alpha
for a in alphas:
    # Définir le paramètre alpha du modèle Ridge
    ridge.set_params(alpha=a)
    
    # Ajuster le modèle Ridge aux données
    ridge.fit(X, Y)
    
    # Ajouter les coefficients du modèle ajusté à la liste
    coefsRidge.append(ridge.coef_)
for a in alphasLasso : 
    
    # Définir le paramètre alpha du modèle Lasso
    lasso.set_params(alpha=a)
    
    # Ajuster le modèle Lasso aux données
    lasso.fit(X, Y)
    
    # Ajouter les coefficients du modèle ajusté à la liste
    coefsLasso.append(lasso.coef_)

coefsRidge = np.array(coefsRidge)
coefsLasso = np.array(coefsLasso)

# Tracer les coefficients en fonction de lambda
plt.figure(figsize=(12, 6))

# Tracé des coefficients Ridge
plt.subplot(1, 2, 1)
for i in range(coefsRidge.shape[1]):
    plt.plot(alphas, coefsRidge[:, i], label=data.columns[i+1])
plt.xlabel('Lambda')
plt.ylabel('Coefficients')
plt.title('Ridge Regression Coefficients')
plt.axhline(0, color='black', linestyle='--')
plt.legend(loc='upper right')

# Tracé des coefficients Lasso
plt.subplot(1, 2, 2)
for i in range(coefsLasso.shape[1]):
    plt.plot(alphasLasso, coefsLasso[:, i], label=data.columns[i+1])
plt.xlabel('Lambda')
plt.ylabel('Coefficients')
plt.title('Lasso Regression Coefficients')
plt.axhline(0, color='black', linestyle='--')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()