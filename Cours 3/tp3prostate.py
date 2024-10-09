import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
X = data.iloc[:, :8].values
Y = data.iloc[:, 8].values

# Ridge regression
alphas = np.arange(0, 25, 0.05)
ridge = Ridge()
coefs = []

for a in alphas:
    ridge.set_params(alpha=a)
    ridge.fit(X, Y)
    coefs.append(ridge.coef_)

coefs = np.array(coefs)

# Tracer les coefficients en fonction de lambda
plt.plot(alphas, coefs)
plt.xlabel('Lambda')
plt.ylabel('Coefficients')
plt.title('Ridge Regression Coefficients')
plt.axhline(0, color='black', linestyle='--')
plt.legend(data.columns[:8], loc='upper right')
plt.show()