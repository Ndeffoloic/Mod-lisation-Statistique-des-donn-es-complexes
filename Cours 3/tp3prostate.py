import os

import pandas as pd

# Nom du fichier à ouvrir
file_name = 'prostate.txt'

# Obtenir le chemin complet du fichier
file_path = os.path.join(os.path.dirname(__file__), file_name)

# Lire le fichier dans un DataFrame avec les bons séparateurs
data = pd.read_csv(file_path, sep=' ', header=True, decimal=' ')
