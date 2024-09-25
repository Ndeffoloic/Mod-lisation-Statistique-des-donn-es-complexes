import os

import pandas as pd

# Nom du fichier à ouvrir (assurez-vous de remplacer 'nom_du_fichier.txt' par le nom réel de votre fichier)
file_name = 'poumon.txt'

# Obtenir le chemin complet du fichier
file_path = os.path.join(os.path.dirname(__file__), file_name)

# Lire le fichier dans un DataFrame avec les bons séparateurs
data = pd.read_csv(file_path, sep='\t', decimal='.')

# Afficher les données pour vérifier qu'elles sont bien importées
print(data.head())