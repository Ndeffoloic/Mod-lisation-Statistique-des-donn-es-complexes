from tkinter import Tk
from tkinter.filedialog import askopenfilename

import pandas as pd

# Ouvrir une boîte de dialogue pour choisir le fichier
Tk().withdraw()  # Empêche l'apparition de la fenêtre principale de Tkinter
file_path = askopenfilename()

# Lire le fichier dans un DataFrame avec les bons séparateurs
data = pd.read_csv(file_path, sep='\t', decimal='.')

# Afficher les données pour vérifier qu'elles sont bien importées
print(data.head())