# Simulation des données
set.seed(123)  # Pour rendre les résultats reproductibles
n <- 100  # Nombre d'observations

X <- rnorm(n, mean = 0, sd = 1)  # Variable X simulée
R <- rbinom(n, 1, 0.7)  # Indicateur de réponse avec probabilité 0.7 de non-manquant

# Crée des valeurs manquantes
X[R == 0] <- NA

