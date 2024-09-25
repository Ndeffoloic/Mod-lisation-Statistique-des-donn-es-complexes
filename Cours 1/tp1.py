import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind

# Step 1: Simulate a vector of binary variable Y of size n = 50
n = 50
Y = np.random.choice([0, 1], size=n)

# Step 2: Simulate a matrix of quantitative covariates X of size n = 50 and p = 100
# ça signifie qu'il y a 100 covariables et chacune d'entre elles est de dimension (n,1). 
p = 100
X = np.random.normal(size=(n, p))

print("Voici X :",X)
print("Voici Y :",Y)
# Step 3: Perform the appropriate test to verify if one covariate is linked to Y
p_values = []

for i in range(p):
    # Séparer X basé sur les valeurs de Y
    X_when_Y_is_0 = X[Y == 0, i]
    X_when_Y_is_1 = X[Y == 1, i]
    
    # Appliquer le test t pour comparer les moyennes
    _, p_value = ttest_ind(X_when_Y_is_0, X_when_Y_is_1)
    p_values.append(p_value)

# Step 4: Find the min and max p-values and plot the histogram
min_p_value = min(p_values)
max_p_value = max(p_values)

plt.hist(p_values, bins=10)
plt.xlabel('p-values')
plt.ylabel('Frequency')
plt.title('Histogram of p-values')
plt.show()

# Step 5: Compute the number of False Positive Significant tests for a level alpha
alpha = 0.05
false_positives = sum(p_value < alpha for p_value in p_values)
print("The number of false positives is : ", false_positives)

# Step 6: Compute the empirical type I error and compare to alpha
empirical_type_I_error = false_positives / p
comparison = "less than" if empirical_type_I_error < alpha else "greater than or equal to"

print(f"The empirical type I error is {empirical_type_I_error:.2f}, which is {comparison} alpha.")

# Step 7: Repeat for different values of p
p_values_list = []
for p in [100, 1000, 10000]:
    X = np.random.normal(size=(n, p))
    p_values = []
    for i in range(p):
        _, p_value = ttest_ind(X[:, i], Y)
        p_values.append(p_value)
    p_values_list.append(p_values)
    
    alpha = 0.05
    false_positives = sum(p_value < alpha for p_value in p_values)
    
    empirical_type_I_error = false_positives / p
    print(f"The empirical_type_I_error is : {empirical_type_I_error:.2f}")
    
    plt.hist(p_values, bins=10)
    plt.xlabel('p-values')
    plt.ylabel('Frequency')
    plt.title('Histogram of p-values')
    plt.show()

