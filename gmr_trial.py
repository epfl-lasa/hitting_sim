
import numpy as np
from gmr import GMM

# Your dataset as a NumPy array of shape (n_samples, n_features):
X = np.random.randn(100, 3)

gmm = GMM(n_components=3, random_state=0)
gmm.from_samples(X)

# Estimate GMM with expectation maximization:
X_sampled = gmm.sample(100)

# Make predictions with known values for the first feature:
# x1 = np.random.randn(20, 1)
# print(x1)
x1 = [[5,4]]
x1_index = [0,1]
x2_predicted_mean = gmm.predict(x1_index, x1)
print(x2_predicted_mean)