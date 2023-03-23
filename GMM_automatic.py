import numpy as np

import h5py
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import ellipse

from sklearn.model_selection import GridSearchCV

def gmm_bic_score(estimator, X):
    """Callable to pass to GridSearchCV that will use the BIC score."""
    # Make it negative since GridSearchCV expects a score to maximize
    return -estimator.bic(X)


param_grid = {
    "n_components": range(1, 7),
    "covariance_type": ["full"],        # We can put this"covariance_type": ["spherical", "tied", "diag", "full"]
}                                       # but shape of grid_search.best_estimator_.covariances_ changes 
grid_search = GridSearchCV(
    GaussianMixture(), param_grid=param_grid, scoring=gmm_bic_score
)


hf = h5py.File('Data/data3.h5', 'r')

# Syntax
parameters = hf['my_data']['params'][:]         #Hitting parameters        (input)
positions = hf['my_data']['box_pos'][:]        #Fibnal position of box    (output)

data = np.concatenate((parameters,positions),axis=1)

# Visualize points
x = positions[:,0]
y = positions[:,1]

fig, ax = plt.subplots()
ax.scatter(x, y, s=20, cmap='viridis', label ='Final position of box')


# Plot initial position of box
ax.scatter(0.5,0.3, s=100, marker='+',color='g', label ='Initial position of box')


t = grid_search.fit(data)
n_components = grid_search.best_estimator_.n_components

# x y axis name and title

for i in range(n_components):
    mean = grid_search.best_estimator_.means_[i][6:8]
    covariance = grid_search.best_estimator_.covariances_[i][6:8,6:8]
    ellipse.plot_ellipse(mean,covariance,ax)


plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Poking reachable space')

ax.legend()
plt.show()


hf.close()