import numpy as np

import h5py
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import ellipse

from sklearn.model_selection import GridSearchCV
from scipy.stats import multivariate_normal
from scipy.stats import chi2


def is_within_sigma(point, mean, covariance,nb_sigma=2):
    # Compute the Mahalanobis distance between the point and the mean vector
    delta = point - mean
    mahalanobis_dist = np.sqrt(delta.T @ np.linalg.inv(covariance) @ delta)
    
    # Calculate the threshold value for 2 sigma
    k = covariance.shape[0] # number of dimensions
    if nb_sigma==1:
        threshold = np.sqrt(chi2.ppf(0.68, k))
    elif nb_sigma==2:
        threshold = np.sqrt(chi2.ppf(0.95, k))

    # Compare the Mahalanobis distance to the threshold value
    if mahalanobis_dist <= threshold:
        return True
    else:
        return False
    

def gmm_bic_score(estimator, X):
    """Callable to pass to GridSearchCV that will use the BIC score."""
    # Make it negative since GridSearchCV expects a score to maximize
    return -estimator.bic(X)


param_grid = {
    "n_components": range(1, 10),
    "covariance_type": ["full"],        # We can put this"covariance_type": ["spherical", "tied", "diag", "full"]
}                                       # but shape of grid_search.best_estimator_.covariances_ changes 
grid_search = GridSearchCV(
    GaussianMixture(), param_grid=param_grid, scoring=gmm_bic_score
)


hf = h5py.File('Data/data_good.h5', 'r')

# Syntax
parameters = hf['my_data']['params'][:]         #Hitting parameters        (input)
positions = hf['my_data']['box_pos'][:]        #Fibnal position of box    (output)

data = np.concatenate((parameters,positions),axis=1)

# Visualize points
x = positions[:,0]
y = positions[:,1]

fig, ax = plt.subplots()
ax.scatter(x, y, s=20,  label ='Final position of box')


# Plot initial position of box
ax.scatter(0.5,0.3, s=100, marker='+',color='g', label ='Initial position of box')



t = grid_search.fit(data)               

n_components = grid_search.best_estimator_.n_components

#  #!!! Maybe instead of grid_search.fit(data) use this with initialization parameter
#gmm = GaussianMixture(n_components,covariance_type='full', random_state=0, init_params="kmeans").fit(data)

# Define specific point
point = [0.0, 0.0]


inn = 0
gaussians = []
for i in range(n_components):
    mean = grid_search.best_estimator_.means_[i][6:8]
    covariance = grid_search.best_estimator_.covariances_[i][6:8,6:8]
    ellipse.plot_ellipse(mean,covariance,ax)
    # create Gaussian distributions
    gaussians.append(multivariate_normal(mean=mean, cov=covariance))

    # Check if point is within 1-sigma
    if is_within_sigma(point, mean, covariance,2): # <2 for 2-sigma
        inn = 1


if inn:
    print("Point is inside the reachable space")
else:
    print("Point is outside the reachable space")

inn = 0

# Calculate sum of PDFs at a point
pdf_value_sum = 0
for gaussian in gaussians:
    pdf_value_sum += gaussian.pdf(point)

#pdf_value = pdf_value_sum/n_components
#print(pdf_value)

# Decide if point is within reachable space or not (maybe if within 1-sigma)
# Mahalanobis distance
#md = np.sqrt(np.sum(np.dot((point - mean), np.linalg.inv(covariance)) * (point - mean), axis=0))
#print(md)


# x y axis name and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Poking reachable space')

ax.legend()
plt.show()


hf.close()

