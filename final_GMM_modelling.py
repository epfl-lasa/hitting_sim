import numpy as np

import h5py
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import ellipse

from sklearn.model_selection import GridSearchCV

from gmr import GMM



def read_data(data_path):
    hf = h5py.File(data_path, 'r')

    # Syntax
    parameters = hf['my_data']['params'][:]         #Hitting parameters        (input)
    positions = hf['my_data']['box_pos'][:]        #Final position of box    (output)

    data = np.concatenate((parameters,positions),axis=1)

    # Visualize points
    x = positions[:,0]
    y = positions[:,1]
    
    hf.close()

    return data, x, y


def fit_data(data):

    def gmm_bic_score(estimator, X):
        """Callable to pass to GridSearchCV that will use the BIC score."""
        # Make it negative since GridSearchCV expects a score to maximize
        return -estimator.bic(X)

    param_grid = {
        "n_components": range(2, 3),
        "covariance_type": ["full"],        # We can put this"covariance_type": ["spherical", "tied", "diag", "full"]
    }                                       # but shape of grid_search.best_estimator_.covariances_ changes 
    grid_search = GridSearchCV(
        GaussianMixture(), param_grid=param_grid, scoring=gmm_bic_score  #initialization in GaussianMixture
    )

    t = grid_search.fit(data)   
    n_components = grid_search.best_estimator_.n_components

    means = []
    covariances = []
    weights = []
    for i in range(n_components):
        mean = grid_search.best_estimator_.means_[i][4:6]
        covariance = grid_search.best_estimator_.covariances_[i][4:6, 4:6]
        weight = grid_search.best_estimator_.weights_[i]

        means.append(mean)
        covariances.append(covariance)
        weights.append(weight)
        
    return n_components, means, covariances, weights


def plot_data(x, y, n_components, means, covariances):
    fig, ax = plt.subplots()
    ax.scatter(x, y, s=20,  label ='Final position of box')

    # Plot initial position of box
    ax.scatter(0.5,0.3, s=100, marker='+',color='g', label ='Initial position of box')

    for i in range(n_components):
        ellipse.plot_ellipse(means[i],covariances[i],ax)

    # x y axis name and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Object reachable space')

    ax.legend()
    plt.show()


def write_model_data(model_data_path,n_components, means, covariances, weights):
    # Open the HDF5 file in write mode
    hf = h5py.File(model_data_path, 'a')
    # Write the parameters
    hf.create_dataset('n_components', data=n_components)
    hf.create_dataset('means', data=means)
    hf.create_dataset('covariances', data=covariances)
    hf.create_dataset('weights', data=weights)

    # Close the HDF5 file
    hf.close()




data_path = 'Data/data_pres.h5'
model_data_path = 'Data/model.h5'
data, x, y =  read_data(data_path)
n_components, means, covariances, weights = fit_data(data)
#plot_data(x, y, n_components, means, covariances)
write_model_data(model_data_path,n_components, means, covariances, weights)


print("n_components",n_components)
print("means", means)
print("covariances", covariances)
print("weights", weights)




