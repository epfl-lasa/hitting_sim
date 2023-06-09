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
        
    return grid_search.best_estimator_.n_components, grid_search.best_estimator_.means_,\
          grid_search.best_estimator_.covariances_, grid_search.best_estimator_.weights_


def plot_data(x, y, n_components, means, covariances):

    means_2d = []
    covariances_2d = []
    weights_2d = []
    for i in range(n_components):
        mean = means[i][4:6]
        covariance = covariances[i][4:6, 4:6]
        weight = weights[i]

        means_2d.append(mean)
        covariances_2d.append(covariance)
        weights_2d.append(weight)


    fig, ax = plt.subplots()
    ax.scatter(x, y, s=20,  label ='Final position of box')

    # Plot initial position of box
    ax.scatter(0.5,0.3, s=100, marker='+',color='g', label ='Initial position of box')

    for i in range(n_components):
        ellipse.plot_ellipse(means_2d[i],covariances_2d[i],ax)

    # x y axis name and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Object reachable space')

    ax.legend()
    plt.show()


def write_model_data(model_data_path,n_components, means, covariances, weights):
    # Open the HDF5 file in write mode
    hf_1 = h5py.File(model_data_path[0], 'a')
    hf_2 = h5py.File(model_data_path[1], 'a')

    # Write the parameters
    hf_1.create_dataset('n_components', data=n_components)
    hf_1.create_dataset('means', data=means)
    hf_1.create_dataset('covariances', data=covariances)
    hf_1.create_dataset('weights', data=weights)
    

    means_2d = []
    covariances_2d = []
    weights_2d = []
    for i in range(n_components):
        mean = means[i][4:6]
        covariance = covariances[i][4:6, 4:6]
        weight = weights[i]

        means_2d.append(mean)
        covariances_2d.append(covariance)
        weights_2d.append(weight)

    hf_2.create_dataset('n_components', data=n_components)
    hf_2.create_dataset('means', data=means_2d)
    hf_2.create_dataset('covariances', data=covariances_2d)
    hf_2.create_dataset('weights', data=weights_2d)

    # Close the HDF5 file
    hf_1.close()
    hf_2.close()





data_path = 'Data/data_no_table_2.h5'
model_data_paths = ['Data/model_no_table_full.h5', 'Data/model_no_table_2d.h5']
data, x, y =  read_data(data_path)
n_components, means, covariances, weights = fit_data(data)
plot_data(x, y, n_components, means, covariances)
write_model_data(model_data_paths,n_components, means, covariances, weights)


print("n_components",n_components)
print("means", means)
print("covariances", covariances)
print("weights", weights)




