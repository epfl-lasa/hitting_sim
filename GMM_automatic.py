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

    print("maha")
    print(mahalanobis_dist)
    print("thresh")
    print(threshold)


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
    GaussianMixture(), param_grid=param_grid, scoring=gmm_bic_score  #initialization in GaussianMixture
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
point = [0.3, 0.8]
p2 = [0.5, 1.1]

inn = 0
in2 = 0
gaussians = []
means = []
covariances = []
weights = []
for i in range(n_components):
    mean = grid_search.best_estimator_.means_[i][6:8]
    covariance = grid_search.best_estimator_.covariances_[i][6:8,6:8]
    weight = grid_search.best_estimator_.weights_[i]

    means.append(mean)
    covariances.append(covariance)
    weights.append(weight)

    ellipse.plot_ellipse(mean,covariance,ax)
    # create Gaussian distributions
    gaussians.append(multivariate_normal(mean=mean, cov=covariance))
    # Check if point is within 1-sigma
    if is_within_sigma(point, mean, covariance,2): # <2 for 2-sigma
        inn = 1
    if is_within_sigma(p2, mean, covariance,2): # <2 for 2-sigma
        in2 = 1

print("means", means)
print("covariances", covariances)
print("weights", weights)


if inn:
    print("Point is inside the reachable space")
else:
    print("Point is outside the reachable space")

inn = 0

if in2:
    print("Point is inside the reachable space")
else:
    print("Point is outside the reachable space")

in2 = 0

# # Calculate sum of PDFs at a point
pdf_value_sum = 0
for gaussian in gaussians:
    pdf_value_sum += weights[i]*gaussian.pdf(point)





# Check for best angle theta to reach point Xm
Xm = [0.0,0.1]
ax.scatter(Xm[0],Xm[1], s=100, marker='+',color='b', label ='Target final position')


max_prob = 0
max_theta = 0

# Sample theta from 0 to 2pi
# Rotate mean and covariance
for theta in np.linspace(0, 2*np.pi, num = 100):
    # Rotate around point (0.5,0.3) different than origin
    P = np.array([0.5,0.3])

    T1 = np.eye(3)
    T1[:2, 2] = -P

    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]])

    T2 = np.eye(3)
    T2[:2, 2] = P

    Transf = T2 @ R @ T1
    RR = R[:2,:2]


    new_gaussians = []
    for i in range(n_components):
        mean = np.ones(3)
        mean[:2] = means[i]
        
        new_mean = Transf @ mean
        new_mean = new_mean[:2]

        covariance =  covariances[i]
        new_covariance = RR @ covariance @ RR.T

        new_gaussians.append(multivariate_normal(mean=new_mean, cov=new_covariance))



    pdf_value_sum = 0
    for gaussian in new_gaussians:
        pdf_value_sum += weights[i]*gaussian.pdf(Xm)

    if pdf_value_sum > max_prob:
        max_prob = pdf_value_sum
        max_theta = theta


T1 = np.eye(3)
T1[:2, 2] = -P

R = np.array([[np.cos(max_theta), -np.sin(max_theta), 0],
            [np.sin(max_theta), np.cos(max_theta), 0],
            [0, 0, 1]])

T2 = np.eye(3)
T2[:2, 2] = P

Transf = T2 @ R @ T1
RR = R[:2,:2]


new_gaussians = []
for i in range(n_components):
    mean = np.ones(3)
    mean[:2] = means[i]
    
    new_mean = Transf @ mean
    new_mean = new_mean[:2]

    covariance =  covariances[i]
    new_covariance = RR @ covariance @ RR.T
    ellipse.plot_ellipse(new_mean,new_covariance,ax)


print(max_theta)

# x y axis name and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Object reachable space')

ax.legend()
plt.show()





hf.close()

