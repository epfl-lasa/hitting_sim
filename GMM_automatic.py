import numpy as np

import h5py
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import ellipse

from sklearn.model_selection import GridSearchCV
from scipy.stats import multivariate_normal
from scipy.stats import chi2

from gmr import GMM

import os
import time

from ds import linear_hitting_ds_pre_impact, linear_ds
from controller import get_joint_velocities_qp_dir_inertia_specific_NS, get_joint_velocities_qp
from get_robot import sim_robot_env
from iiwa_environment import object
from iiwa_environment import physics as phys
import functions as f


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
positions = hf['my_data']['box_pos'][:]        #Final position of box    (output)

data = np.concatenate((parameters,positions),axis=1)

# Visualize points
x = positions[:,0]
y = positions[:,1]

fig, ax = plt.subplots()
ax.scatter(x, y, s=20,  label ='Final position of box')


# Plot initial position of box
ax.scatter(0.5,0.3, s=100, marker='+',color='g', label ='Initial position of box')


t = grid_search.fit(data)               


# gmm = GMM(n_components=3, random_state=0)
# gmm.from_samples(data)

# REGRESSION
gmm = GMM(
   n_components=grid_search.best_estimator_.n_components, priors=grid_search.best_estimator_.weights_, 
   means=grid_search.best_estimator_.means_, covariances=np.array([c for c in grid_search.best_estimator_.covariances_]))


# Uncomment this code to predict the final position of the box given the hitting parameters
# theta_reg = 0
# p_des_reg = 0.9
# x_impact_reg = 0.5
# y_impact_reg = 0.0
# x1 = [[np.cos(theta_reg), np.sin(theta_reg), 0.0,p_des_reg, x_impact_reg, y_impact_reg]]  #Parameters
# x1_index = [0,1,2,3,4,5]                                                                  #Parameters

# Uncomment this code to predict the hitting parameters given the final position of the box
x_reg = 0.5 #0.49751953
y_reg = 1.0 #0.49648312
z_reg = 0.44922494
x1 = [[x_reg, y_reg, z_reg]]                      #Positions
x1_index = [6,7,8]                                #Positions

x2_predicted_mean = gmm.predict(x1_index, x1)
print(x2_predicted_mean)


#theta_act = np.arccos(x2_predicted_mean[0])
h_dir_act = x2_predicted_mean[0][:3]
p_des_act = x2_predicted_mean[0][3]
x_impact_act = x2_predicted_mean[0][4]
y_impact_act = x2_predicted_mean[0][5]
X_ref_act = [x_impact_act,0,0.5]


trailDuration = 0 # Make it 0 if you don't want the trail to end
contactTime = 0.5 # This is the time that the robot will be in contact with the box

################## GET THE ROBOT + ENVIRONMENT #########################
box = object.Box([0.2, 0.2, 0.2], 0.5)  # the box is a cube of size 20 cm, and it is 0.5 kg in mass
iiwa = sim_robot_env(1, box)

###################### INIT CONDITIONS #################################
X_init = [0.3, -0.2, 0.5]
q_init = iiwa.get_IK_joint_position(X_init) # Currently I am not changing any weights here
Lambda_init = iiwa.get_inertia_matrix_specific(q_init)

box_position_orientation = iiwa.get_box_position_orientation()
box_position_init = box_position_orientation[0]
box_orientation_init = box_position_orientation[1]

A = np.array([[-2, 0, 0], [0, -2, 0], [0, 0, -2]])




# initialize the robot and the box
iiwa.set_to_joint_position(q_init)
iiwa.reset_box(box_position_init, box_orientation_init)
lambda_dir = h_dir_act.T @ Lambda_init @ h_dir_act
is_hit = False

# take some time
time.sleep(1)    # !!!!

# initialise the time
time_init = time.time()

# Start the motion
while 1:
    X_qp = np.array(iiwa.get_ee_position())
    
    if not is_hit:
        dX = linear_hitting_ds_pre_impact(A, X_qp, X_ref_act, h_dir_act, p_des_act, lambda_dir, box.mass)
    else:
        dX = linear_ds(A, X_qp, X_ref_act)

    hit_dir = dX / np.linalg.norm(dX)

    lambda_current = iiwa.get_inertia_matrix()
    lambda_dir = hit_dir.T @ lambda_current @ hit_dir
    
    jac = np.array(iiwa.get_trans_jacobian())
    q_dot = get_joint_velocities_qp_dir_inertia_specific_NS(dX, jac, iiwa, hit_dir, 0.15, lambda_dir)
    
    iiwa.move_with_joint_velocities(q_dot)

    ## Need something more here later, this is contact detection and getting the contact point
    if(iiwa.get_collision_points().size != 0):
        is_hit = True
        iiwa.get_collision_position()

        
    iiwa.step()
    time_now = time.time()

    if((is_hit and iiwa.get_box_speed() < 0.001 and time_now - time_init > contactTime) or (time_now - time_init > 10)):
        box_pos = np.array(iiwa.get_box_position_orientation()[0])
        break




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

