import numpy as np

import h5py
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import ellipse


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

# GMM
n_components=4

gmm = GaussianMixture(n_components,covariance_type='full', random_state=0).fit(data)

for i in range(n_components):
    mean = gmm.means_[i][6:8]
    covariance = gmm.covariances_[i][6:8,6:8]
    ellipse.plot_ellipse(mean,covariance,ax)

# x y axis name and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Poking reachable space')

ax.legend()
plt.show()


# print(gmm.bic(data))

hf.close()