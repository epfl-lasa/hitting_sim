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
ax.scatter(x, y, s=20, cmap='viridis');


# Plot initial position of box
ax.plot(0.5,0.3,marker='+',color='g')

# GMM
n_components=5

gmm = GaussianMixture(n_components,covariance_type='full', random_state=0).fit(data)

for i in range(n_components):
    mean = gmm.means_[i][6:8]
    covariance = gmm.covariances_[i][6:8,6:8]
    ellipse.plot_ellipse(mean,covariance,ax)

plt.show()




hf.close()