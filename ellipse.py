import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def plot_ellipse(mean, covariance,ax):

    # mean vector and covariance matrix
    
    # generate points on the ellipse boundary
    theta = np.linspace(0, 2*np.pi, 100)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    width, height = 2*np.sqrt(eigenvalues)
    rotation = np.degrees(np.arctan2(*eigenvectors[::-1, 0]))
    
    # plot the ellipse
    ellipse1 = Ellipse(xy=mean, width=width, height=height, angle=rotation, edgecolor='r', fc='None', lw=2)
    ax.add_patch(ellipse1)
    ellipse2 = Ellipse(xy=mean, width=2*width, height=2*height, angle=rotation, edgecolor='b', fc='None', lw=2)
    ax.add_patch(ellipse2)
    ax.scatter(mean[0], mean[1], marker='+', s=100, color='r')
    plt.axis('equal')
