import numpy as np
from scipy.optimize import minimize, basinhopping, Bounds

from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import ellipse


#WEIGHTS OF COMPONENTS!!!!!!

def fun(theta):
    f = 0
    R = np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]])
    for i in range(n_components):
        R = np.squeeze(R)
        new_mean = R @ (means[i]-P) + P
        new_covariance = R @ covariances[i] @ R.T

        det_cov = np.linalg.det(new_covariance)
        constant = 1 / (2 * np.pi * np.sqrt(det_cov))

        # Calculate the exponent
        inv_cov = np.linalg.inv(new_covariance)
        exponent = -0.5 * (Xm - new_mean).T @ inv_cov @ (Xm - new_mean)

        # Calculate the probability density function
        pdf = constant * np.exp(exponent)


        f = f + pdf
    return -f

def find_theta(P,Xm,means, covariances,n_components):  
   #  can try different initial guesses by randomly 
   # generating values or using heuristic
   if Xm[0]<P[0]:
       if Xm[1]>P[1]:
           guess = np.arctan((P[0]-Xm[0])/(Xm[1]-P[1]))
       else:
           guess = np.arctan((P[0]-Xm[0])/(Xm[1]-P[1])) + np.pi
   else:
        if Xm[1]<P[1]:
           guess = np.arctan((P[0]-Xm[0])/(Xm[1]-P[1])) + np.pi  
        else:
            guess = np.arctan((Xm[1]-P[1])/(Xm[1]-P[1])) + (3/2)*np.pi   
   print(guess)

   # I'll have to verify if guess is within constraints or not
       
   
   bnds = Bounds([0], [2*np.pi])
   #cons = ({'type': 'ineq', 'fun': lambda theta:  theta},
   #        {'type': 'ineq', 'fun': lambda theta:  2*np.pi - theta})
   result = minimize(fun,guess, method='Nelder-Mead', bounds= bnds) 
   # Adapt tol to avoid local minima

   
   #result = basinhopping(fun,0, niter=500) #changing niter to avoid local minimum
   return result.x

P = np.array([0.5,0.3])
Xm = [0.0,0.0]


n_components = 2

means = np.array([[0.47891806, 0.9040961], 
                  [0.50023427, 0.51117959]])


covariances = np.array([[[ 0.00438958, -0.00240083],
                        [-0.00240083,  0.02641489]],
                        [[ 0.0006874,  -0.0005142 ],
                        [-0.0005142,   0.00842463]]])




max_theta = find_theta(P,Xm,means, covariances,n_components)

RR = np.array([[np.cos(max_theta), -np.sin(max_theta)],
            [np.sin(max_theta), np.cos(max_theta)]])
RR = np.squeeze(RR)


fig, ax = plt.subplots()

# Plot initial position of box
ax.scatter(P[0],P[1], s=100, marker='+',color='g', label ='Initial position of box')
ax.scatter(Xm[0],Xm[1], s=100, marker='+',color='k', label ='Xm')

for i in range(n_components):
    ellipse.plot_ellipse(means[i],covariances[i],ax)

    new_mean = RR @ (means[i]-P) + P
    new_covariance = RR @ covariances[i] @ RR.T

    ellipse.plot_ellipse(new_mean,new_covariance,ax)


print("max_theta =", max_theta[0])

plt.xlabel('X-axis')
plt.ylabel('Y-axis')

ax.legend()
plt.show()


