import numpy as np
from scipy.optimize import minimize, basinhopping, Bounds
from scipy.stats import multivariate_normal
from scipy.integrate import nquad

from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import ellipse

# x[0] = x
# x[1] = y
# x[2] = theta_2
# x[3] = theta_1



def integral_intersection_area(x): #means1, covariances1, weights1, means2, covariances2, weights2):
    R1 = np.array([[np.cos(x[3]), -np.sin(x[3])],
                [np.sin(x[3]), np.cos(x[3])]])   
    
    R2 = np.array([[np.cos(x[2]), -np.sin(x[2])],
                [np.sin(x[2]), np.cos(x[2])]])
    
    new_means1=[]
    new_covariances1=[]
    new_means2=[]
    new_covariances2=[]
    for i in range(n_components):
        R1 = np.squeeze(R1)
        new_mean1 = R1 @ (means1[i]-P) + P
        new_covariance1 = R1 @ covariances1[i] @ R1.T
        new_means1.append(new_mean1)
        new_covariances1.append(new_covariance1)

        R2 = np.squeeze(R2)
        mean2 = means2[i] + (x[:2]-means2[0])
        new_mean2 = R2 @ (mean2-x[:2]) + x[:2] 
        new_covariance2 = R2 @ covariances2[i] @ R2.T
        new_means2.append(new_mean2)
        new_covariances2.append(new_covariance2)


    def integrand(x, y):
        return (weights1[0] * multivariate_normal.pdf([x, y], mean=new_means1[0], cov=new_covariances1[0]) + \
                weights1[1] * multivariate_normal.pdf([x, y], mean=new_means1[1], cov=new_covariances1[1])) * \
                (weights2[0] * multivariate_normal.pdf([x, y], mean=new_means2[0], cov=new_covariances2[0]) + \
                weights2[1] * multivariate_normal.pdf([x, y], mean=new_means2[1], cov=new_covariances2[1]))
    return nquad(integrand, [[-np.inf, np.inf], [-np.inf, np.inf]])[0]



def fun(x):
    f = 0
    R2 = np.array([[np.cos(x[2]), -np.sin(x[2])],
                [np.sin(x[2]), np.cos(x[2])]])
    for i in range(n_components):
        R2 = np.squeeze(R2)
        mean = means2[i] + (x[:2]-means2[0])
        new_mean = R2 @ (mean-x[:2]) + x[:2] 
        new_covariance = R2 @ covariances2[i] @ R2.T

        det_cov = np.linalg.det(new_covariance)
        constant = 1 / (2 * np.pi * np.sqrt(det_cov))

        # Calculate the exponent
        inv_cov = np.linalg.inv(new_covariance)
        exponent = -0.5 * (Xf - new_mean).T @ inv_cov @ (Xf - new_mean)

        # Calculate the probability density function
        pdf = weights2[i]*constant * np.exp(exponent)


        f = f + pdf
    return -f

def find_sol():#P,Xf,means2, covariances2,n_components):  
   #  can try different initial guesses by randomly 
   # generating values or using heuristic
#    if Xf[0]<P[0]:
#        if Xf[1]>P[1]:
#            guess = np.arctan((P[0]-Xf[0])/(Xf[1]-P[1]))
#        else:
#            guess = np.arctan((P[0]-Xf[0])/(Xf[1]-P[1])) + np.pi
#    else:
#         if Xf[1]<P[1]:
#            guess = np.arctan((P[0]-Xf[0])/(Xf[1]-P[1])) + np.pi  
#         else:
#             guess = np.arctan((Xf[1]-P[1])/(Xf[1]-P[1])) + (3/2)*np.pi   
#    print(guess)

   # I'll have to verify if guess is within constraints or not

   threshold = 0.1    
   
   #bnds = Bounds([0], [2*np.pi])
   cons = ({'type': 'ineq', 'fun': lambda x:  integral_intersection_area(x) - threshold})
           
   #cons = ({'type': 'ineq', 'fun': lambda theta:  theta},
   #        {'type': 'ineq', 'fun': lambda theta:  2*np.pi - theta})

   guess = [1.0,1.0,np.pi/2,0]
   result = minimize(fun, guess, constraints=cons, options={'maxiter': 1000}) # bounds= bnds) 
   # Adapt tol to avoid local minima

   
   #result = basinhopping(fun,0, niter=500) #changing niter to avoid local minimum
   return result.x

P = np.array([0.5,0.3])
Xf = [1.0,1.0]


n_components = 2

means1 = np.array([[0.47891806, 0.9040961], 
                  [0.50023427, 0.51117959]])


covariances1 = np.array([[[ 0.00438958, -0.00240083],
                        [-0.00240083,  0.02641489]],
                        [[ 0.0006874,  -0.0005142 ],
                        [-0.0005142,   0.00842463]]])

weights1 = np.array([0.27344326491,
                    0.7265567351])

means2 = np.array([[0.47891806, 0.9040961], 
                  [0.50023427, 0.51117959]])


covariances2 = np.array([[[ 0.00438958, -0.00240083],
                        [-0.00240083,  0.02641489]],
                        [[ 0.0006874,  -0.0005142 ],
                        [-0.0005142,   0.00842463]]])

weights2 = np.array([0.27344326491,
                    0.7265567351])




X = find_sol() #P,Xf,means2, covariances2,n_components)

print(X)

# RR = np.array([[np.cos(max_theta), -np.sin(max_theta)],
#             [np.sin(max_theta), np.cos(max_theta)]])
# RR = np.squeeze(RR)


# fig, ax = plt.subplots()

# # Plot initial position of box
# ax.scatter(P[0],P[1], s=100, marker='+',color='g', label ='Initial position of box')
# ax.scatter(Xf[0],Xf[1], s=100, marker='+',color='k', label ='Xf')

# for i in range(n_components):
#     ellipse.plot_ellipse(means2[i],covariances2[i],ax)

#     new_mean = RR @ (means2[i]-P) + P
#     new_covariance = RR @ covariances2[i] @ RR.T

#     ellipse.plot_ellipse(new_mean,new_covariance,ax)


# print("max_theta =", max_theta[0])

# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')

# ax.legend()
# plt.show()


