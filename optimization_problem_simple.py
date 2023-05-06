import numpy as np
from scipy.optimize import minimize, basinhopping, Bounds
from scipy.stats import multivariate_normal
from scipy.integrate import nquad, dblquad

from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import ellipse

# x[0] = x
# x[1] = y
# x[2] = theta_2
# x[3] = theta_1

# def area(x):
#     def integrand(xx, yy):
#         eigenvalues1, eigenvectors1 = np.linalg.eigh(covariances1)
#         width1, height1 = 2*np.sqrt(eigenvalues1)                      #np.abs?
#         rotation1 = np.degrees(np.arctan2(*eigenvectors1[::-1, 0]))

#         eigenvalues2, eigenvectors2 = np.linalg.eigh(covariances2)
#         width2, height2 = 2*np.sqrt(eigenvalues2)                      #np.abs?
#         rotation2 = np.degrees(np.arctan2(*eigenvectors2[::-1, 0]))
#         return ((xx - 0) / width1)**2 + ((yy - 0) / height1)**2 - 1 <= 0 and ((xx - x[0]) / width2)**2 + ((yy - x[1]) / height2)**2 - 1 <= 0
# # Calculate the area of intersection using double integration
#     return dblquad(lambda x, y: integrand(x, y), -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf)[0]

# def area_of_intersection(x):
#     # x is the array of variables to be optimized
#     # a1, b1, x1, y1, a2, b2, x2, y2 are the parameters that define the two ellipses
#     # threshold is the minimum area of intersection required
#     eigenvalues1, eigenvectors1 = np.linalg.eigh(covariances1)
#     width1, height1 = 2*np.sqrt(eigenvalues1)                      #np.abs?
#     rotation1 = np.degrees(np.arctan2(*eigenvectors1[::-1, 0]))

#     eigenvalues2, eigenvectors2 = np.linalg.eigh(covariances2)
#     width2, height2 = 2*np.sqrt(eigenvalues2)                      #np.abs?
#     rotation2 = np.degrees(np.arctan2(*eigenvectors2[::-1, 0]))
    
#     # Define the equations of the two ellipses
#     def ellipse1(xx, yy):
#         return ((xx-0)/width1)**2 + ((yy-0)/height1)**2 - 1
    
#     def ellipse2(xx, yy):
#         return ((xx-x[0])/width2)**2 + ((yy-x[1])/height2)**2 - 1
    
#     # Define the function to be integrated
#     def integrand(xx, yy):
#         return np.minimum(ellipse1(xx, yy), ellipse2(xx, yy))
    
#     # Calculate the area of intersection using nquad integration
#     area, _ = nquad(integrand, [[-np.inf, np.inf], [-np.inf, np.inf]])
    
#     # Return the difference between the calculated area and the threshold
#     return area


def integral_intersection_area(x): #means1, covariances1, weights1, means2, covariances2, weights2):
    R1 = np.array([[np.cos(x[3]), -np.sin(x[3])],
                [np.sin(x[3]), np.cos(x[3])]])   
    
    R2 = np.array([[np.cos(x[2]), -np.sin(x[2])],
                [np.sin(x[2]), np.cos(x[2])]])
    
    new_means1=[]
    new_covariances1=[]
    new_means2=[]
    new_covariances2=[]
    R1 = np.squeeze(R1)
    new_mean1 = R1 @ (means1-P) + P
    new_covariance1 = R1 @ covariances1 @ R1.T
    new_means1.append(new_mean1)
    new_covariances1.append(new_covariance1)

    R2 = np.squeeze(R2)
    mean2 = means2 + (x[:2]-means2)
    new_mean2 = R2 @ (mean2-x[:2]) + x[:2] 
    new_covariance2 = R2 @ covariances2 @ R2.T
    new_means2.append(new_mean2)
    new_covariances2.append(new_covariance2)


    def integrand(xx, yy):
        return (multivariate_normal.pdf([xx, yy], mean=new_mean1, cov=new_covariance1)) * \
                ( multivariate_normal.pdf([xx, yy], mean=new_mean2, cov=new_covariance2))
    return nquad(integrand, [[-np.inf, np.inf], [-np.inf, np.inf]])[0]



def fun(x):
    f = 0
    R2 = np.array([[np.cos(x[2]), -np.sin(x[2])],
                [np.sin(x[2]), np.cos(x[2])]])
    R2 = np.squeeze(R2)
    mean = means2 + (x[:2]-means2)
    new_mean = R2 @ (mean-x[:2]) + x[:2] 
    new_covariance = R2 @ covariances2 @ R2.T

    det_cov = np.linalg.det(new_covariance)
    constant = 1 / (2 * np.pi * np.sqrt(det_cov))

    # Calculate the exponent
    inv_cov = np.linalg.inv(new_covariance)
    exponent = -0.5 * (Xf - new_mean).T @ inv_cov @ (Xf - new_mean)

    # Calculate the probability density function
    pdf = constant * np.exp(exponent)


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

   threshold = 0.0   
   
   #bnds = Bounds([0], [2*np.pi])
   cons = ({'type': 'ineq', 'fun': lambda x:  integral_intersection_area(x) - threshold})
           
   #cons = ({'type': 'ineq', 'fun': lambda theta:  theta},
   #        {'type': 'ineq', 'fun': lambda theta:  2*np.pi - theta})

   guess = [0.0,0.0,0*np.pi/2,0]
   result = minimize(fun, guess, constraints=cons, options={'maxiter': 2}) # bounds= bnds) 
   # Adapt tol to avoid local minima

   
   #result = basinhopping(fun,0, niter=500) #changing niter to avoid local minimum
   return result.x

P = np.array([0.0,0.0])
Xf = [0.1,0.1]


n_components = 1

means1 = np.array([0.0, 0.0])


covariances1 = np.array([[ 1.0, 0],
                        [0,  1.0]])

weights1 = 1.0

means2 = np.array([0.0, 0.0])


covariances2 = np.array([[ 1.0, 0],
                        [0,  1.0]])

weights2 = 1.0




X = find_sol() #P,Xf,means2, covariances2,n_components)

X[2]=np.rad2deg(X[2])
print(X)

#print(integral_intersection_area([0.0,0.0,0.0,0.0]))

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


