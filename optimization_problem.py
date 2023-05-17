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
    dets_cov1 = []
    new_means2=[]
    new_covariances2=[]
    dets_cov2 = []    
    for i in range(n_components):
        R1 = np.squeeze(R1)
        mean1 = means1[i] + (P-box1)
        new_mean1 = R1 @ (mean1-P) + P
        new_covariance1 = R1 @ covariances1[i] @ R1.T
        det_cov1 = np.linalg.det(new_covariance1)
        new_means1.append(new_mean1)
        new_covariances1.append(new_covariance1)
        dets_cov1.append(det_cov1)

        R2 = np.squeeze(R2)
        mean2 = means2[i] + (x[:2]-box2)
        new_mean2 = R2 @ (mean2-x[:2]) + x[:2] 
        new_covariance2 = R2 @ covariances2[i] @ R2.T
        det_cov2 = np.linalg.det(new_covariance2)
        new_means2.append(new_mean2)
        new_covariances2.append(new_covariance2)
        dets_cov2.append(det_cov2)

    def GMM_product(xx, yy):
        return (weights1[0] *(2 * np.pi * np.sqrt(dets_cov1[0]))* multivariate_normal.pdf([xx, yy], mean=new_means1[0], cov=new_covariances1[0]) + \
                weights1[1] *(2 * np.pi * np.sqrt(dets_cov1[1]))* multivariate_normal.pdf([xx, yy], mean=new_means1[1], cov=new_covariances1[1])) * \
                (weights2[0]*(2 * np.pi * np.sqrt(dets_cov2[0]))* multivariate_normal.pdf([xx, yy], mean=new_means2[0], cov=new_covariances2[0]) + \
                weights2[1] *(2 * np.pi * np.sqrt(dets_cov2[1]))* multivariate_normal.pdf([xx, yy], mean=new_means2[1], cov=new_covariances2[1]))

    # def GMM_product(xx, yy):
    #     return (multivariate_normal.pdf([xx, yy], mean=new_means1[0], cov=new_covariances1[0]) + \
    #             multivariate_normal.pdf([xx, yy], mean=new_means1[1], cov=new_covariances1[1])) * \
    #             (multivariate_normal.pdf([xx, yy], mean=new_means2[0], cov=new_covariances2[0]) + \
    #             multivariate_normal.pdf([xx, yy], mean=new_means2[1], cov=new_covariances2[1]))
    return GMM_product(x[0],x[1])

# not integrate but evaluate the product at position of box at GMM2 (or just take max of product)



# Plot pdf values with surface plot or color map

def fun(x):
    f = 0
    R2 = np.array([[np.cos(x[2]), -np.sin(x[2])],
                [np.sin(x[2]), np.cos(x[2])]])
    for i in range(n_components):
        R2 = np.squeeze(R2)
        mean = means2[i] + (x[:2]-box2)
        new_mean = R2 @ (mean-x[:2]) + x[:2] 
        new_covariance = R2 @ covariances2[i] @ R2.T

        det_cov = np.linalg.det(new_covariance)
        #constant = 1 / (2 * np.pi * np.sqrt(det_cov))

        # To have same scale for gaussians
        constant = 1

        # Calculate the exponent
        inv_cov = np.linalg.inv(new_covariance)
        exponent = -0.5 * (Xf - new_mean).T @ inv_cov @ (Xf - new_mean)

        # Calculate the probability density function
        pdf = weights2[i]* constant * np.exp(exponent)
        #pdf = constant * np.exp(exponent)

        f = f + pdf
    return -f


margin = 0.0

# Table always on Table
def cons_1(x, alpha, Xf, x_limits):
    return (1 - alpha) * x[0] + alpha * Xf[0] - (x_limits[0] + margin)

def cons_2(x, alpha, Xf, x_limits, y_limits):
    if ((1 - alpha) * x[1] + alpha * Xf[1]) < y_limits[1]:
        return -((1 - alpha) * x[0] + alpha * Xf[0]) + (x_limits[1] - margin)
    else:
        return -((1 - alpha) * x[0] + alpha * Xf[0]) + (x_limits[2] - margin)
    
def cons_3(x, alpha, Xf, y_limits):
    return -((1 - alpha) * x[1] + alpha * Xf[1]) + (y_limits[2] - margin)   #-x[1]+y_limits[2]

def cons_4(x, alpha, Xf, x_limits, y_limits):   
    if ((1 - alpha) * x[0] + alpha * Xf[0]) < x_limits[1]:  #x[0]<x_limits[1]
        return ((1 - alpha) * x[1] + alpha * Xf[1]) - (y_limits[0] + margin)
    else:
        return ((1 - alpha) * x[1] + alpha * Xf[1]) - (y_limits[1] + margin)


def find_sol(environment,x_limits,y_limits, guess, intersection_threshold):#P,Xf,means2, covariances2,n_components):  
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
    #bnds = Bounds([0], [2*np.pi])


    if environment:
        # x_sampled = (1-alpha)*x[:2]+alpha*Xf
        cons = ({'type': 'ineq', 'fun': lambda x:  integral_intersection_area(x) - intersection_threshold},
                {'type': 'ineq', 'fun': lambda x:  x[0]-x_limits[0] },
                {'type': 'ineq', 'fun': lambda x:  -x[0]+x_limits[1] if (x[1]<y_limits[1]) else -x[0]+x_limits[2]}, #-x[0]+x_limits[2]
                {'type': 'ineq', 'fun': lambda x:  x[1]-y_limits[0] if(x[0]<x_limits[1]) else x[1]-y_limits[1]},
                {'type': 'ineq', 'fun': lambda x:  -x[1]+y_limits[2]},
                {'type': 'ineq', 'fun': lambda x:  -x[2]+np.pi},
                {'type': 'ineq', 'fun': lambda x:  x[2]+np.pi},

                {'type': 'ineq', 'fun': lambda x: cons_1(x, 0.1, Xf, x_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_2(x, 0.1, Xf, x_limits, y_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_1(x, 0.2, Xf, x_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_2(x, 0.2, Xf, x_limits, y_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_1(x, 0.3, Xf, x_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_2(x, 0.3, Xf, x_limits, y_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_1(x, 0.4, Xf, x_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_2(x, 0.4, Xf, x_limits, y_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_1(x, 0.5, Xf, x_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_2(x, 0.5, Xf, x_limits, y_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_1(x, 0.6, Xf, x_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_2(x, 0.6, Xf, x_limits, y_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_1(x, 0.7, Xf, x_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_2(x, 0.7, Xf, x_limits, y_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_1(x, 0.8, Xf, x_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_2(x, 0.8, Xf, x_limits, y_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_1(x, 0.9, Xf, x_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_2(x, 0.9, Xf, x_limits, y_limits)},

                {'type': 'ineq', 'fun': lambda x: cons_3(x, 0.1, Xf, x_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_4(x, 0.1, Xf, x_limits, y_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_3(x, 0.2, Xf, x_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_4(x, 0.2, Xf, x_limits, y_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_3(x, 0.3, Xf, x_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_4(x, 0.3, Xf, x_limits, y_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_3(x, 0.4, Xf, x_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_4(x, 0.4, Xf, x_limits, y_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_3(x, 0.5, Xf, x_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_4(x, 0.5, Xf, x_limits, y_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_3(x, 0.6, Xf, x_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_4(x, 0.6, Xf, x_limits, y_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_3(x, 0.7, Xf, x_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_4(x, 0.7, Xf, x_limits, y_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_3(x, 0.8, Xf, x_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_4(x, 0.8, Xf, x_limits, y_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_3(x, 0.9, Xf, x_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_4(x, 0.9, Xf, x_limits, y_limits)},

                {'type': 'ineq', 'fun': lambda x: cons_1(x, 0.1, P, x_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_2(x, 0.1, P, x_limits, y_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_1(x, 0.2, P, x_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_2(x, 0.2, P, x_limits, y_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_1(x, 0.3, P, x_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_2(x, 0.3, P, x_limits, y_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_1(x, 0.4, P, x_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_2(x, 0.4, P, x_limits, y_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_1(x, 0.5, P, x_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_2(x, 0.5, P, x_limits, y_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_1(x, 0.6, P, x_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_2(x, 0.6, P, x_limits, y_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_1(x, 0.7, P, x_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_2(x, 0.7, P, x_limits, y_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_1(x, 0.8, P, x_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_2(x, 0.8, P, x_limits, y_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_1(x, 0.9, P, x_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_2(x, 0.9, P, x_limits, y_limits)},

                {'type': 'ineq', 'fun': lambda x: cons_3(x, 0.1, P, x_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_4(x, 0.1, P, x_limits, y_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_3(x, 0.2, P, x_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_4(x, 0.2, P, x_limits, y_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_3(x, 0.3, P, x_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_4(x, 0.3, P, x_limits, y_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_3(x, 0.4, P, x_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_4(x, 0.4, P, x_limits, y_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_3(x, 0.5, P, x_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_4(x, 0.5, P, x_limits, y_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_3(x, 0.6, P, x_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_4(x, 0.6, P, x_limits, y_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_3(x, 0.7, P, x_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_4(x, 0.7, P, x_limits, y_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_3(x, 0.8, P, x_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_4(x, 0.8, P, x_limits, y_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_3(x, 0.9, P, x_limits)},
                {'type': 'ineq', 'fun': lambda x: cons_4(x, 0.9, P, x_limits, y_limits)})
                #{'type': 'ineq', 'fun': lambda x: x[2] - (np.arctan2(Xf[1]-x[1],Xf[0]-x[0])-1.0)},
                #{'type': 'ineq', 'fun': lambda x: -x[2] + (np.arctan2(Xf[1]-x[1],Xf[0]-x[0])+1.0)},
                #{'type': 'ineq', 'fun': lambda x: np.dot(Xf-x[:2],x[:2]-P)})

    else:
        cons = ({'type': 'ineq', 'fun': lambda x:  integral_intersection_area(x) - intersection_threshold})




    #guess = [0.75,0.5,3*np.pi/2,3*np.pi/2]
    #guess = [0,0,0,0]
    result = minimize(fun, guess, method='COBYLA', constraints=cons)#, options={'tol': 1e-200}) # bounds= bnds) 
    # Adapt tol to avoid local minima


    #result = basinhopping(fun,0, niter=500) #changing niter to avoid local minimum
    return result.x


n_components = 2

box1 = np.array([0.5,0.3])

means1 = np.array([[0.47891806, 0.9040961], 
                  [0.50023427, 0.51117959]])


covariances1 = np.array([[[ 0.00438958, -0.00240083],
                        [-0.00240083,  0.02641489]],
                        [[ 0.0006874,  -0.0005142 ],
                        [-0.0005142,   0.00842463]]])

weights1 = np.array([0.27344326491,
                    0.7265567351])

box2 = np.array([0.5,0.3])

means2 = np.array([[0.47891806, 0.9040961], 
                  [0.50023427, 0.51117959]])


covariances2 = np.array([[[ 0.00438958, -0.00240083],
                        [-0.00240083,  0.02641489]],
                        [[ 0.0006874,  -0.0005142 ],
                        [-0.0005142,   0.00842463]]])

weights2 = np.array([0.27344326491,
                    0.7265567351])


P = np.array([0.0,0.0])


# Xf = [0.7,0.3]
# x_limits = [-0.25, 0.25, 0.9]  #[-0.25, 0.5]
# y_limits = [-0.2, 0.2, 0.4]
# environment = True

# Xf = [1.0,0.7]
# x_limits = [-0.25, 0.4, 1.2]  #[-0.25, 0.5]
# y_limits = [-0.2, 0.4, 0.9]
# environment = True

Xf = [1.55,0.2]
x_limits = [-0.25, 1.9, 1.9]
y_limits = [-0.2, 0.4, 0.4]
environment = True

min_ob = 0
X_opt =[]

# for x in np.linspace(x_limits[0],x_limits[2],4):
#     print("x=",x)
#     for y in np.linspace(y_limits[0],y_limits[2],4):
#         print("y=",y)
#         for t1 in np.linspace(0,2*np.pi,4):
#             print("t1=",t1)
#             for t2 in np.linspace(0,2*np.pi,4):
#                 for intersection_threshold in np.linspace(0.005,0.02,3):
#                     guess = [x,y,t1,t2]

#                     X = find_sol(environment, x_limits, y_limits, guess, intersection_threshold)

#                     if fun(X)<min_ob:
#                         min_ob = fun(X)
#                         X_opt = X

#guess = [0.0,0.5,3*np.pi/2,0*np.pi/2]
guess = [0,0,0,0]
guess[0] = (P[0]+Xf[0])/2
guess[1] = (P[1]+Xf[1])/2
guess[2] = np.arctan2(Xf[1]-guess[1],Xf[0]-guess[0])-np.pi/2
guess[3] = np.arctan2(guess[1]-P[1],guess[0]-P[0]) -np.pi/2

intersection_threshold = 0.005

X_opt = find_sol(environment, x_limits, y_limits, guess, intersection_threshold)

print("theta_1 = ", np.rad2deg(X_opt[3]))
print("Xm = ", X_opt[:2])
print("theta_2 = ", np.rad2deg(X_opt[2]))

print("intersection = ", integral_intersection_area(X_opt))
print("objective function = ", fun(X_opt))



fig, ax = plt.subplots()

# Plot initial position of box
ax.scatter(P[0],P[1], s=100, marker='+',color='g', label ='Xi')
ax.scatter(X_opt[0],X_opt[1], s=100, marker='+',color='b', label ='Xm')

ax.plot([P[0],X_opt[0]],[P[1],X_opt[1]],color='g', linestyle='dashed')
ax.plot([X_opt[0],Xf[0]],[X_opt[1],Xf[1]],color='g', linestyle='dashed')
if environment:
    ax.plot(x_limits[0]*np.ones(20),np.linspace(y_limits[0],y_limits[2],20),color='k')
    ax.plot(x_limits[1]*np.ones(20),np.linspace(y_limits[0],y_limits[1],20),color='k')
    ax.plot(x_limits[2]*np.ones(20),np.linspace(y_limits[1],y_limits[2],20),color='k')    
    ax.plot(np.linspace(x_limits[0],x_limits[2],20),y_limits[2]*np.ones(20),color='k')
    ax.plot(np.linspace(x_limits[1],x_limits[2],20),y_limits[1]*np.ones(20),color='k')
    ax.plot(np.linspace(x_limits[0],x_limits[1],20),y_limits[0]*np.ones(20),color='k', label ='Table')

R1 = np.array([[np.cos(X_opt[3]), -np.sin(X_opt[3])],
            [np.sin(X_opt[3]), np.cos(X_opt[3])]])   

R2 = np.array([[np.cos(X_opt[2]), -np.sin(X_opt[2])],
            [np.sin(X_opt[2]), np.cos(X_opt[2])]])
    
new_means1=[]
new_covariances1=[]
new_means2=[]
new_covariances2=[]
for i in range(n_components):
    R1 = np.squeeze(R1)
    mean1 = means1[i] + (P-box1)
    new_mean1 = R1 @ (mean1-P) + P
    new_covariance1 = R1 @ covariances1[i] @ R1.T
    new_means1.append(new_mean1)
    new_covariances1.append(new_covariance1)

    R2 = np.squeeze(R2)
    mean2 = means2[i] + (X_opt[:2]-box2)
    new_mean2 = R2 @ (mean2-X_opt[:2]) + X_opt[:2] 
    new_covariance2 = R2 @ covariances2[i] @ R2.T
    new_means2.append(new_mean2)
    new_covariances2.append(new_covariance2)


    ellipse.plot_ellipse(new_mean1,new_covariance1,ax)
    ellipse.plot_ellipse(new_mean2,new_covariance2,ax)



ax.scatter(Xf[0],Xf[1], s=100, marker='+',color='k', label ='Xf')


plt.xlabel('X-axis')
plt.ylabel('Y-axis')

plt.title(f'Optimal setup to reach Xf = {Xf}')


ax.legend()
plt.show()


