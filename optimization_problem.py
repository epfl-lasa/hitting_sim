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

def find_sol(environment,x_limits,y_limits):#P,Xf,means2, covariances2,n_components):  
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
    threshold = 5
    


    if environment:
        cons = ({'type': 'ineq', 'fun': lambda x:  integral_intersection_area(x) - threshold},
                {'type': 'ineq', 'fun': lambda x:  x[1]-x_limits[0]},
                {'type': 'ineq', 'fun': lambda x:  -x[1]+x_limits[1]},
                {'type': 'ineq', 'fun': lambda x:  x[0]-y_limits[0]},
                {'type': 'ineq', 'fun': lambda x:  -x[0]+y_limits[1]})

    else:
        cons = ({'type': 'ineq', 'fun': lambda x:  integral_intersection_area(x) - threshold})



    guess = [0.0,0.6,3*np.pi/2,0.0]
    #guess = [0.75,0.5,3*np.pi/2,3*np.pi/2]
    #guess = [0,0,0,0]
    result = minimize(fun, guess, method='COBYLA', constraints=cons)#, options={'tol': 1e-200}) # bounds= bnds) 
    # Adapt tol to avoid local minima


    #result = basinhopping(fun,0, niter=500) #changing niter to avoid local minimum
    return result.x

P = np.array([0.0,0.0])
Xf = [0.7,0.6]


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


x_limits = [-0.25, 0.25]  #[-0.25, 0.5]
y_limits = [0.42, 0.8]
environment = True

X = find_sol(environment, x_limits, y_limits)

print("theta_1 = ", np.rad2deg(X[3]))
print("Xm = ", X[:2])
print("theta_2 = ", np.rad2deg(X[2]))

print("intersection = ", integral_intersection_area(X))
print("objective function = ", fun(X))



fig, ax = plt.subplots()

# Plot initial position of box
ax.scatter(P[0],P[1], s=100, marker='+',color='g', label ='Xi')
ax.scatter(Xf[0],Xf[1], s=100, marker='+',color='k', label ='Xf')
ax.scatter(X[0],X[1], s=100, marker='+',color='b', label ='Xm')

if environment:
    ax.plot(x_limits[0]*np.ones(20),np.linspace(-0.2,y_limits[1],20),color='k')
    ax.plot(x_limits[1]*np.ones(20),np.linspace(-0.2,y_limits[0],20),color='k')
    ax.plot(np.linspace(x_limits[0],Xf[0]+0.2,20),y_limits[1]*np.ones(20),color='k')
    ax.plot(np.linspace(x_limits[1],Xf[0]+0.2,20),y_limits[0]*np.ones(20),color='k')
    ax.plot(np.linspace(x_limits[0],x_limits[1],20),-0.2*np.ones(20),color='k')
    ax.plot((Xf[0]+0.2)*np.ones(20),np.linspace(y_limits[0],y_limits[1],20),color='k', label ='Table')

R1 = np.array([[np.cos(X[3]), -np.sin(X[3])],
            [np.sin(X[3]), np.cos(X[3])]])   

R2 = np.array([[np.cos(X[2]), -np.sin(X[2])],
            [np.sin(X[2]), np.cos(X[2])]])
    
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
    mean2 = means2[i] + (X[:2]-box2)
    new_mean2 = R2 @ (mean2-X[:2]) + X[:2] 
    new_covariance2 = R2 @ covariances2[i] @ R2.T
    new_means2.append(new_mean2)
    new_covariances2.append(new_covariance2)


    ellipse.plot_ellipse(new_mean1,new_covariance1,ax)
    ellipse.plot_ellipse(new_mean2,new_covariance2,ax)



plt.xlabel('X-axis')
plt.ylabel('Y-axis')

plt.title(f'Optimal setup to reach Xf = {Xf}')


ax.legend()
plt.show()


