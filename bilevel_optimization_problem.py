import numpy as np
from scipy.optimize import minimize, basinhopping, Bounds
from scipy.stats import multivariate_normal
from scipy.integrate import nquad

from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import ellipse

'''
Changing to two optimization variables

The optimization variable for robot 1

y = angle of the first hit (theta_1)

The optimization variable for robot 2

x[0] = x - the middle position of the box
x[1] = y - the middle position of the box
x[2] = theta_2 - the angle of the second hit

'''

# 1 and 2 correspond to the two GMMs which is the two robots

def pdf_first_reach(theta, x): #means1, covariances1, weights1, means2, covariances2, weights2):
    R1 = np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]])   

    R1 = np.squeeze(R1)

    new_means1=[]
    new_covariances1=[]
    dets_cov1 = []
  
    for i in range(n_components):
        mean1 = means1[i] + (P-box1)
        new_mean1 = R1 @ (mean1-P) + P
        new_covariance1 = R1 @ covariances1[i] @ R1.T
        det_cov1 = np.linalg.det(new_covariance1)
        new_means1.append(new_mean1)
        new_covariances1.append(new_covariance1)
        dets_cov1.append(det_cov1)

 
    pdf = 0
    for i in range(n_components):
        pdf += weights1[i] * (2 * np.pi * np.sqrt(dets_cov1[i])) * multivariate_normal.pdf([x[0], x[1]], mean=new_means1[i], cov=new_covariances1[i])
    
    return -pdf


# def cost_fun(x, type):

#     f = 0
#     R2 = np.array([[np.cos(x[2]), -np.sin(x[2])],
#                 [np.sin(x[2]), np.cos(x[2])]])
#     for i in range(n_components):
#         R2 = np.squeeze(R2)
#         mean = means2[i] + (x[:2]-box2)
#         new_mean = R2 @ (mean-x[:2]) + x[:2] 
#         new_covariance = R2 @ covariances2[i] @ R2.T
#         det_cov = np.linalg.det(new_covariance)
#         pdf = weights2[i]*(2 * np.pi * np.sqrt(det_cov))* multivariate_normal.pdf(Xf, mean=new_mean, cov=new_covariance)
#         f = f + pdf
    
#     if type == 'golf':
#         return -f - 0.2*(pdf_first_reach(x)-0.5)
#     else:
#         return -f*pdf_first_reach(x) - 0.00*(pdf_first_reach(x) - intersection_threshold)

# For the bilevel optimization problem

def cost_fun_bilevel(x):

    f = 0
    R2 = np.array([[np.cos(x[2]), -np.sin(x[2])],
                [np.sin(x[2]), np.cos(x[2])]])
    for i in range(n_components):
        R2 = np.squeeze(R2)
        mean = means2[i] + (x[:2]-box2)
        new_mean = R2 @ (mean-x[:2]) + x[:2] 
        new_covariance = R2 @ covariances2[i] @ R2.T
        det_cov = np.linalg.det(new_covariance)
        pdf = weights2[i]*(2 * np.pi * np.sqrt(det_cov))* multivariate_normal.pdf(Xf, mean=new_mean, cov=new_covariance)
        f = f + pdf
    
    return -f

def fun1(x,Xf):

    f = 0
    R2 = np.array([[np.cos(x[2]), -np.sin(x[2])],
                [np.sin(x[2]), np.cos(x[2])]])
    for i in range(n_components):
        R2 = np.squeeze(R2)
        mean = means2[i] + (x[:2]-box2)
        new_mean = R2 @ (mean-x[:2]) + x[:2] 
        new_covariance = R2 @ covariances2[i] @ R2.T
        det_cov = np.linalg.det(new_covariance)
        pdf = weights2[i]*(2 * np.pi * np.sqrt(det_cov))* multivariate_normal.pdf(Xf, mean=new_mean, cov=new_covariance)

        f = f + pdf
    return f

# CONSTRAINTS
margin = 0.05

# Table constraints:
def table_consx_1(x,x_limits,y_limits,direction):
    if direction[1] == 'right':
        return x[0]-(x_limits[0]+margin) 
    else:
        if direction[0] == 'up':
            return x[0]-(x_limits[0]+margin) if (x[1]>y_limits[1]) else x[0]-(x_limits[1]+margin) 
        else:
            return x[0]-(x_limits[0]+margin) if (x[1]<y_limits[1]) else x[0]-(x_limits[1]+margin) 

def table_consx_2(x,x_limits,y_limits,direction):
    if direction[1] == 'right':
        if direction[0] == 'up':
            return -x[0]+(x_limits[2]-margin)  if (x[1]>y_limits[1]) else -x[0]+(x_limits[1]-margin) 
        else:
            return -x[0]+(x_limits[2]-margin)  if (x[1]<y_limits[1]) else -x[0]+(x_limits[1]-margin) 
    else:
        -x[0]+(x_limits[2]-margin) 

def table_consy_1(x,x_limits,y_limits,direction):
    if direction[0] == 'up':
        return -x[1]+(y_limits[2]-margin)  
    else:
        if direction[1] == 'right':
            return -x[1]+(y_limits[2]-margin)  if(x[0]<x_limits[1]) else -x[1]+(y_limits[1]-margin) 
        else:
            return -x[1]+(y_limits[2]-margin)  if(x[0]>x_limits[1]) else -x[1]+(y_limits[1]-margin) 

def table_consy_2(x,x_limits,y_limits,direction):
    if direction[0] == 'up':
        if direction[1] == 'right':
            return x[1]-(y_limits[0]+margin)  if(x[0]<x_limits[1]) else x[1]-(y_limits[1]+margin) 
        else:
            return x[1]-(y_limits[0]+margin)  if(x[0]>x_limits[1]) else x[1]-(y_limits[1]+margin) 
    else:
        return x[1]-(y_limits[0]+margin) 

# Box always on Table
def cons_1(x, alpha, Xf, x_limits, direction):
    if direction[1] == 'right':
        return ((1 - alpha) * x[0] + alpha * Xf[0]) - (x_limits[0] + margin) 
    else:
        if direction[0] == 'up':
            return ((1 - alpha) * x[0] + alpha * Xf[0])-(x_limits[0]+margin) if (((1 - alpha) * x[1] + alpha * Xf[1])>y_limits[1]) \
                else ((1 - alpha) * x[0] + alpha * Xf[0])-(x_limits[1]+margin) 
        else:
            return ((1 - alpha) * x[0] + alpha * Xf[0])-(x_limits[0]+margin) if (((1 - alpha) * x[1] + alpha * Xf[1])<y_limits[1]) \
                else ((1 - alpha) * x[0] + alpha * Xf[0])-(x_limits[1]+margin) 

def cons_2(x, alpha, Xf, x_limits, y_limits, direction):
    if direction[1] == 'right':
        if direction[0] == 'up':
            return -((1 - alpha) * x[0] + alpha * Xf[0])+(x_limits[2]-margin)  if (((1 - alpha) * x[1] + alpha * Xf[1])>y_limits[1]) \
                else -((1 - alpha) * x[0] + alpha * Xf[0])+(x_limits[1]-margin) 
        else:
            return -((1 - alpha) * x[0] + alpha * Xf[0])+(x_limits[2]-margin)  if (((1 - alpha) * x[1] + alpha * Xf[1])<y_limits[1]) \
                else -((1 - alpha) * x[0] + alpha * Xf[0])+(x_limits[1]-margin) 
    else:
        -((1 - alpha) * x[0] + alpha * Xf[0])+(x_limits[2]-margin) 

def cons_3(x, alpha, Xf, y_limits, direction):
    if direction[0] == 'up':
        return -((1 - alpha) * x[1] + alpha * Xf[1])+(y_limits[2]-margin)  
    else:
        if direction[1] == 'right':
            return -((1 - alpha) * x[1] + alpha * Xf[1])+(y_limits[2]-margin)  if(((1 - alpha) * x[0] + alpha * Xf[0])<x_limits[1]) \
                else -((1 - alpha) * x[1] + alpha * Xf[1])+(y_limits[1]-margin) 
        else:
            return -((1 - alpha) * x[1] + alpha * Xf[1])+(y_limits[2]-margin)  if(((1 - alpha) * x[0] + alpha * Xf[0])>x_limits[1]) \
                else -((1 - alpha) * x[1] + alpha * Xf[1])+(y_limits[1]-margin) 
        
def cons_4(x, alpha, Xf, x_limits, y_limits, direction):   
    if direction[0] == 'up':
        if direction[1] == 'right':
            return ((1 - alpha) * x[1] + alpha * Xf[1])-(y_limits[0]+margin)  if(((1 - alpha) * x[0] + alpha * Xf[0])<x_limits[1]) \
                else ((1 - alpha) * x[1] + alpha * Xf[1])-(y_limits[1]+margin) 
        else:
            return ((1 - alpha) * x[1] + alpha * Xf[1])-(y_limits[0]+margin)  if(((1 - alpha) * x[0] + alpha * Xf[0])>x_limits[1]) \
                else ((1 - alpha) * x[1] + alpha * Xf[1])-(y_limits[1]+margin) 
    else:
        return ((1 - alpha) * x[1] + alpha * Xf[1])-(y_limits[0]+margin) 

def constraints(Xf, x_limits, y_limits, direction):
    cons = []

    # cons.append({'type': 'ineq', 'fun': lambda y:  pdf_first_reach(x) - intersection_threshold}) #+ 0.001*x[4]},
        
    for a in np.linspace(0.0,1.0,num=30):
        cons.append({'type': 'ineq', 'fun': lambda x, a=a: cons_1(x, a, Xf, x_limits, direction)})
        cons.append({'type': 'ineq', 'fun': lambda x, a=a: cons_2(x, a, Xf, x_limits, y_limits, direction)})
        cons.append({'type': 'ineq', 'fun': lambda x, a=a: cons_3(x, a, Xf, y_limits, direction)})
        cons.append({'type': 'ineq', 'fun': lambda x, a=a: cons_4(x, a, Xf, x_limits, y_limits, direction)})

        cons.append({'type': 'ineq', 'fun': lambda x, a=a: cons_1(x, a, P, x_limits, direction)})
        cons.append({'type': 'ineq', 'fun': lambda x, a=a: cons_2(x, a, P, x_limits, y_limits, direction)})
        cons.append({'type': 'ineq', 'fun': lambda x, a=a: cons_3(x, a, P, y_limits, direction)})
        cons.append({'type': 'ineq', 'fun': lambda x, a=a: cons_4(x, a, P, x_limits, y_limits, direction)})
    
    return cons

def guess(P,Xf, table_direction):
    guess = [0,0,0,0]

    if table_direction[0] == 'up' and table_direction[1] == 'right':
        guess[0] = (P[0]+Xf[0])/2
        guess[1] = (P[1]+Xf[1])/2
        guess[2] = np.arctan2(Xf[1]-guess[1],Xf[0]-guess[0])-np.pi/2
        guess[3] = np.arctan2(guess[1]-P[1],guess[0]-P[0]) -np.pi/2
    elif table_direction[0] == 'up' and table_direction[1] == 'left':
        guess[0] = (P[0]+Xf[0])/2
        guess[1] = (P[1]+Xf[1])/2
        guess[2] = np.arctan2(guess[0]-Xf[0],Xf[1]-guess[1])#+np.pi/2
        guess[3] = np.arctan2(P[0]-guess[0],guess[1]-P[1]) #+np.pi/2
    elif table_direction[0] == 'down' and table_direction[1] == 'left':
        guess[0] = (P[0]+Xf[0])/2
        guess[1] = (P[1]+Xf[1])/2
        guess[2] = np.arctan2(guess[1]-Xf[1],guess[0]-Xf[0])+np.pi/2
        guess[3] = np.arctan2(P[1]-guess[1],P[0]-guess[0]) +np.pi/2
    else:
        guess[0] = (P[0]+Xf[0])/2
        guess[1] = (P[1]+Xf[1])/2
        guess[2] = np.arctan2(Xf[0]-guess[0],guess[1]-Xf[1])+np.pi
        guess[3] = np.arctan2(guess[0]-P[0],P[1]-guess[1]) +np.pi
    return guess

def plot_(X_opt, environment, x_limits, y_limits, table_direction, colormap,colormap1):
    fig, ax = plt.subplots()

    if colormap:
        x0_range = np.linspace(x_limits[0], x_limits[2], 100)
        x1_range = np.linspace(y_limits[0], y_limits[2], 100)

        pdf_values = np.zeros((len(x0_range), len(x1_range)))
        for i, x0_val in enumerate(x0_range):
            for j, x1_val in enumerate(x1_range):
                X1 = [x0_val, x1_val]
                pdf_values[j, i] = -fun1(X_opt, X1)

        plt.imshow(pdf_values, extent=[x_limits[0], x_limits[2], y_limits[0], y_limits[2]], origin='lower', cmap='viridis')

    if colormap1:
        x0_range = np.linspace(x_limits[0], x_limits[2], 100)
        x1_range = np.linspace(y_limits[0], y_limits[2], 100)

        pdf_values = np.zeros((len(x0_range), len(x1_range)))
        for i, x0_val in enumerate(x0_range):
            for j, x1_val in enumerate(x1_range):
                X1 = [x0_val, x1_val]
                X_try = list(X_opt)
                X_try[:2] = X1
                pdf_values[j, i] = -fun1(X_try, Xf)

        plt.imshow(pdf_values, extent=[x_limits[0], x_limits[2], y_limits[0], y_limits[2]], origin='lower', cmap='viridis')

    # Plot initial position of box
    ax.scatter(P[0],P[1], s=100, marker='+',color='g', label ='Xi')
    ax.scatter(X_opt[0],X_opt[1], s=100, marker='+',color='b', label ='Xm')

    ax.plot([P[0],X_opt[0]],[P[1],X_opt[1]],color='k', linestyle='dashed')
    ax.plot([X_opt[0],Xf[0]],[X_opt[1],Xf[1]],color='k', linestyle='dashed')

    if environment:
        if table_direction[0] == 'up' and table_direction[1] == 'right':
            ax.plot(x_limits[0]*np.ones(2),[y_limits[0],y_limits[2]],color='k')
            ax.plot(x_limits[1]*np.ones(2),[y_limits[0],y_limits[1]],color='k')
            ax.plot(x_limits[2]*np.ones(2),[y_limits[1],y_limits[2]],color='k') 

            ax.plot([x_limits[0],x_limits[2]],y_limits[2]*np.ones(2),color='k')
            ax.plot([x_limits[1],x_limits[2]],y_limits[1]*np.ones(2),color='k')
            ax.plot([x_limits[0],x_limits[1]],y_limits[0]*np.ones(2),color='k', label ='Table')
        elif table_direction[0] == 'up' and table_direction[1] == 'left':
            ax.plot(x_limits[0]*np.ones(2),[y_limits[1],y_limits[2]],color='k')
            ax.plot(x_limits[1]*np.ones(2),[y_limits[0],y_limits[1]],color='k')
            ax.plot(x_limits[2]*np.ones(2),[y_limits[0],y_limits[2]],color='k') 

            ax.plot([x_limits[0],x_limits[2]],y_limits[2]*np.ones(2),color='k')
            ax.plot([x_limits[0],x_limits[1]],y_limits[1]*np.ones(2),color='k')
            ax.plot([x_limits[1],x_limits[2]],y_limits[0]*np.ones(2),color='k', label ='Table')
        elif table_direction[0] == 'down' and table_direction[1] == 'left':
            ax.plot(x_limits[0]*np.ones(2),[y_limits[0],y_limits[1]],color='k')
            ax.plot(x_limits[1]*np.ones(2),[y_limits[1],y_limits[2]],color='k')
            ax.plot(x_limits[2]*np.ones(2),[y_limits[0],y_limits[2]],color='k') 

            ax.plot([x_limits[1],x_limits[2]],y_limits[2]*np.ones(2),color='k')
            ax.plot([x_limits[0],x_limits[1]],y_limits[1]*np.ones(2),color='k')
            ax.plot([x_limits[0],x_limits[2]],y_limits[0]*np.ones(2),color='k', label ='Table')
        else:
            ax.plot(x_limits[0]*np.ones(2),[y_limits[0],y_limits[2]],color='k')
            ax.plot(x_limits[1]*np.ones(2),[y_limits[1],y_limits[2]],color='k')
            ax.plot(x_limits[2]*np.ones(2),[y_limits[0],y_limits[1]],color='k') 

            ax.plot([x_limits[0],x_limits[1]],y_limits[2]*np.ones(2),color='k')
            ax.plot([x_limits[1],x_limits[2]],y_limits[1]*np.ones(2),color='k')
            ax.plot([x_limits[0],x_limits[2]],y_limits[0]*np.ones(2),color='k', label ='Table')


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

def find_sol(environment,x_limits,y_limits,direction, intersection_threshold):#P,Xf,means2, covariances2,n_components):  
    if environment:
        cons = constraints(Xf, x_limits, y_limits, direction)
    else:
        cons = ({'type': 'ineq', 'fun': lambda x:  pdf_first_reach(x) - intersection_threshold})

    guess_ = guess(P,Xf,direction)
    result = minimize(cost_fun_bilevel, guess_, method='COBYLA', constraints=cons, tol=1e-8, options={'disp': True})
    
    return result.x

def find_theta(P,Xm,means, covariances,n_components): 
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
   print("theta_1 guess is: ", guess)
   
   bnds = Bounds([0], [2*np.pi])
   result = minimize(pdf_first_reach, guess, method='Nelder-Mead', args=Xm, bounds= bnds, options={'disp': True}) 
   print("pdf is: ", pdf_first_reach(result.x,Xm)) 
   return result.x

n_components = 2

box1 = np.array([0.5,0.3])

# TODO: link this to the other file so that the means and variances are automatically updated
means1 = np.array([[0.5021755 , 0.48001064], 
                  [0.48307691, 0.73329577]])

covariances1 = np.array([[[0.00026569, 0.00025749],
                        [0.00025749, 0.00407336]],
                        [[ 0.00184485, -0.00123859],
                        [-0.00123859,  0.01655933]]])

weights1 = np.array([0.5083096399095122, 0.49169036009048783])

box2 = np.array([0.5,0.3])

means2 = np.array([[0.5021755 , 0.48001064], 
                  [0.48307691, 0.73329577]])

covariances2 = np.array([[[0.00026569, 0.00025749],
                        [0.00025749, 0.00407336]],
                        [[ 0.00184485, -0.00123859],
                        [-0.00123859,  0.01655933]]])

weights2 = np.array([0.5083096399095122, 0.49169036009048783])


# P = [0.0,0.0]
# Xf = [0.7,0.3]
# x_limits = [-0.25, 0.25, 0.9]  #[-0.25, 0.5]
# y_limits = [-0.2, 0.2, 0.4]
# table_direction = ['up','right']
# environment = True

P = [0.6,-0.18]
Xf = [-0.0,0.3]
x_limits = [-0.25, 0.25, 0.9]  #[-0.25, 0.5]
y_limits = [-0.2, 0.2, 0.6]
table_direction = ['up','left']
environment = True

# P = [0.05,0.38]
# Xf = [0.7,0.0]
# x_limits = [-0.25, 0.25, 0.9]  #[-0.25, 0.5]
# y_limits = [-0.2, 0.2, 0.4]
# table_direction = ['down','right']
# environment = True

# P = [0.5,0.38]
# Xf = [0.0,0.0]
# x_limits = [-0.25, 0.25, 0.9]  #[-0.25, 0.5]
# y_limits = [-0.2, 0.15, 0.4]
# table_direction = ['down','left']
# environment = True

# P = [0.0,0.0]
# Xf = [1.0,0.7]
# x_limits = [-0.25, 0.4, 1.2]  #[-0.25, 0.5]
# y_limits = [-0.2, 0.4, 0.9]
# table_direction = ['up','right']
# environment = True

# P = [0.0,0.0]
# Xf = [1.55,0.2]
# x_limits = [-0.25, 1.9, 1.9]
# y_limits = [-0.2, 0.4, 0.4]
# table_direction = ['up','right']
# environment = True

# P = [0.0,0.0]
# Xf = [0.7,0.3]
# x_limits = [-0.25, 0.25, 0.9]  #[-0.25, 0.5]
# y_limits = [-0.2, 0.2, 0.4]
# table_direction = ['up','right']
# environment = True

colormap = True
colormap1 = True
intersection_threshold = 0.6

X_opt = find_sol(environment, x_limits, y_limits, table_direction, intersection_threshold)


print(X_opt)

theta_1 = find_theta(P, X_opt[:2], means1, covariances1, n_components)

X_opt[3] = theta_1
print("theta_1 = ", np.rad2deg(theta_1))
print("theta_1 = ", np.rad2deg(X_opt[3]))
print("Xm = ", X_opt[:2])
print("theta_2 = ", np.rad2deg(X_opt[2]))

# print("intersection = ", pdf_first_reach(X_opt))
print("objective function = ", cost_fun_bilevel(X_opt))

plot_(X_opt, environment, x_limits, y_limits, table_direction, colormap,colormap1)

