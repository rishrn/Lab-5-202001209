# -*- coding: utf-8 -*-
"""Rishabh_nandania.ipynb






# **Lab 5 Gradient Descent**
# **Name - Rishabh nandania**
# **ID - 202001209**
---
Question 
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from math import *

"""Salary_data"""

Y = pd.read_csv('salary_data.csv',usecols=['Salary'])  # load 2nd column of Salary_data in vector Y
np.vectorize(Y)
X = np.loadtxt('salary_data.csv', skiprows = 1, delimiter=',')  # load Salary_data in X matrix 

j = 0            # Replace the 1st column of X with 1, that is the constant term of linear regression equation
for i in X:
  X[j][1] = 1
  j = j+1

a = X.transpose()
x = np.dot(a, X)        # (X^T)*(X)
b = np.linalg.inv(x)    # ((X^T)*(X))^-1

c = np.dot(a, Y)        # (X^T)*(Y)
W = np.dot(b, c)        # (((X^T)*(X))^-1)*((X^T)*(Y))
print(W)                # weight vector

"""stats_females"""

Y = pd.read_csv('stats_females.csv',usecols=['Height'])
np.vectorize(Y)

X = np.loadtxt('stats_females.csv', skiprows = 1, delimiter=',')

j = 0
for i in X:
  X[j][0] = 1
  j = j+1

a = X.transpose()
x = np.dot(a, X)
b = np.linalg.inv(x)

c = np.dot(a, Y)
W = np.dot(b, c)
print(W)

"""Question 2"""

def weight_vector(d = 2,limits = np.full(2,1),increment = 0.1):
    w = np.arange(-limits[0],limits[0],increment)

    for i in range(1,d):
        w = np.vstack([w,np.arange(-limits[i],limits[i],increment)]) 

    print(w)

    return w

def get_vars(data,y_idx):
    N = len(data)
    m = len(data[0])
    y = np.array(data[:,y_idx],dtype = 'f')
    x = np.zeros((N,m),dtype = 'f')
    for i in range(0,N):
        x[i][0] = 1;
        k = 1;
        for j in range(0,m):
            if j != y_idx:
                x[i][k] = data[i][j]
                k += 1
    
    return x,y

def hypothesis_function(w,x):
    """
        w = [w0,w1,w2,...,wn] - 1 x (n + 1)
        x = [[1,x10,x20,...,xn0],[1,x11,x21,...,xn1],...,[1,x1(N - 1),x2(N - 1),...,xn(N - 1)]] - N x (n + 1)
        return h = [h0,h1,...,h(N - 1)] - 1 x N ; hi = wx'; 0 <= i < N 
    """
    h = np.array([],dtype = 'f')
    l = len(x) # l = N
    for i in range(0,l):
        t = np.dot(w,x[i].transpose())
        h = np.append(h,t)

    return h

def error_metric(y,h):
    N = len(y)
    sum = 0
    for i in range(0,N):
        sum += (y[i] - h[i]) ** 2

    return sum / N

def error_metric_2(y,x,w):
    z = np.zeros((len(w[0]),len(w[1])))
    for i in range(0,len(w[0])):
        for j in range(0,len(w[1])):
            t = np.array([w[0][i],w[1][j]])
            h = hypothesis_function(t,x)
            z[i][j] = error_metric(y,h)
    return z

def derivative_wrt_wj(y,h,xj):
    """
        xj = [xj0,xj1,...,xj(N - 1)] - 1 x N
    """
    N = len(y)
    sum = 0
    for i in range(0,N):
        sum += (y[i] - h[i]) * xj[i]

    return (-2 * sum) / N

def gradient_decent(initial_guess,learning_rate,y,x,max_iteration = 1000,error_threshold = 1e-5,sample_at = 1):
    l = len(initial_guess)
    w_i = np.copy(initial_guess)
    w_p = np.copy(w_i)
    w_c = np.copy(w_p)
    w_h = np.copy(w_i)

    errors = np.full(l,inf)

    for itr in range(0,max_iteration):
        for i in range(0,l):
            h = hypothesis_function(w_p,x)
            w_c[i] = w_p[i] - learning_rate * derivative_wrt_wj(y,h,x[:,i])


        for i in range(0,l):
            errors[i] = abs(w_c[i] - w_p[i])

        if itr % sample_at == 0:
            w_h = np.vstack([w_h,w_c])

        w_p = np.copy(w_c)

        if all([e <= error_threshold for e in errors]):
            break;

    return w_h

def generate_contour_plot(x, y, z, num_of_contours = 15, ax = None, ax_title = "Plot of $J(w)$", x_label = "$w_0$", y_label = "$w_1$"):
    X,Y = np.meshgrid(x,y)

    if ax == None:
        ax = plt.axes()

    ax.contour(X, Y, z, num_of_contours)
    ax.set_title(ax_title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

def generate_gradient_decent_progress_contour_plot(w,f,w_h,ax,legend_label = None,ax_title = "Plot of $J(w)$"):
    """Generate a contour plot of cost function f along with iterative process of gradient decent algorithm"""

    if legend_label == None:
        legend_label = "Learning points"

    generate_contour_plot(w[0],w[1],f,ax=ax,ax_title = ax_title)
    ax.scatter(w_h[:,0],w_h[:,1],s = 100,marker = 'o',c = 'r')

    for j in range(1, len(w_h)):
        ax.annotate('', xy=np.array([w_h[j][0], w_h[j][1]]), xytext=np.array([w_h[j - 1][0], w_h[j - 1][1]]),
                    arrowprops={'arrowstyle': '->', 'lw': 2},
                    va='center', ha='center')
        
    legend_1 = mpatches.Patch(label = legend_label,color = 'r')

    ax.legend(handles=[legend_1],loc = 'best')

def generate_3D_plot(x, y, z, colorMap = 'magma', ax = None, ax_title = 'Plot of $J(w)$',x_label = "$w_0$", y_label = "$w_1$",z_label = "$J(w)$"):
    X, Y = np.meshgrid(x, y)

    if ax == None:
        ax = plt.axes(projection = '3d')

    surf = ax.plot_surface(X, Y, z, cmap=colorMap)
    surf._facecolors2d = surf._facecolors3d
    surf._edgecolors2d = surf._edgecolors3d

    ax.set_title(ax_title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)

def generate_gradient_decent_progress_3d_plot(w, f,w_h,value_w_h,ax, colorMap = 'cool',ax_title = 'Plot of $J(w)$',x_label = "$w_0$", y_label = "$w_1$",z_label = "$J(w)$"):
    """Generate a 3d plot of cost function f along with iterative process of gradient decent algorithm"""

    generate_3D_plot(w[0],w[1],f,colorMap = colorMap,ax = ax,ax_title = ax_title,x_label = x_label,y_label = y_label,z_label = z_label)

    ax.scatter(w_h[:,0],w_h[:,1],value_w_h,s = 50,marker = 'o',c = 'black')

# with salary data
data = np.genfromtxt('salary_data.csv',delimiter = ',')
data_w = np.delete(data,0,axis=0)
x,y = get_vars(data_w,1)
initial_guess = np.array([0,0],dtype = 'f')
learning_rate = 0.01

w_h = gradient_decent(initial_guess,learning_rate,y,x,max_iteration = 1000,sample_at=100)
opt = w_h[-1]

print(f'Obtained value of W with Intial guess = ({initial_guess[0]},{initial_guess[1]}) and Learning rate = {learning_rate} is ({opt[0]},{opt[1]})')

x1 = data_w[:,0]
y = data_w[:,1]
x1c = np.arange(x1.min(),x1.max() + 1,1)
h = opt[0] + opt[1] * x1c

ax = plt.axes()
ax.scatter(x1,y,c = 'r')
ax.plot(x1c,h,c = 'b')
legend_1 = mpatches.Patch(label = 'data points',color = 'r')
legend_2 = mpatches.Patch(label = 'fitting curve',color = 'b')
ax.legend(handles=[legend_1,legend_2],loc = 'best')
ax.set_title(f'Best fit curve using Gradient Decent algorithm \n $(\omega_0,\omega_1) = ({opt[0]},{opt[1]})$')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$y = \omega_0 + \omega_1x_1$')
plt.show()

w = weight_vector(limits=np.full(2,50000),increment = 1000)
z = error_metric_2(y,x,w)
generate_3D_plot(w[0],w[1],z)
value_w_h = np.zeros(len(w_h))
for i in range(0,len(w_h)):
    h = hypothesis_function(w_h[i],x)
    value_w_h[i] = error_metric(y,h)

ax = plt.axes(projection = '3d')
generate_gradient_decent_progress_3d_plot(w,z,w_h,value_w_h,ax)

w = weight_vector(limits=np.full(2,50000),increment = 1000)
z = error_metric_2(y,x,w)
ax = plt.axes()
generate_gradient_decent_progress_contour_plot(w,z,w_h,ax)
plt.show()

# with female hight data
data = np.genfromtxt('stats_females.csv',delimiter = ',')
data_w = np.delete(data,0,axis=0)
x,y = get_vars(data_w,0)
initial_guess = np.array([18,0,0],dtype = 'f')
learning_rate = 0.0001

w_h = gradient_decent(initial_guess,learning_rate,y,x,max_iteration = 1000,sample_at=1)
opt = w_h[-1]

print(f'Obtained value of W with Intial guess = ({initial_guess[0]},{initial_guess[1]},{initial_guess[2]}) and Learning rate = {learning_rate} is ({opt[0]},{opt[1]},{opt[2]})')

x1 = data_w[:,1]
x2 = data_w[:,2]
y = data_w[:,0]
x1_range = np.linspace(x1.min(), x1.max(), 1000)
x2_range = np.linspace(x2.min(), x2.max(), 1000)
x1c,x2c = np.meshgrid(x1_range,x2_range)
h = opt[0] + opt[1]*x1c + opt[2]*x2c

ax = plt.axes(projection = '3d')
ax.scatter3D(x1,x2,y,c = 'r')
ax.plot_surface(x1c,x2c,h,cmap = 'cool')
legend_1 = mpatches.Patch(label = 'data points',color = 'r')
legend_2 = mpatches.Patch(label = 'fitting curve',color = 'b')
ax.legend(handles=[legend_1,legend_2],loc = 'best')
ax.set_title(f'Best fit curve using Gradient Decent algorithm \n $(\omega_0,\omega_1,\omega_2) = ({opt[0]},{opt[1]},{opt[2]})$')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$y = \omega_0 + \omega_1x_1 + \omega_2x_2$')
plt.show()

"""Question 3"""

x = np.loadtxt('salary_data.csv',skiprows = 1,delimiter = ',')
row=x.shape
avg_x = 0
avg_y = 0
for i in range(0,row[0],1):
  avg_x = avg_x + x[i][0]
  avg_y = avg_y + x[i][1]

avg_x=avg_x/row[0]
avg_y=avg_y/row[0]

Nr = 0
Dr = 0
for i in range(0,row[0],1):
  Nr = Nr + (x[i][0]-avg_x)*(x[i][1]-avg_y)
  Dr = Dr + (x[i][0]-avg_x) ** 2

w1 = Nr/Dr
w0 = avg_y - w1*avg_x

print(w0)
print(w1)
