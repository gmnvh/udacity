# Given the formula for a gaussian
# Calculate f(x) for
# u = 10
# variance = 4
# x = 8
#
# f(x) = 1/sqr(2pi variance) exp (-1/2 (x-u)^2)/variance

from math import sqrt, exp, pi

def gaussian(x, u, variance):
    y = (1/sqrt(2*pi*variance)) * exp(-1.0/2 * (((x-u)**2)/variance))
    return y

print 'f(x) for u=10, variance=4 and x = 8 is ', gaussian(8.0, 10.0, 4.0)

# Multiplying two gaussians
#
# u' = 1/v1+v2 [v2.u1 + v1.u2]
#
# v' = 1/ (1/v1 + 1/v2)

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab

x = np.linspace(--5, 20, 100)

u1 = 10
v1 = 8
u2 = 13
v2 = 2

plt.plot(x,mlab.normpdf(x, u1, sqrt(v1)), label='1')
plt.plot(x,mlab.normpdf(x, u2, sqrt(v2)), label='2')

u3 = 1.0/(v1+v2) * (v2*u1 + v1*u2)
v3 = 1.0/(1.0/v1 + 1.0/v2)
plt.plot(x,mlab.normpdf(x, u3, sqrt(v3)), label='3')

plt.legend()
print '\n\r Multiplying two gaussians: u3: ', u3, ' v3: ', v3
plt.show()

# Measument step of the filter
def update(mean1, var1, mean2, var2):
    new_mean = 1.0/(var1+var2) * (var2*mean1 + var1*mean2)
    new_var = 1.0/(1.0/var1 + 1.0/var2)
    return [new_mean, new_var]

# Motion update
#
# u' = u + distance
# v' = v + v_of_move
    
def predict(mean1, var1, mean2, var2):
    new_mean = mean1 + mean2
    new_var = var1 + var2
    return [new_mean, new_var]

new_mean, new_var = predict(8, 4, 10, 6)
print '\r\nmean=8, var=4, move 10 var=6 : new mean ', new_mean, ', new var ', new_var 

# Sequence of measuments and motions
measurements = [5., 6., 7., 9., 10.]
motion = [1., 1., 2., 1., 1.]
measurement_sig = 4.
motion_sig = 2.
mu = 0.
sig = 10000.

p = [mu, sig]
for measurement, m in zip(measurements, motion):
    p = update(p[0], p[1], measurement, measurement_sig)
    print p
    p = predict(p[0], p[1], m, motion_sig)
    print p


