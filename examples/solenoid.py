# Copyright Qiqi Wang (qiqi@mit.edu) 2013

import sys
from pylab import *
from numpy import *

from scipy.sparse.linalg import spsolve
from mpl_toolkits.mplot3d import Axes3D

sys.path.append('..')
from lssmap import *
from lssmap import _block_diag

Lamb = (1 + sqrt(5)) / 2

def solenoid(u, R):
    shape = u.shape
    x, y, z = u.reshape([-1, 3]).T
    # transform
    t = arctan2(x, y)
    r = sqrt(x**2 + y**2)
    dr = r - R
    # map
    dr = dr / 4 + cos(t) / 2
    z = z / 4 + sin(t) / 2
    t = t * 2 # + z
    # transform back
    r = dr + R
    x = r * sin(t)
    y = r * cos(t)
    return transpose([x, y, z]).reshape(shape)
    
def J(u, R):
    x, y, z = u.reshape([-1, 3]).T
    return sqrt(x**2 + y**2 + z**2)

R = 3

# # visualize
# n = 100000
# dr, z = rand(2, n)
# t = linspace(0, 2*pi, n)
# r = dr + R
# x = r * cos(t)
# y = r * sin(t)
# u = transpose([x, y, z])

# for i in range(10):
#     u = solenoid(u, R)
# x, y, z = u.reshape([-1, 3]).T
# figure().add_subplot(111, projection='3d')
# plot(x, y, z, '.k', ms=1)
# gca().auto_scale_xyz([-R-1,R+1], [-R-1,R+1], [-R-1,R+1])
# axis('scaled'); xlabel('x'); ylabel('y')

# lss
u0 = rand(3); u0[0] += R
n0, n = 10, 1000
tan = Tangent(solenoid, u0, R, n0, n)
dJds = tan.dJds(J)
print dJds[0]
