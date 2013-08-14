# Copyright Qiqi Wang (qiqi@mit.edu) 2013
import sys
sys.path.append('..')
from pylab import *
from numpy import *

from scipy.sparse.linalg import spsolve
from mpl_toolkits.mplot3d import Axes3D

sys.path.append('..')
from lssmap import *

def solenoid(u, R):
    x, y, z = u.reshape([-1, 3]).T
    # transform from (x,y) to to (t,r)
    t = arctan2(x, y)
    r = sqrt(x**2 + y**2)
    dr = r - R
    # map in the (t,r-R,z) space
    dr = dr / 4 + cos(t) / 2
    z = z / 4 + sin(t) / 2
    t = t * 2 # + z
    # transform back from (t,r) to (x,y)
    r = dr + R
    x = r * sin(t)
    y = r * cos(t)
    return transpose([x, y, z])
    
def J(u, R):
    x, y, z = u.reshape([-1, 3]).T
    return sqrt(x**2 + y**2 + z**2)

Rs = linspace(1,3,21)

tangent = []
R = 2.0
n0 = 100
n = 100
u0 = rand(3); u0[0] += R
tan = Tangent(solenoid, u0, R, n0, n)

v0 = tan.u[:,:2] / sqrt((tan.u[:,:2]**2).sum(1))[:,newaxis]
err = v0 - tan.v[:,:2]

figure(figsize=(6,4))
semilogy(sqrt((err**2).sum(1)), 'o')
xlabel(r'$i$')
ylabel(r'Least squares shadowing error')
grid()
savefig('solenoid_err.png')
savefig('solenoid_err.eps')
