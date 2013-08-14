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

R = 1.0
n0, n = 100, 100000

while True:  # keep writing new lines on solenoid_truth.txt for averaging
    u0 = rand(3); u0[0] += R
    tan = Tangent(solenoid, u0, R, n0, n)
    with file('solenoid_truth.txt', 'at') as f:
        f.write('%f\n' % tan.dJds(J, n0skip=20, n1skip=20))
