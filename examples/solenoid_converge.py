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
n0, nTruth = 1000, 1000000
ns = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]

# u0 = rand(3); u0[0] += R
# tan = Tangent(solenoid, u0, R, n0, nTruth)
# tangent = [tan.dJds(J)]
tangent = [0.98334097]

print tangent[0]
sys.stdout.flush()

for i, n in enumerate(ns):
    print n
    for j in range(3):
        u0 = rand(3); u0[0] += R
        tan = Tangent(solenoid, u0, R, n0, n)
        tangent.append(tan.dJds(J))

err = array(tangent[1:]).reshape([len(ns), 3]) - tangent[0]

figure(figsize=(5,4))
loglog(ns, abs(err), 'ok')
grid()
