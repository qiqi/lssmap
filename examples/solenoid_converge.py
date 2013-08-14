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
ns = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]

truth = 0.931450
# +-0.000017 with 3 sigma confidence,
# from 1117 LSS solutions with n=100000, # each has 20 iterations at beginning
# and 20 iterations at the end of the trajectory truncated from averaging.
nrep = 16

tangent = []
for i, n in enumerate(ns):
    print n
    for j in range(nrep):
        u0 = rand(3); u0[0] += R
        tan = Tangent(solenoid, u0, R, n0, n)
        tangent.append(tan.dJds(J))
        print '    ', tangent[-1]

err = reshape(tangent, [len(ns), nrep]) - truth

figure(figsize=(5,4))
gcf().add_axes([.2, .2, .7, .7])
loglog(ns, abs(err).mean(1), 'ok', ms=8)
plot([10, 10000], [.1, 0.0001], '--k')
plot([10, 100000], [0.01, 0.0001], ':k')
xlabel(r'$n$')
ylabel(r'mean error')
savefig('solenoid_converge.png')
savefig('solenoid_converge.eps')
