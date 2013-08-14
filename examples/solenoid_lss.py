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
for R in Rs:
    print R
    for i in range(3):
        n0 = 100
        n = [100, 1000, 10000][i]
        nj = [3, 2, 1][i]
        for j in range(nj):
            u0 = rand(3); u0[0] += R
            tan = Tangent(solenoid, u0, R, n0, n)
            tangent.append(tan.dJds(J))

tangent = array(tangent).reshape([Rs.size, -1])

# load up fd result
Jmean, Jstd2 = loadtxt('solenoid_fd.txt').T
RsMid = 0.5 * (Rs[1:] + Rs[:-1])
dJmean = (Jmean[1:] - Jmean[:-1]) / (Rs[1:] - Rs[:-1])
dJstd2 = sqrt(Jstd2[1:]**2 + Jstd2[:-1]**2) / (Rs[1:] - Rs[:-1])

figure(figsize=(6,4))
plot([RsMid, RsMid], [dJmean + dJstd2, dJmean- dJstd2], '-k', lw=2)
plot(Rs, tangent[:,:3], 'xr')
plot(Rs, tangent[:,3:5], '.g')
plot(Rs, tangent[:,5], '-b')
ylim([0.9, 1])
grid()
xlabel(r'$R$')
ylabel(r'$d\langle J\rangle/dR$')
savefig('solenoid_lss.png')
savefig('solenoid_lss.eps')

