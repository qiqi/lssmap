# Copyright Qiqi Wang (qiqi@mit.edu) 2013
import sys
sys.path.append('..')
import matplotlib
try: matplotlib.use('Agg')
except: pass

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

Jval = []
for R in Rs:
    print R
    n0, n = 100, 10000
    u = rand(10000, 3); u[:,0] += R
    for i in range(n0):
        u = solenoid(u, R)
    Jval.append(zeros(u.shape[0]))
    for i in range(n):
        u = solenoid(u, R)
        Jval[-1] += J(u, R)
    Jval[-1] /= n

Jval = array(Jval)
Jmean, Jstd2 = Jval.mean(1), Jval.std(1) / sqrt(Jval.shape[1]) * 2

savetxt('solenoid_fd.txt', transpose([Jmean, Jstd2]))

# plotting
Jmean, Jstd2 = loadtxt('solenoid_fd.txt').T

# figure(figsize=(5,4))
# plot([Rs, Rs], [Jmean + Jstd2, Jmean- Jstd2], '-k', lw=2)
# grid()

RsMid = 0.5 * (Rs[1:] + Rs[:-1])
dJmean = (Jmean[1:] - Jmean[:-1]) / (Rs[1:] - Rs[:-1])
dJstd2 = sqrt(Jstd2[1:]**2 + Jstd2[:-1]**2) / (Rs[1:] - Rs[:-1])

figure(figsize=(6,4))
plot([RsMid, RsMid], [dJmean + dJstd2, dJmean- dJstd2], '-k', lw=2)
ylim([0.9, 1])
grid()
xlabel(r'$R$')
ylabel(r'$d\overline{J}/dR$')
savefig('solenoid_fd.png')
savefig('solenoid_fd.eps')
