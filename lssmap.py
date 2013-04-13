# Copyright Qiqi Wang (qiqi@mit.edu) 2013
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

'''This module contains tools for performing tangent sensitivity analysis
and adjoint sensitivity analysis.  The details are described in our paper
"A mathematical analysis of the least squares sensitivity method"
at arXiv

User should define two bi-variate functions, f and J

f(u, s) defines a dynamical system u[i+1] = f(u[i],s) parameterized by s
        inputs:
        u: size (m,) or size (N,m). It's the state of the m-degree-of-freedom
           discrete dynamical system
        s: parameter of the dynamical system.
           Tangent sensitivity analysis: s must be a scalar.
           Adjoint sensitivity analysis: s may be a scalar or vector.
        return: du/dt, should be the same size as the state u.
                if u.shape == (m,): return a shape (m,) array
                if u.shape == (N,m): return a shape (N,m) array

J(u, s) defines the objective function, whose ergodic long time average
        is the quantity of interest.
        inputs: Same as in f(u,s)
        return: instantaneous objective function to be time averaged.
                Tangent sensitivity analysis:
                    J may return a scalar (single objectives)
                              or a vector (n objectives).
                    if u.shape == (m,): return a scalar or vector of shape (n,)
                    if u.shape == (N,m): return a vector of shape (N,)
                                         or vector of shape (N,n)
                Adjoint sensitivity analysis:
                    J must return a scalar (single objective).
                    if u.shape == (m,): return a scalar
                    if u.shape == (N,m): return a vector of shape (N,)

Using tangent sensitivity analysis:
        u0 = rand(m)       # initial condition of m-degree-of-freedom system
        n0, n = 1000, 1000 # spin-up time and trajectory length
        tan = Tangent(f, u0, s, n0, n)
        dJds = tan.dJds(J)
        # you can use the same "tan" for more "J"s ...

Using adjoint sensitivity analysis:
        adj = Adjoint(f, u0, s, n0, n, J)
        dJds = adj.dJds()
        # you can use the same "adj" for more "s"s
        #     via adj.dJds(dfds, dJds)... See doc for the Adjoint class
'''

import numpy as np
from scipy import sparse
import scipy.sparse.linalg as splinalg


__all__ = ["ddu", "dds", "Tangent", "Adjoint"]


def _block_diag(A):
    'Construct a block diagonal sparse matrix, A[i,:,:] is the ith block'
    assert A.ndim == 3
    n = A.shape[0]
    return sparse.bsr_matrix((A, np.r_[:n], np.r_[:n+1]))


EPS = 1E-7

class ddu(object):
    '''
    A callable object that returns the total derivative of a given function f
    with respect to its first argument
    '''
    def __init__(self, f):
        self.f = f

    def __call__(self, u, s):
        f0 = self.f(u, s)
        assert f0.shape[0] == u.shape[0]
        N = f0.shape[0]
        n, m = f0.size / N, u.shape[1]
        dfdu = np.zeros( (N, n, m) )
        for i in range(m):
            u[:,i] += EPS
            fp = self.f(u, s).copy()
            u[:,i] -= EPS * 2
            fm = self.f(u, s).copy()
            u[:,i] += EPS
            dfdu[:,:,i] = (fp - fm).reshape([N, n]) / (2 * EPS)
        return dfdu


class dds(object):
    '''
    A callable object that returns the total derivative of a given function f
    with respect to its second argument
    '''
    def __init__(self, f):
        self.f = f

    def __call__(self, u, s):
        f0 = self.f(u, s)
        assert f0.shape[0] == u.shape[0]
        N = f0.shape[0]
        n, m = f0.size / N, s.size
        dfds = np.zeros( (N, n, m) )
        for i in range(m):
            s[i] += EPS
            fp = self.f(u, s).copy()
            s[i] -= EPS * 2
            fm = self.f(u, s).copy()
            s[i] += EPS
            dfds[:,:,i] = (fp - fm).reshape([N, n]) / (2 * EPS)
        return dfds


class LSS(object):
    '''
    Base class for both tangent and adjoint sensitivity analysis
    During __init__, a trajectory is computed,
    and the matrices used for both tangent and adjoint are built
    '''
    def __init__(self, f, u0, s, n0, n, dfdu=None):
        self.f = f
        self.t = arange(n)
        self.s = np.array(s).copy()

        if self.s.ndim == 0:
            self.s = self.s[np.newaxis]

        if dfdu is None:
            dfdu = ddu(f)
        self.dfdu = dfdu

        # run up for n0 steps
        for i in xrange(n0):
            u0 = self.f(u0, s)

        # compute a trajectory
        self.u = zeros([n, u0.size])
        self.u[0] = f(u0, s) 
        for i in xrange(1, n):
            self.u[i] = f(self.u[i-1], s)

        self.buildSparseMatrices()
    
    def buildSparseMatrices(self):
        '''
        Building B: the block-bidiagonal matrix
        '''
        N, m = self.u.shape[0] - 1, self.u.shape[1]

        J = self.dfdu(self.u[:-1], self.s)
        eye = np.eye(m,m)
    
        L = sparse.bsr_matrix((J, np.r_[:N], np.r_[:N+1]), \
                              shape=(N*m, (N+1)*m))
        I = sparse.bsr_matrix((eyeDt, np.r_[1:N+1], np.r_[:N+1]))
    
        self.B = I.tocsr() - L.tocsr()

    def Schur(self):
        'Builds the Schur complement of the KKT system'
        return (self.B * self.B.T)

    def evaluate(self, J):
        'Evaluate a time averaged objective function'
        return J(self.u, self.s).mean(0)


class Tangent(LSS):
    def __init__(self, f, u0, s, n0, n, dfds=None, dfdu=None):
        LSS.__init__(self, f, u0, s, n0, n, dfdu)

        S = self.Schur()

        if dfds is None:
            dfds = dds(f)
        b = dfds(self.uMid, self.s)
        assert b.size == S.shape[0]

        w = splinalg.spsolve(S, np.ravel(b))
        v = self.B.T * w

        self.v = v.reshape(self.u.shape)

    def dJds(self, J):
        dJdu, dJds = ddu(J), dds(J)

        Jp = J(self.u + EPS * self.v, self.s).mean(0)
        Jm = J(self.u - EPS * self.v, self.s).mean(0)
        grad1 = (Jp - Jm) / (2*EPS)

        grad2 = dJds(self.uMid, self.s).mean(0)
        return ravel(grad1 + grad2)


class Adjoint(LSS):
    def __init__(self, f, u0, s, t, J, dJdu=None, dfdu=None):
        LSS.__init__(self, f, u0, s, t, dfdu)

        S = self.Schur()

        if dJdu is None:
            dJdu = ddu(J)
        g = dJdu(self.u, self.s) / self.u.shape[0]      # multiplier on v
        assert g.size == self.u.size

        b = self.B * np.ravel(g)
        wa = splinalg.spsolve(S, b)

        self.wa = wa.reshape(self.uMid.shape)
        self.J, self.dJdu = J, dJdu

    def dJds(self, dfds=None, dJds=None):
        if dfds is None:
            dfds = dds(self.f)
        if dJds is None:
            dJds = dds(self.J)

        prod = self.wa[:,:,np.newaxis] * dfds(self.uMid, self.s)
        grad1 = prod.sum(0).sum(0)
        grad2 = dJds(self.uMid, self.s).mean(0)
        return ravel(grad1 + grad2)
