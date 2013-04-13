This module contains tools for performing tangent sensitivity analysis
and adjoint sensitivity analysis.

The details are described in our paper
"A mathematical analysis of the least squares sensitivity method"
at arXiv

User should define two bi-variate functions, f and J.

f(u, s) defines a dynamical system u_{i+1} = f(u_i,s) parameterized by s

J(u, s) defines the objective function, whose ergodic long time average
        is the quantity of interest.

# Use:
u0 = rand(m)       # initial condition of m-degree-of-freedom system
n0, n = 1000, 1000 # spin-up time and trajectory length

# Using tangent sensitivity analysis:
tan = Tangent(f, u0, s, n0, n)

dJds = tan.dJds(J)

# Using tangent sensitivity analysis:
adj = Adjoint(f, u0, s, n0, n, J)

dJds = adj.dJds()

