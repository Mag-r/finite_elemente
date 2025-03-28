# %% [markdown]
# # Cahn-Hilliard equation
# In this script we show how to solve vector valued problems, how to use
# adaptivity to improve both, solution quality and runtime performance.
# Let's start with importing the necessary modules.

# %%

import time
import numpy as np

import dune.fem as fem
from dune.common import comm
from dune.fem.space import lagrange, finiteVolume
from dune.fem.scheme import galerkin
from dune.fem.function import gridFunction
from dune.fem import integrate
from dune.grid import cartesianDomain
from dune.alugrid import aluConformGrid as leafGridView
from dune.fem.view import adaptiveLeafGridView

# use threading to utilize more cores
# fem.threading.use = 8 # or simply call
# fem.threading.useMax()

# ufl utilities from dune
from dune.ufl import DirichletBC, Constant, cell

# general ufl classes
import ufl
from ufl import (TestFunction,TrialFunction,SpatialCoordinate, dx, div, grad,inner, sin, cos, exp, pi, sqrt,ln, conditional, sign, as_vector)

# %% [markdown]
# 
# To define the mathematical model, let $\psi\colon{\mathbb R} \rightarrow
# \mathbb{R}$ be defined as
# $$\psi(x) = \frac{(1-x^2)^2}{4} \mbox{ and } \phi(x) := \psi(x)^{\prime} = x^3 - x.$$
# 
# The strong form for the solution
# $u\colon \Omega \times [0,T] \rightarrow {\mathbb R}$
# is given by
# \begin{align*}
# \partial_t u  - \Delta (\phi(u)-\epsilon^2 \Delta u) = 0
# \quad &\text{in} \ \Omega \times [0,T] ,\\
# u(\cdot,0) = u_0(\cdot)  \quad &\text{in} \ \Omega,\\
# \partial_n u = \partial_n \big( \phi(u) - \epsilon^2\Delta u \big) = 0
# \quad &\text{on} \ \partial \Omega \times [0,T].
# \end{align*}

# %%
## setup of the problem
useVTK = True

N = 10       # cells in each coordinate direction
dim = 2      # dimension of domain
W, H = 1, 1  # width and height
T = 4        # final time

mxl = 4 # level of initial refinement (half grid width each time)

# create domain
domain = cartesianDomain([0, 0], [W, H], [N, round(H/W)*N], periodic=[False, False])

gridView = leafGridView(domain, dimgrid=dim)

# %% [markdown]
# It is very important that for adaptive computations we use an adaptive leaf
# grid view. Otherwise the data transfer between adaptation steps will not work
# correctly.
# 
# TODO: Create an `adaptiveLeafGridView` passing an existing `leafGridView` as
# argument and simply overload the existing `gridView`

# %%
##### TASK
gridView = adaptiveLeafGridView(gridView)

# %% [markdown]
# Some initial refinement to allow for sufficient resolution of the initial
# data.

# %%
gridView.hierarchicalGrid.globalRefine(mxl * gridView.hierarchicalGrid.refineStepsForHalf)
maxLevel = gridView.hierarchicalGrid.maxLevel

# space for solution
space = lagrange(gridView, order=1, dimRange=2)

# space for indicator function
fvspc = finiteVolume(gridView, dimRange=1)

## discrete functions needed for form
u_prev  = space.interpolate([0]*2, name="u_h_prev")
u_h = space.interpolate([0]*2, name="u_h")

indicator = fvspc.interpolate([0], name="indicator")

u = TrialFunction(space)
v = TestFunction(space)

eps = 0.05

## dune.ufl.Constants
tau  = Constant(1., name="tau") # timestep constant
eps2 = Constant(eps**2, name="eps2") # we need eps^2

# 1 for implicit Euler and 0.5 for Crank-Nicolson
theta = Constant(1.0, name="theta")

# Eyre approximation of energy potential
# use in weak form as phi(u[0], u_prev[0])
phi = lambda u_np1, u_n: u_np1**3 - u_n

# eta_(n+theta) # implicit-explicit splitting
eta_theta = (1.0-theta)*u_prev[1] + theta*u[1]

# time derivative for phase field M:
M = inner(u[0] - u_prev[0], v[0]) * dx

# spatial derivative first equation L0:
L0 = tau*inner(grad(eta_theta), grad(v[0]))*dx

# spatial derivative second equation L1:
L1 =  inner(u[1], v[1])*dx \
     -inner(phi(u[0], u_prev[0]), v[1])*dx \
     -eps2*inner(grad(u[0]), grad( v[1] ))*dx

# The overall weak form is obtained by simply adding up all forms.
L = M + L0 + L1

# Solvers
parameters = {
    "newton.tolerance": 1e-10,
    "newton.linear.preconditioning.relaxation": 0.8,
    "newton.linear.tolerance.strategy": "eisenstatwalker",
    "newton.linear.errormeasure": "residualreduction",
    "newton.linear.preconditioning.method": "ssor",
    "newton.linear.maxIteration": 10000,
    # "newton.verbose": True,        # Newton solver verbosity
    # "newton.linear.verbose": True  # Linear solver verbosity
}

scheme = galerkin(L == 0,
                  solver=("istl", "gmres"),
                  parameters=parameters)


# initial data
def initial(x):
    h = 0.01
    g0  = lambda x,x0,T: conditional(x-x0<-T/2,0,conditional(x-x0>T/2,0,sin(2*pi/T*(x-x0))**3))
    # u_0
    G   = lambda x,y,x0,y0,T: g0(x,x0,T)*g0(y,y0,T)
    # eta_ 0
    eta = lambda v : phi(v,v) - eps2*div(grad(v))
    return as_vector([ G(x[0],x[1],0.5,0.5,50*h), eta(G(x[0],x[1],0.5,0.5,50*h))])

# %% [markdown]
# 
# A suitable refinement indicator as `gridFunction` is given by
# the absolute value of the gradient of the first component of $u_h$ on each element, i.e. $| \nabla ((u_h)_0)_{|E}|$.
# In addition we apply a scaling of the indicator with the difference of maximum and minimum value of all
# indicators, yielding
# $$ \chi_E := \frac{| \nabla ((u_h)_0)_{|E}|}{\chi_{S}}$$
# with $\chi_{S} := \max_{E} { | (\nabla (u_h)_0)_{|E}| } - \min_{E} { | (\nabla (u_h)_0)_{|E}| }$.
# 
# We apply a In order to achieve
# Set `refineTol = 0.75` and the `coarsenTol = 0.1 * refineTol`.
# 
# To easily compute $\chi_S$ (use the `as_numpy` feature) interpolate the gridFunction into the discrete
# function `indicator` which was created above.
# 
# Furthermore, add a function `adapt` takes one argument `maxLevel` and carries out the adaptation steps, as
# discussed in the lecture.
# 

# %%
##### TASK


# indicator = (indicator - minIndicator) / (maxIndicator - minIndicator)

def adapt(maxLevel):
    indicator = sqrt(inner(grad(u_h[0]),grad(u_h[0])))
    indicator = fvspc.interpolate(indicator, name="indicator")
    minIndicator = np.min(indicator.as_numpy)
    maxIndicator = np.max(indicator.as_numpy)
    if maxIndicator > minIndicator:
        indicator.as_numpy[:] -= minIndicator
        indicator.as_numpy[:] /= (maxIndicator - minIndicator)
    
    fem.mark(indicator, refineTolerance=0.75, coarsenTolerance=0.075, maxLevel=maxLevel)
    fem.adapt([u_h,u_prev])

# %% [markdown]
# 

# %%
x = SpatialCoordinate(cell(dim))
# interpolate initial data
u_h.interpolate(initial(x))

# visualization
if useVTK:
    write_solution = gridView.sequencedVTK(
            "cahn_parallel",
            pointdata=[u_h],
            subsampling=0,
        )
else:
    # only plot the phase field, not the auxiliary variable
    write_solution = lambda : u_h[0].plot()

# time step size
tau.value = 1e-2
t = 0

# write 25 communication steps
saveinterval = T / 25.

# %% [markdown]
# 
# Compute the energy. This value should be decaying over time.

# %%

W = lambda v: 1/4*(v**2-1)**2
# energy
Eint  = lambda v: eps*eps/2*inner(grad(v[0]),grad(v[0]))+W(v[0])

# %% [markdown]
# Time-loop

# %%
start = time.time()

# pre-adapt mesh
for i in range(10):
    adapt(maxLevel)

adaptStep = 5

# write initial solution
write_solution()

i = 0
import time
start = time.time()
while t < T:
    # write or plot solution from time to time
    if t // saveinterval > (t - tau.value) // saveinterval:
        energy = integrate(Eint(u_h), gridView=gridView, order=5)
        write_solution()
        print(f"# t, dt, size: {t:.3f}, {tau.value:.2e}, {gridView.size(0)}, Energy = {energy}") if comm.rank == 0 else None
    if t > 0.8:
        tau.value = 4e-2


    # overwrite previous value
    u_prev.assign( u_h ) # u^n
    # Solve for new (u,eta)
    info = scheme.solve(target = u_h)
    # adapt before Navier-Stokes step because this will alter the
    # incompressibility constraint which is then fixed by the solver
    if i % adaptStep == 0:
        adapt(maxLevel)
    if i %20 == 0:
        fem.loadBalance([u_h,u_prev])

    # increment time
    t += tau.value
    i += 1

write_solution()
print("Elapsed time: ", time.time()-start) if comm.rank == 0 else None

