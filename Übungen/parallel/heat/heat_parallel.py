# %% [markdown]
# # Heat Equation with moving Source
# 
# ## Problem Description
# 
# We model a heated room with a window and a moving oven by the
# following initial-boundary-value-problem
# 
# \begin{align*}
#     \partial_t u - K\,\Delta u &= f\;\text{ in }\Omega=(-1,1)\times(-1,1),\\
#     K\,\partial_\nu u &= \alpha\,(g_R - u)\;\text{ on }\Gamma_R,\\
#     K\,\partial_\nu u &= 0\;\text{ on }\partial\Omega\setminus \Gamma_R,
# \end{align*}
# 
# with initial condition $u=0$, the diffusion coefficient $K\equiv 1/100$, the time-dependent
# load function
# 
# \begin{equation*}
#   f(x,\,t)=\begin{cases}
#     \mu,& x\in\,B_r(P(t))\\
#     0,& else.
#   \end{cases},\;
#   P(t) = R\,
#   \begin{pmatrix}
#     \cos(\omega\,t)\\
#     \sin(\omega\,t)
#   \end{pmatrix},
# \end{equation*}
# 
# $\mu=8$, $r=0.2$, $R=0.6$, $\omega=0.01$. The window is located at
# 
# \begin{equation*}
#   \Gamma_R=(-0.5,\,0.5)\times\{1\}
# \end{equation*}
# 
# and ``connects'' the outside temperature $g_R\equiv -5$ with an
# heat-conductivity-coefficient of $\alpha=1.2$ to the inside. The
# remaining parts of the walls are considered to be ideal isolators.
# 
# ## Weak Formulation
# 
# The weak formulation reads as follows: find $u\in H^1(\Omega)$ s.t.
# 
# \begin{equation*}
#   0=
#   \frac{\mathrm{d}}{\mathrm{d}t}\int_\Omega u\,\phi
#   +
#   \int_\Omega K\,\nabla u\cdot\nabla\phi-f\,\phi\,{\rm dx}
#   +
#   \int_{\Gamma_R}\alpha\,(u-g_R)\,\phi\,{\rm ds}\quad\forall\;\phi\in H^1(\Omega).
# \end{equation*}
# 
# or equivalently using characteristic functions for the support of
# $f$ and the Robin-boundary
# 
# \begin{equation*}
#   0=
#   \frac{\mathrm{d}}{\mathrm{d}t}\int_\Omega u\,\phi
#   +
#   \int_\Omega K\,\nabla u\cdot\nabla\phi-\mu\,\chi_{B_r(P(t)}\,\phi\,{\rm dx}
#   +
#   \int_{\partial\Omega}\alpha\,(u-g_R)\,\chi_{\Gamma_R}\,\phi\,{\rm ds}.
# \end{equation*}
# 
# ## Error Estimator
# 
# Given a discrete solution $u_h$ one can use the following expression
# as a local error indicator $\eta_T$ on an element $T$
# 
# \begin{equation*}
#   \begin{split}
#     \eta_T &= h_T^2||R(u_h^{n+1},u_h^n)||^2_{L^2(T)} + \sum_{e\in{\cal F}^i(T)} h_e\,||[K\,\partial_n u_h^{n+1}]||^2_{L^2(e)}\\
#     &
#     +\sum_{e\in{\cal F}^o(T)} h_e\,||K\,\partial_n u_h^{n+1}-\alpha\,(u_h^{n+1}-g_R)\,\chi_{\Gamma_R}||^2_{L^2(e)}
#   \end{split}
# \end{equation*}
# 
# with the element residual $R(v,w)=(v-w)/\Delta t - K\,\Delta v -
# \mu\,\chi_{B_r(P(t^{n+1}))}$.  ${\cal F}^i(T)$ denotes set sets of
# inner faces of an element $T$, ${\cal F}^o(T)$ denotes the sets of
# faces of $T$ which intersect the boundary.

# %% [markdown]
# ## Program Setup

# %%
import ufl
from ufl import grad, div, jump, avg, dot, dx, ds, dS, inner, sin, cos, pi, exp, sqrt, Integral
import dune.ufl
import dune.grid
import dune.fem
import dune.generator
import dune.alugrid
from dune.common import comm

useAdaptivity = True

endTime  = 100
# ... keep large in order to avoid huge PDFs
saveInterval = 4 # for VTK
initialRefinements = 2
globalTolerance = 5e-1
refineFraction = 0.9
coarsenFraction = 0.4

domain = dune.grid.cartesianDomain([-1,-1],[1,1],[20,20])
# Use an unstructured grid in order to be able to use local mesh adaptation
gridView = dune.fem.view.adaptiveLeafGridView(dune.alugrid.aluConformGrid(domain, lbMethod=13))
gridView.hierarchicalGrid.globalRefine(initialRefinements)
dune.fem.loadBalance(gridView.hierarchicalGrid)

space = dune.fem.space.lagrange(gridView, order=1, storage="istl")
u     = ufl.TrialFunction(space)
phi   = ufl.TestFunction(space)
x     = ufl.SpatialCoordinate(space)
dt    = dune.ufl.Constant(0.1, "timeStep")
t     = dune.ufl.Constant(0.0, "time")

# define storage for discrete solutions
uh     = space.interpolate(0, name="uh")
uh_old = uh.copy()

# initial solution (passed to interpolate later)
initial = 0

# %% [markdown]
# ## Problem definition
# 

# %% [markdown]
# ### Possible implementation for the oven

# %%
ROven = 0.6
omegaOven = 0.05*2*pi
P = lambda s: ufl.as_vector([ROven*cos(omegaOven*s), ROven*sin(omegaOven*s)])
rOven = 0.2
ovenEnergy = 8
chiOven = lambda s: ufl.conditional(dot(x-P(s), x-P(s)) < rOven**2, 1, 0)
ovenLoad = lambda s: ovenEnergy * chiOven(s)

# %% [markdown]
# ### One possibility to model the desk

# %%
deskCenter = [-0.8, -0.8]
deskSize = 0.2
chiDesk = ufl.conditional(abs(x[0]-deskCenter[0]) < deskSize, 1, 0)\
  * ufl.conditional(abs(x[1] - deskCenter[1]) < deskSize, 1, 0)

# %% [markdown]
# 
# #### Possible implementation of $\chi_{\Gamma_R}$:

# %%
windowWidth = 0.5
chiWindow = ufl.conditional(abs(x[1]-1.0) < 1e-8, 1, 0)*ufl.conditional(abs(x[0]) < windowWidth, 1, 0)

# %%
transmissionCoefficient = 1.2
outerTemperature = -5.0
rBC = transmissionCoefficient * (u - outerTemperature) * chiWindow

# %% [markdown]
# ### Write down the UFL-form, generate the scheme, provide the initial value ...

# %%
# heat diffussion
K = 0.01

# space form
diffusiveFlux = K*grad(u)
source = -ovenLoad(t+dt)

xForm = dot(diffusiveFlux, grad(phi)) * dx + source * phi * dx + rBC * phi * ds

# add time discretization
form = dot(u - uh_old, phi) * dx + dt * xForm

scheme = dune.fem.scheme.galerkin(form == 0, solver="cg")

# %% [markdown]
# ### Define the residual error estimator given above

# %%
##### TASK

elementStorage = dune.fem.space.finiteVolume(gridView)
chiT = ufl.TestFunction(elementStorage)
n= ufl.FacetNormal(elementStorage)
hT = ufl.MaxCellEdgeLength(elementStorage)
he = ufl.MaxFacetEdgeLength(elementStorage)


# %% [markdown]
# #### Residual Estimator as UFL Form
# 
# Here it is important to note, that the test-function $\chi_T$ is in
# particular needed to make sure that the element contribution is
# stored into the correct componnt of the DOF-vector of the discrete
# estimate "function". Hence it must be there and cannot be ommitted
# although $\chi_T$ on each element just evaluates to $1$, and
# $\mathrm{avg}(\chi_T)$ evaluates to 0.5.

# %%
residual = (u-uh_old)/dt - div(diffusiveFlux) + source

estimatorForm = hT**2 * residual**2 * chiT * dx\
  + he * (inner(diffusiveFlux, n) - rBC)**2 * chiT * ds\
  + he * inner(jump(diffusiveFlux), n('+'))**2 * avg(chiT) * dS
estimator = dune.fem.operator.galerkin(estimatorForm)
estimate = elementStorage.interpolate(0, name="estimate")

# %% [markdown]
# ### Prepare for the time loop: initial data

# %%
nextSaveTime = saveInterval
uh.interpolate(initial)
@dune.fem.function.gridFunction(gridView, codim=0, order=0)
def rank(element, x):
    return comm.rank
comm_dist = elementStorage.interpolate(rank, name="rank")
vtk = gridView.sequencedVTK("heatrobin", pointdata=[uh,estimate,comm_dist])
vtk()

total_number_grid = comm.sum(gridView.size(0))
# %%
iter = 0
while t.value < endTime:
    uh_old.assign(uh)
    info = scheme.solve(target=uh)
    deskTemperature = dune.fem.function.integrate(gridView, uh * chiDesk, order=1) / deskSize**2 / 4
    estimator(uh, estimate)

    errorEstimate = sum(estimate.dofVector)
    # print("estimated error: ", sqrt(errorEstimate))
    # print("min. est.^2: ", min(estimate.dofVector))
    # print("max. est.^2: ", max(estimate.dofVector))

    t.value += dt.value
    print("Computed solution at time", t.value,
              "desk temperature", deskTemperature,
              "iterations: ", info["linear_iterations"],
              "#Ent: ", gridView.size(0) ) if comm.rank == 0 else None
    if t.value >= nextSaveTime or t.value >= endTime:
        vtk()
        # uh.plot()
        nextSaveTime += saveInterval

    if useAdaptivity:
        iter += 1
        total_number_grid = comm.sum(gridView.size(0))

        avgTolerance = globalTolerance**2 / total_number_grid
        refineTol = avgTolerance * refineFraction
        coarsenTol = avgTolerance * coarsenFraction
        # print("local tolerances ", avgTolerance, refineTol, coarsenTol) if comm.rank == 0 else None
        [refined, coarsened] = dune.fem.mark(estimate, refineTol, coarsenTol)
        # print("#refined/coarsended: ", refined, coarsened) if comm.rank == 0 else None
        dune.fem.adapt([uh])
        if iter % 5 == 0:
            # dune.fem.loadBalance([uh])
            elementStorage = dune.fem.space.finiteVolume(gridView)
            comm_dist = elementStorage.interpolate(rank, name="rank")



