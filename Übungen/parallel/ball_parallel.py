# %% [markdown]
# # Ball refinement
# This is a simple grid refinement and coarsening example.
# We define a simple time dependent indicator function to
# mark elements for refinement and coarsening.

# %%

import numpy as np
from dune.common import FieldVector
from dune.grid import Marker
from dune.alugrid import aluCubeGrid, aluConformGrid
from dune.grid import cartesianDomain


import dune.fem
from dune.fem.view import adaptiveLeafGridView
from dune.fem.space import finiteVolume
from dune.fem.function import gridFunction

from ufl import SpatialCoordinate, sin, cos, pi, sqrt, dot, as_vector, conditional
from dune.ufl import Constant

lbMethod=9 # help(aluConformGrid) for more information

domain = cartesianDomain([0,0], [1,1], [16,16])
# view = aluCubeGrid( domain )
view = aluConformGrid( domain, lbMethod=lbMethod )

maxLevel = 5
t = 0.

# %% [markdown]
# As an example we implement a simple refinement indicator.
# \begin{equation}
#  \eta_E = \left \{ \begin{array}{ll}
#                        1 & \mbox{ if } \quad 0.15 < |c_E - y(t)| < 0.25 \\
#                        0 & \mbox{ else }.
#                       \end{array} \right .
# \end{equation}
# where $c_E$ is the elements center and $y(t) = (r_0\cos(t 2 \pi ) + 0.5, r_0\sin(t 2 \pi) + 0.5 )^T$ and $r_0 = 0.3$.
# Mark an element $E$ for refinement, if $\eta_E = 1$, otherwise mark for coarsening.
# 
# Implement $\eta_E$ and a marking function with the following signature:
# ```
# def eta( element ):
#     return 0.
# 
# def markh( element ):
#     return Marker.keep
# ```

# %%
##### TASK
def eta(element):
    try:
        center = element.geometry.center
    except AttributeError:
        center = element
    ball = FieldVector([0.3 * cos(t * 2 * pi) + 0.5, 0.3 * sin(t * 2 * pi) + 0.5])
    
    if np.abs(sqrt((center - ball).two_norm) - 0.3) < 0.05:
        return 1
    else:
        return 0
    
def markh(element):
    return Marker.refine if eta(element) else Marker.coarsen 

# %% [markdown]
# We invoke some initial refinement

# %%

# initial refinement
view.hierarchicalGrid.globalRefine(maxLevel-1)

from dune.grid import gridFunction
@gridFunction(view)
def levelFct(e, x):
    return e.level

# %% [markdown]
# Adapt the grid in each time step in the time interval $[0,T]$ with $T=1.5$ and
# use a time step of $\Delta t = 0.01$.
# Use the hierarchical grids adapt function:
# ```
# view.hierarchicalGrid.adapt(markh)
# ```
view = adaptiveLeafGridView( aluConformGrid( domain, lbMethod=lbMethod ) )
# initial refinement
view.hierarchicalGrid.globalRefine(maxLevel-1)

# a space to store the indicator
space = finiteVolume( view )

# %% [markdown]
# Task: Write $\eta_E$ as UFL expression and use the routines
# ```
# dune.fem.mark(indicator, refineTolerance=0.7, coarsenTolerance=0.1, maxLevel=maxLevel )
# dune.fem.adapt( view.hierarchicalGrid )
# ```
# to carry out the adaptation.
# 

# %%
indicator = space.interpolate( eta , name = "indicator" )
write_solution = view.sequencedVTK(
            "phasefield",
            pointdata=[indicator],
            subsampling=0
        )
t = 0
dune.fem.loadBalance(view.hierarchicalGrid)
while t < 0.5:
    print("t = ", t)
    indicator = space.interpolate( eta , name = "indicator" )
    dune.fem.mark(indicator = indicator,maxLevel=maxLevel, refineTolerance=0.5, coarsenTolerance=0.5)
    dune.fem.adapt(view.hierarchicalGrid)
    dune.fem.loadBalance(view.hierarchicalGrid)
    write_solution()
    t += 0.01
    


