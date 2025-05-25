import dune
import numpy as np
import pygmsh
import ufl
from dune import fem
from dune.alugrid import aluConformGrid as leafGridView
from dune.common import comm
from dune.fem.space import lagrange, dgonb
from dune.grid import cartesianDomain
from matplotlib import pyplot as plt
#####################################
# additional import
from gmsh2dgf import gmsh2DGF # should be included as a python file in the same directory
from dune.grid import reader as gridReader

H = 0.41
L = 2.2
C = [0.2, 0.2]
R = 0.05
eps = 1e-10
maxH = R # 0.02

with pygmsh.geo.Geometry() as geom:
    geom.set_mesh_size_callback(lambda dim, tag, x, y, z, lc:
        min(maxH + 1.5*( (x-0.2)**2+(y-0.2)**2), H / 2.0)
    )
    obstacle = geom.add_circle([*C, 0.0], R, make_surface=False)
    domain = geom.add_rectangle(0, L, 0, H, 0, holes=[obstacle.curve_loop])
    geom.add_physical(domain, 'domain')
    geom.add_physical(obstacle.curve_loop.curves, 'obstacle')
    mesh = geom.generate_mesh(dim=2)
    points, cells = mesh.points, mesh.cells_dict
    boundaryDomains = {
        1: [[0.0-eps, 0.0-eps], [0.0+eps, H+eps]], # bbox inflow
        2: [[L-eps, 0.0-eps], [L+eps, H+eps]], # bbox outflow
        3: [[C[0]-R-eps, C[1]-R-eps], [C[0]+R+eps, C[1]+R+eps]], # bbox hole
        4: "default", # top and bottom
    }
    dgf = gmsh2DGF(points, cells, bndDomain=boundaryDomains, dim=2)
    obstacleFacets = mesh.cells[0].data[mesh.cell_sets['obstacle'][0]]
    dgf += f'''
Projection
function d(x) = x - {C}
function p(x) = {R} * d(x) / |d(x)| + {C}
'''
    for facet in obstacleFacets:
        dgf += f'''segment {facet[0]} {facet[1]} p
'''
    dgf += '''
#

GridParameter
Name ChannelWithHole
RefinementEdge longest
bisectioncompatibility 1
#
'''
domain = (gridReader.dgfString, dgf)
lb_method = 13
vortex_street_grid = (
    leafGridView(domain, dimgrid=2, lbMethod=lb_method)
    if comm.rank == 0
    else leafGridView({"vertices": [], "cubes": []}, dimgrid=2, lbMethod=lb_method)
)
vortex_street_grid = fem.view.adaptiveLeafGridView(vortex_street_grid)
vortex_street_grid.hierarchicalGrid.globalRefine(2)
# refinement should not be to high as this creates a lot of cells arround the circle 