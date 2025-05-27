# look at first solve only (even only first of icps?), very fine grid, no adaption (not measured anyway)
# not parallel? because not working. Seriell can start with 
# no prints or vtks

from dune.alugrid import aluCubeGrid
import dune
import numpy as np
import pygmsh
import ufl
from dune import fem
from dune.alugrid import aluConformGrid as leafGridView
from dune.common import comm
from dune.fem.space import lagrange
from dune.grid import cartesianDomain
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy
from scipy.sparse import linalg
import sys
import os
from gmsh2dgf import gmsh2DGF
from dune.grid import reader as gridReader
original_dir = os.getcwd()
output_dir = os.path.join(original_dir, "karman/petsc_"+str(comm.size)+"p_precond")
os.makedirs(output_dir, exist_ok=True)
os.chdir(output_dir)
import functools
print = functools.partial(print, flush = True) if comm.rank==0 else lambda *a, **kw: None
import time

#help(aluCubeGrid)
mu  = dune.ufl.Constant(1, "mu")
rho = dune.ufl.Constant(1, "rho")
dt = dune.ufl.Constant(0.02, "dt")

class NavierStokesSolver:
    def __init__(self, gridView, order, mu, rho, dt, max_refinement_level=5, init_method="monolithic"):
        self.gridView = gridView
        self.order = order
        self.mu = mu
        self.rho = rho
        self.dt = dt
        self.init_method = init_method
        self.t = dune.ufl.Constant(0, name="time")
        self.setup_space()
        self.max_refinement_level = max_refinement_level
        self.gridView.hierarchicalGrid.loadBalance()
        dune.fem.loadBalance([self.solution_u, self.solution_p, self.u_old, self.p_old])
        #self.curl = self.calc_curl()
        #self.vtk = self.gridView.sequencedVTK(
        #    "karman", pointdata=[self.solution_u, self.solution_p, self.curl],
        #)
        #self.vtk()  # stores image

    def setup_space(self):
        self.V_space = lagrange(self.gridView, order=self.order, dimRange=self.gridView.dimension) #, storage='petsc'
        # pressure space
        self.P_space = lagrange(self.gridView, order=self.order - 1) #, storage='petsc'
        self.x_u = ufl.SpatialCoordinate(self.V_space)
        self.n_u = ufl.FacetNormal(self.V_space)
        self.u = ufl.TrialFunction(self.V_space)
        self.v = ufl.TestFunction(self.V_space)
        self.x_p = ufl.SpatialCoordinate(self.P_space)
        self.n_p = ufl.FacetNormal(self.P_space)
        self.p = ufl.TrialFunction(self.P_space)
        self.q = ufl.TestFunction(self.P_space)
        self.solution_u = self.V_space.function(name="solution_u")  # all zeros
        self.solution_p = self.P_space.function(name="solution_p")
        self.f = dune.ufl.Constant((0, 0), "f")
        self.u_old = self.V_space.function(name="u_old")
        shape_u = self.u_old.as_numpy[:].shape[0]
        self.p_old = self.P_space.function(name="p_old")
        shape_p = self.p_old.as_numpy[:].shape[0]
        self.element_storage = fem.space.finiteVolume(self.gridView,dimRange=1)
        if self.init_method == "monolithic":
            try:
                self.u_old.as_numpy[:] = (
                    np.load("../initial_velocity"+str(shape_u)+".npy") if comm.rank == 0 else None
                )
                #assign( #apy
                self.p_old.as_numpy[:] = (
                    np.load("../initial_pressure"+str(shape_p)+".npy") if comm.rank == 0 else None
                )
            except Exception as e:
                print(f"No initial condition found, computing quasi-stokes, with error: {e}")
                compute_quasi_stokes(self.rho, self.mu, self, order=self.order) if comm.rank == 0 else None
                self.u_old.as_numpy[:] = (
                    np.load("../initial_velocity"+str(shape_u)+".npy") if comm.rank == 0 else None
                )
                self.p_old.as_numpy[:] = (
                    np.load("../initial_pressure"+str(shape_p)+".npy") if comm.rank == 0 else None
                )
            self.solution_u.as_numpy[:] = self.u_old.as_numpy[:]
            self.solution_p.as_numpy[:] = self.p_old.as_numpy[:]
            vtk = self.gridView.sequencedVTK("initialization", pointdata=[self.solution_u, self.solution_p])
            vtk()
            self.element_storage = fem.space.finiteVolume(self.gridView)
        elif self.init_method == "none" or self.init_method == "increase":
            pass
        else:
            raise Exception("Chosen initialization method does not exist.")
        return self.solution_u.copy(), self.solution_p.copy()

    def calc_curl(self):
        curl = ufl.curl(ufl.as_vector([self.solution_u[0], self.solution_u[1]]))
        #curl = ufl.sqrt(ufl.inner(curl, curl))
        curl = ufl.sqrt(fem.integrate(ufl.dot(curl, curl), self.gridView, order=2))
        curl = self.element_storage.interpolate(curl, name="curl")

        min_curl = comm.min(np.min(np.abs(curl.as_numpy)))
        max_curl = comm.max(np.max(np.abs(curl.as_numpy)))
        if max_curl > min_curl:
            curl.as_numpy[:] -= min_curl
            curl.as_numpy[:] /= max_curl - min_curl
        print(f"max curl: {max_curl}, min curl: {min_curl}")
        return curl

    def generate_navier_stokes_schemes(
        self,
        velocity_boundary_condition,
        pressure_boundary_condition,
        neumann_boundary_form,
    ):
        self.epsilon = lambda w: 1 / 2 * (ufl.nabla_grad(w) + ufl.nabla_grad(w).T)
        self.sigma = -self.p_old * ufl.Identity(
            self.gridView.dimension
        ) + 2 * self.mu * self.epsilon((self.u + self.u_old) / 2.0)
        step_one_form = (
            self.rho * ufl.dot((self.u - self.u_old) / self.dt, self.v) * ufl.dx
            + ufl.inner(self.sigma, self.epsilon(self.v)) * ufl.dx
            - ufl.dot(self.f, self.v) * ufl.dx
            + self.rho
            * ufl.dot(ufl.dot(self.u_old, ufl.nabla_grad(self.u_old)), self.v)
            * ufl.dx
        )
        step_one_form -= neumann_boundary_form

        step_two_form = (
            ufl.dot(ufl.nabla_grad(self.p), ufl.nabla_grad(self.q)) * ufl.dx
            - ufl.dot(ufl.nabla_grad(self.p_old), ufl.nabla_grad(self.q))* ufl.dx
            + rho / dt * ufl.div(self.solution_u) * self.q * ufl.dx
        )

        step_three_form = (
            self.rho * ufl.dot(self.u - self.solution_u, self.v) * ufl.dx
            + self.dt
            * ufl.dot(ufl.nabla_grad(self.solution_p - self.p_old), self.v)
            * ufl.dx
        )
        precon = "none"
        params_verbose = {
            "nonlinear.verbose": True,
            "linear.verbose": True,
            #"linear.tolerance": 1e-6,
            "linear.preconditioning.relaxation": 1.5,
            "linear.maxiterations": 1000,
            "linear.preconditioning.method": precon,
            "linear.petsc.blockedmode": False,
            "linear.errormeasure": "absolute",
            #"logging": "log-ilu", #braucht es gar nicht, linear.verbose: True ist ausreichen. Scheint auch nicht zu helfen residuals zu speichern
        }
        params_no_verbose = {
            "nonlinear.verbose": False,
            "linear.verbose": False,
            #"linear.tolerance": 1e-6,
            "linear.preconditioning.relaxation": 1.5,
            #"linear.maxiterations": 50000,
            "linear.preconditioning.method": precon,
            "linear.petsc.blockedmode": False,
            "linear.errormeasure": "absolute",
            #"logging": "log-ilu", #braucht es gar nicht, linear.verbose: True ist ausreichen. Scheint auch nicht zu helfen residuals zu speichern
        }
        self.step_one_scheme = fem.scheme.galerkin(
            [step_one_form == 0, *velocity_boundary_condition],
            solver=("petsc","gmres"),
            parameters=params_no_verbose,
        )
        
        self.step_two_scheme = fem.scheme.galerkin(
            [step_two_form == 0, *pressure_boundary_condition],
            solver=("petsc","gmres"),
            parameters=params_no_verbose,
        )
        self.step_three_scheme = fem.scheme.galerkin(
            [step_three_form == 0, *velocity_boundary_condition],
            solver=("petsc","gmres"),
            parameters=params_verbose,
        )

    def perform_one_step(self):
        #start_time_step_one = time.time()
        self.step_one_scheme.solve(target=self.solution_u)
        #end_time_step_one = time.time()
        #print()
        #print(end_time_step_one-start_time_step_one)
        # self.vtk()
        start_time_step_two = time.time()
        self.step_two_scheme.solve(target=self.solution_p)
        end_time_step_two = time.time()
        print(end_time_step_two-start_time_step_two)
        self.step_three_scheme.solve(target=self.solution_u)
        self.p_old.as_numpy[:] = self.solution_p.as_numpy[:]
        self.u_old.as_numpy[:] = self.solution_u.as_numpy[:]

    def integrate(self, endTime, time_between_plots=None):
        progress_bar = tqdm(total=endTime, desc="Integration Progress")
        while self.t.value < endTime:
            self.t.value += self.dt.value
            self.perform_one_step()
            progress_bar.update(self.dt.value) if comm.rank == 0 else None

def compute_quasi_stokes(rho, mu, vortex_solver, order=2, H=0.41, L=2.2, r=0.05):
    print("computing quasi-stokes")
    grid = vortex_solver.gridView
    space_u = lagrange(grid, order=order, dimRange=2)
    space_p = lagrange(grid, order=order - 1)
    composite_space = fem.space.composite(
        space_u, space_p, components=["velocity", "pressure"]
    )

    U = ufl.TrialFunction(composite_space)
    V = ufl.TestFunction(composite_space)
    x = ufl.SpatialCoordinate(composite_space)

    u = ufl.as_vector([U[0], U[1]])
    v = ufl.as_vector([V[0], V[1]])
    p = U[2]
    q = V[2]

    epsilon = lambda w: 0.5 * (ufl.nabla_grad(w) + ufl.nabla_grad(w).T)
    sigma = -p * ufl.Identity(2) + 2 * mu * epsilon(u)

    quasi_stokes_form = (
        rho * ufl.dot(u, v) * ufl.dx
        + ufl.inner(sigma, epsilon(v)) * ufl.dx
        + ufl.div(u) * q * ufl.dx
    )

    no_slip_bottom = dune.ufl.DirichletBC(composite_space, [0, 0, None], x[1] < 1e-10)
    no_slip_top = dune.ufl.DirichletBC(composite_space, [0, 0, None], x[1] > H - 1e-10)
    no_slip_cylinder = dune.ufl.DirichletBC(
        composite_space,
        [0, 0, None],
        ufl.sqrt((x[0] - 0.2) ** 2 + (x[1] - 0.2) ** 2) < r + 1e-10,
    )
    velocity_inflow = dune.ufl.DirichletBC(
        composite_space,
        [(6 * x[1] * (H - x[1])) / np.power(H, 2), 0, None],
        x[0] < 1e-10,
    )
    outflow = dune.ufl.DirichletBC(
        composite_space, [None, None, 0], x[0] > L - 1e-10
    )  # outflow condition

    params = {"nonlinear.verbose": False, "linear.verbose": False}

    quasi_stokes_scheme = fem.scheme.galerkin(
        [
            quasi_stokes_form == 0,
            no_slip_bottom,
            no_slip_top,
            no_slip_cylinder,
            velocity_inflow,
            outflow,
        ],
        solver=("petsc", "gmres"),
        parameters=params,
    )

    solution = composite_space.function(name="solution_quasi_stokes")
    quasi_stokes_scheme.solve(target=solution)
    indices_split = vortex_solver.V_space.size
    np.save("../initial_velocity"+str(vortex_solver.u_old.as_numpy[:].shape[0])+".npy", solution.as_numpy[:indices_split])
    np.save("../initial_pressure"+str(vortex_solver.p_old.as_numpy[:].shape[0])+".npy", solution.as_numpy[indices_split:])

order = 2
# wrong hole
# with pygmsh.occ.Geometry() as geom:
#     # Domain size
#     L, H = 2.2, 0.41
#     r = 0.05
#     rectangle = geom.add_rectangle([0.0, 0.0, 0.0], L, H)
#     cylinder = geom.add_disk([0.2, 0.2, 0.0], r)
#     domain = geom.boolean_difference([rectangle], [cylinder])
#     mesh = geom.generate_mesh()
#     points, cells = mesh.points, mesh.cells_dict
#     domain = {
#         "vertices": points[:, :2].astype(float),
#         "simplices": cells["triangle"].astype(int),
#     }

with pygmsh.geo.Geometry() as geom:
    # Domain size
    L, H = 2.2, 0.41
    r = 0.05
    eps=1e-10
    geom.set_mesh_size_callback(lambda dim, tag, x, y, z, lc:
        min(r + 1.5*( (x-0.2)**2+(y-0.2)**2), H / 2.0)
    )
    obstacle = geom.add_circle([0.2, 0.2, 0.0], r, make_surface=False)
    domain = geom.add_rectangle(0, L, 0, H, 0, holes=[obstacle.curve_loop])
    geom.add_physical(domain, 'domain')
    geom.add_physical(obstacle.curve_loop.curves, 'obstacle')
    mesh = geom.generate_mesh(dim=2)
    points, cells = mesh.points, mesh.cells_dict
    boundaryDomains = {
        1: [[0.0-eps, 0.0-eps], [0.0+eps, H+eps]], # bbox inflow
        2: [[L-eps, 0.0-eps], [L+eps, H+eps]], # bbox outflow
        3: [[0.2-r-eps, 0.2-r-eps], [0.2+r+eps, 0.2+r+eps]], # bbox hole
        4: "default", # top and bottom
    }
    dgf = gmsh2DGF(points, cells, bndDomain=boundaryDomains, dim=2)
    obstacleFacets = mesh.cells[0].data[mesh.cell_sets['obstacle'][0]]
    dgf += f'''
Projection
function d(x) = x - {[0.2, 0.2]}
function p(x) = {r} * d(x) / |d(x)| + {[0.2, 0.2]}
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
# help(leafGridView)
vortex_street_grid = (
    leafGridView(domain, dimgrid=2, lbMethod=lb_method)
    if comm.rank == 0
    else leafGridView({"vertices": [], "cubes": []}, dimgrid=2, lbMethod=lb_method)
)
vortex_street_grid = fem.view.adaptiveLeafGridView(vortex_street_grid)
vortex_street_grid.hierarchicalGrid.globalRefine(4)
mu.value = 1e-3
dt.value = 5e-5
t_end = 5e-5
vortex_solver = NavierStokesSolver(vortex_street_grid, order, mu, rho, dt, max_refinement_level=6)
no_slip_bottom = dune.ufl.DirichletBC(
    vortex_solver.V_space, [0, 0], vortex_solver.x_u[1] < 1e-10
)
no_slip_top = dune.ufl.DirichletBC(
    vortex_solver.V_space, [0, 0], vortex_solver.x_u[1] > H - 1e-10
)
no_slip_cylinder = dune.ufl.DirichletBC(
    vortex_solver.V_space,
    [0, 0],
    ufl.sqrt((vortex_solver.x_u[0] - 0.2) ** 2 + (vortex_solver.x_u[1] - 0.2) ** 2)
    < r + 1e-10,
)

pressure_dirichlet_right = dune.ufl.DirichletBC(
    vortex_solver.P_space, [0.0], vortex_solver.x_p[0] > L - 1e-10
)
neumann_boundary_form_right = (
    ufl.dot(
        mu
        * ufl.nabla_grad((vortex_solver.u + vortex_solver.u_old) / 2.0)
        * vortex_solver.n_u,
        vortex_solver.v,
    )
    * ufl.ds
    - ufl.dot(vortex_solver.p_old * vortex_solver.n_p, vortex_solver.v) * ufl.ds
)
if vortex_solver.init_method=="increase":
    increasingInflowDuration = 0.1
    velocity_inflow_boundary = dune.ufl.DirichletBC(
        vortex_solver.V_space,
        [(6 * vortex_solver.x_u[1] * (H - vortex_solver.x_u[1])) / np.pow(H, 2) * ufl.min_value(1.0, vortex_solver.t/increasingInflowDuration), 0],
        vortex_solver.x_u[0] < 1e-10,
    )
else:
    velocity_inflow_boundary = dune.ufl.DirichletBC(
        vortex_solver.V_space,
        [(6 * vortex_solver.x_u[1] * (H - vortex_solver.x_u[1])) / np.pow(H, 2), 0],
        vortex_solver.x_u[0] < 1e-10,
    )

print("init schemes")
vortex_solver.generate_navier_stokes_schemes(
    [no_slip_bottom, no_slip_top, no_slip_cylinder, velocity_inflow_boundary],
    [pressure_dirichlet_right],
    neumann_boundary_form_right,
)
print("integrating")
vortex_solver.integrate(t_end, time_between_plots=0.01)