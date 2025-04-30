
from dune.grid import cartesianDomain
from dune.alugrid import aluConformGrid as leafGridView
from dune.fem.space import lagrange
import dune
from dune import fem
import numpy as np
import ufl
from tqdm import tqdm
from matplotlib import pyplot as plt
import pygmsh
from dune.common import comm


mu  = dune.ufl.Constant(1, "mu")
rho = dune.ufl.Constant(1, "rho")
dt = dune.ufl.Constant(0.02, "dt")

class NavierStokesSolver:
    def __init__(self, gridView, order, mu, rho, dt, max_refinement_level=5):
        self.gridView = gridView
        self.order = order
        self.mu = mu
        self.rho = rho
        self.dt = dt
        self.t = dune.ufl.Constant(0, name="time")
        # self.gridView.plot() if comm.rank == 0 else None
        self.init()
        self.vtk = self.gridView.sequencedVTK("karman", pointdata=[self.solution_u, self.solution_p])
        self.max_refinement_level = max_refinement_level
        self.gridView.hierarchicalGrid.loadBalance()
        dune.fem.loadBalance([self.solution_u, self.solution_p, self.u_old, self.p_old])
        print(self.gridView.size(0))


    def init(self):
        self.V_space = lagrange(self.gridView, order=self.order, dimRange=self.gridView.dimension) 
    # pressure space
        self.P_space = lagrange(self.gridView, order=self.order-1)
        self.x_u = ufl.SpatialCoordinate(self.V_space)
        self.n_u = ufl.FacetNormal(self.P_space)
        self.u = ufl.TestFunction(self.V_space)
        self.v = ufl.TrialFunction(self.V_space)
        self.x_p = ufl.SpatialCoordinate(self.P_space)
        self.n_p = ufl.FacetNormal(self.P_space)
        self.p = ufl.TestFunction(self.P_space)
        self.q = ufl.TrialFunction(self.P_space)
        self.solution_u = self.V_space.function(name="solution_u")
        self.solution_p = self.P_space.function(name="solution_p")
        self.f = dune.ufl.Constant((0, 0), "f")
        self.u_old = self.V_space.function(name="u_old")
        self.p_old = self.P_space.function(name="p_old")
        self.u_old.as_numpy[:] = np.load("initial_velocity.npy") if comm.rank == 0 else None
        self.p_old.as_numpy[:] = np.load("initial_pressure.npy") if comm.rank == 0 else None
        self.solution_u.as_numpy[:] = self.u_old.as_numpy[:]
        self.solution_p.as_numpy[:] = self.p_old.as_numpy[:]
        self.element_storage = fem.space.finiteVolume(self.gridView)

    def adapt(self):
        curl = ufl.curl(ufl.as_vector([self.solution_u[0], self.solution_u[1]]))
        curl = ufl.inner(curl, curl)
        curl = self.element_storage.interpolate(curl, name="curl")
        min_curl = comm.min(np.min(curl.as_numpy))
        max_curl = comm.max(np.max(curl.as_numpy))
        if max_curl > min_curl:
            curl.as_numpy[:] -= min_curl
            curl.as_numpy[:] /= (max_curl - min_curl)
        fem.mark(curl,refineTolerance=0.75, coarsenTolerance=0.05, maxLevel=self.max_refinement_level, minLevel=2)
        fem.adapt([self.solution_u, self.solution_p, self.u_old, self.p_old])

    def generate_navier_stokes_schemes(self, velocity_boundary_condition, pressure_boundary_condition, neumann_boundary_form):
        self.epsilon = lambda w:  1/2*(ufl.nabla_grad(w) + ufl.nabla_grad(w).T)
        self.sigma = -self.solution_p*ufl.Identity(self.gridView.dimension) + 2*self.mu*self.epsilon(self.u)
        step_one_form = (
        rho * ufl.dot((self.u - self.u_old) / self.dt,self. v) * ufl.dx
        + ufl.inner(self.sigma, self.epsilon(self.v)) * ufl.dx
        - ufl.inner(self.f, self.v) * ufl.dx
        + self.rho * ufl.inner(ufl.dot(self.u_old, ufl.nabla_grad(self.u_old)), self.v) * ufl.dx 
        )
        step_one_form -= neumann_boundary_form
        step_two_form = ufl.dot(ufl.grad(self.p), ufl.grad(self.q)) * ufl.dx - (ufl.dot(ufl.grad(self.p_old),ufl.grad(self.q))- rho/dt * ufl.div(self.u_old) * self.q) * ufl.dx

        step_three_form = ufl.inner(self.u,self.v) * ufl.dx - (ufl.inner(self.u_old,self.v) - self.dt *ufl.inner(ufl.grad(self.solution_p-self.p_old),self.v))* ufl.dx
        params = {"nonlinear.verbose": False,
                  "linear.verbose": False,
        }
        self.step_one_scheme = fem.scheme.galerkin([step_one_form == 0, *velocity_boundary_condition], solver="gmres", parameters=params)
        self.step_two_scheme = fem.scheme.galerkin([step_two_form == 0, *pressure_boundary_condition], solver="gmres", parameters=params)
        self.step_three_scheme = fem.scheme.galerkin([step_three_form == 0, *velocity_boundary_condition], solver="gmres", parameters=params)
        

    def perform_one_step(self):
        self.step_one_scheme.solve(target=self.solution_u)
        self.u_old.as_numpy[:] = self.solution_u.as_numpy
        self.step_two_scheme.solve(target=self.solution_p)
        self.p_old.as_numpy[:] = self.solution_p.as_numpy
        self.step_three_scheme.solve(target=self.solution_u)
        self.u_old.as_numpy[:] = self.solution_u.as_numpy
        
    def integrate(self, endTime, time_between_plots = None):
        next_plot_time = 0
        progress_bar = tqdm(total=endTime, desc="Integration Progress")
        while self.t.value < endTime:
            self.t.value += self.dt.value
            self.perform_one_step()
            progress_bar.update(self.dt.value) if comm.rank == 0 else None
            if time_between_plots is not None and self.t.value >= next_plot_time:
                next_plot_time += time_between_plots
                self.vtk()
                self.adapt()
                dune.fem.loadBalance([self.solution_p, self.solution_u, self.u_old, self.p_old])
                
def compute_quasi_stokes(rho, vortex_solver, no_slip_bottom, no_slip_top, no_slip_cylinder, velocity_inflow_boundary):
    print("computing quasi-stokes")
    epsilon = lambda w:  1/2*(ufl.nabla_grad(w) + ufl.nabla_grad(w).T)
    sigma = -vortex_solver.p*ufl.Identity(vortex_solver.gridView.dimension) + 2*vortex_solver.mu*epsilon(vortex_solver.u)
    quasi_stokes = rho*ufl.dot(vortex_solver.u, vortex_solver.v)*ufl.dx + ufl.inner(sigma, epsilon(vortex_solver.v))*ufl.dx + ufl.div(vortex_solver.u)*vortex_solver.q*ufl.dx
    params = {"nonlinear.verbose": False,
                  "linear.verbose": True,
                  "linear.preconditing.method": "sor",
        }
    quasi_stokes_scheme = fem.scheme.galerkin([quasi_stokes == 0, no_slip_top, no_slip_bottom, no_slip_cylinder, velocity_inflow_boundary], solver = "gmres", parameters = params)
    solution_quasi_stokes = vortex_solver.compositeTaylorHoodSpace.function(name="solution_quasi_stokes")
    quasi_stokes_scheme.solve(target=solution_quasi_stokes)

    vortex_solver.u_old.as_numpy[:] = solution_quasi_stokes.as_numpy[:vortex_solver.indices_split]
    vortex_solver.p_old.as_numpy[:] = solution_quasi_stokes.as_numpy[vortex_solver.indices_split:]
    np.save("initial_velocity.npy", vortex_solver.u_old.as_numpy[:vortex_solver.indices_split])
    np.save("initial_pressure.npy", vortex_solver.p_old.as_numpy[:])

      
order = 2
t_end = 1  
with pygmsh.occ.Geometry() as geom:
    # Domain size
    L, H = 2.2, 0.41
    r = 0.05
    rectangle = geom.add_rectangle([0.0, 0.0, 0.0], L, H)
    cylinder = geom.add_disk([0.2, 0.2, 0.0], r)
    domain = geom.boolean_difference([rectangle], [cylinder])
    mesh = geom.generate_mesh()
    points, cells = mesh.points, mesh.cells_dict
    domain = {"vertices": points[:,:2].astype(float),
              "simplices": cells["triangle"].astype(int)}
lb_method = 9
vortex_street_grid = leafGridView(domain, dimgrid=2, lbMethod = lb_method) if comm.rank == 0 else leafGridView({"vertices":[], "cubes":[]}, dimgrid=2, lbMethod = lb_method)
vortex_street_grid = fem.view.adaptiveLeafGridView(vortex_street_grid)
vortex_street_grid.hierarchicalGrid.globalRefine(2)

mu.value = 1E-3
dt.value = 5E-5
vortex_solver = NavierStokesSolver(vortex_street_grid, order, mu, rho, dt)
no_slip_bottom = dune.ufl.DirichletBC(vortex_solver.V_space, [0, 0], vortex_solver.x_u[1] < 1e-10)
no_slip_top = dune.ufl.DirichletBC(vortex_solver.V_space, [0, 0], vortex_solver.x_u[1] > H - 1e-10)
no_slip_cylinder = dune.ufl.DirichletBC(vortex_solver.V_space, [0, 0], ufl.sqrt((vortex_solver.x_u[0] - 0.2)**2 + (vortex_solver.x_u[1] - 0.2)**2) < r + 1e-10)

pressure_dirichlet_right = dune.ufl.DirichletBC(vortex_solver.P_space, [0.0], vortex_solver.x_p[0] > L - 1e-10)
neumann_boundary_form_right = ufl.inner(
        ufl.dot(ufl.nabla_grad(vortex_solver.u).T, vortex_solver.n_u)*vortex_solver.mu - vortex_solver.solution_p * vortex_solver.n_u, vortex_solver.v
    ) * ufl.conditional(ufl.gt(vortex_solver.x_u[0], L - 1e-10), 1, 0) * ufl.ds
velocity_inflow_boundary = dune.ufl.DirichletBC(vortex_solver.V_space, [(6 * vortex_solver.x_u[1]*(H-vortex_solver.x_u[1]))/np.pow(H,2), 0], vortex_solver.x_u[0] < 1e-10)

print("init schemes")
vortex_solver.generate_navier_stokes_schemes([no_slip_bottom, no_slip_top, no_slip_cylinder, velocity_inflow_boundary], [pressure_dirichlet_right], neumann_boundary_form_right)
print("integrating")
vortex_solver.integrate(t_end, time_between_plots=0.001)
