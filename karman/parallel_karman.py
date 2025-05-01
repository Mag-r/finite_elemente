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

mu = dune.ufl.Constant(1, "mu")
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
        self.setup_space()
        self.vtk = self.gridView.sequencedVTK(
            "karman", pointdata=[self.solution_u, self.solution_p]
        )
        self.max_refinement_level = max_refinement_level
        dune.fem.loadBalance([self.solution_u, self.solution_p, self.u_old, self.p_old])
        print(self.gridView.size(0))
        self.vtk()

    def setup_space(self):
        self.V_space = lagrange(
            self.gridView, order=self.order, dimRange=self.gridView.dimension
        )
        # pressure space
        self.P_space = lagrange(self.gridView, order=self.order - 1)
        self.x_u = ufl.SpatialCoordinate(self.V_space)
        self.n_u = ufl.FacetNormal(self.P_space)
        self.u = ufl.TrialFunction(self.V_space)
        self.v = ufl.TestFunction(self.V_space)
        self.x_p = ufl.SpatialCoordinate(self.P_space)
        self.n_p = ufl.FacetNormal(self.P_space)
        self.p = ufl.TrialFunction(self.P_space)
        self.q = ufl.TestFunction(self.P_space)
        self.solution_u = self.V_space.function(name="solution_u")
        self.solution_p = self.P_space.function(name="solution_p")
        self.f = dune.ufl.Constant((0, 0), "f")
        self.u_old = self.V_space.function(name="u_old")
        self.p_old = self.P_space.function(name="p_old")
        self.u_old.as_numpy[:] = (
            np.load("initial_velocity.npy") if comm.rank == 0 else None
        )
        self.p_old.as_numpy[:] = (
            np.load("initial_pressure.npy") if comm.rank == 0 else None
        )
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
            curl.as_numpy[:] /= max_curl - min_curl
        fem.mark(
            curl,
            refineTolerance=0.75,
            coarsenTolerance=0.05,
            maxLevel=self.max_refinement_level,
            minLevel=2,
        )
        fem.adapt([self.solution_u, self.solution_p, self.u_old, self.p_old])

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
            - (
                ufl.dot(ufl.nabla_grad(self.p_old), ufl.nabla_grad(self.q))
                + rho / dt * ufl.div(self.solution_u) * self.q
            )
            * ufl.dx
        )
        
        step_three_form = (
            self.rho * ufl.dot(self.u - self.solution_u, self.v) * ufl.dx
            + self.dt
            * ufl.dot(ufl.nabla_grad(self.solution_p - self.p_old), self.v)
            * ufl.dx
        )
        params = {
            "nonlinear.verbose": False,
            "linear.verbose": False,
        }
        self.step_one_scheme = fem.scheme.galerkin(
            [step_one_form == 0, *velocity_boundary_condition],
            solver="gmres",
            parameters=params,
        )
        self.step_two_scheme = fem.scheme.galerkin(
            [step_two_form == 0, *pressure_boundary_condition],
            solver="gmres",
            parameters=params,
        )
        self.step_three_scheme = fem.scheme.galerkin(
            [step_three_form == 0, *velocity_boundary_condition],
            solver="gmres",
            parameters=params,
        )

    def perform_one_step(self):
        self.step_one_scheme.solve(target=self.solution_u)
        self.step_two_scheme.solve(target=self.solution_p)
        self.step_three_scheme.solve(target=self.solution_u)
        self.p_old.as_numpy[:] = self.solution_p.as_numpy
        self.u_old.as_numpy[:] = self.solution_u.as_numpy

    def integrate(self, endTime, time_between_plots=None):
        next_plot_time = 0
        progress_bar = tqdm(total=endTime, desc="Integration Progress")
        while self.t.value < endTime:
            self.t.value += self.dt.value
            self.perform_one_step()
            progress_bar.update(self.dt.value) if comm.rank == 0 else None
            if time_between_plots is not None and self.t.value >= next_plot_time:
                next_plot_time += time_between_plots
                self.vtk()
                # self.adapt()
                # dune.fem.loadBalance([self.solution_p, self.solution_u, self.u_old, self.p_old])


def compute_quasi_stokes(rho, vortex_solver, order=2, H=0.41, L=2.2, r=0.05):
    assert comm.size == 1, "Quasi-stokes solver only works in serial"
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
        solver="gmres",
        parameters=params,
    )

    solution = composite_space.function(name="solution_quasi_stokes")
    quasi_stokes_scheme.solve(target=solution)
    indices_split = vortex_solver.V_space.size
    np.save("initial_velocity.npy", solution.as_numpy[:indices_split])
    np.save("initial_pressure.npy", solution.as_numpy[indices_split:])
    


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
    domain = {
        "vertices": points[:, :2].astype(float),
        "simplices": cells["triangle"].astype(int),
    }
lb_method = 9
vortex_street_grid = (
    leafGridView(domain, dimgrid=2, lbMethod=lb_method)
    if comm.rank == 0
    else leafGridView({"vertices": [], "cubes": []}, dimgrid=2, lbMethod=lb_method)
)
vortex_street_grid = fem.view.adaptiveLeafGridView(vortex_street_grid)
vortex_street_grid.hierarchicalGrid.globalRefine(2)

mu.value = 1e-3
dt.value = 5e-5
vortex_solver = NavierStokesSolver(vortex_street_grid, order, mu, rho, dt)
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
velocity_inflow_boundary = dune.ufl.DirichletBC(
    vortex_solver.V_space,
    [(6 * vortex_solver.x_u[1] * (H - vortex_solver.x_u[1])) / np.pow(H, 2), 0],
    vortex_solver.x_u[0] < 1e-10,
)

compute_quasi_stokes(rho, vortex_solver)
print("init schemes")
vortex_solver.generate_navier_stokes_schemes(
    [no_slip_bottom, no_slip_top, no_slip_cylinder, velocity_inflow_boundary],
    [pressure_dirichlet_right],
    neumann_boundary_form_right,
)
print("integrating")
vortex_solver.integrate(t_end, time_between_plots=0.001)
