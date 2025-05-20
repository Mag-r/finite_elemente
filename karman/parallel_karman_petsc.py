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
original_dir = os.getcwd()
output_dir = os.path.join(original_dir, "karman/petsc_"+str(comm.size)+"p")
os.makedirs(output_dir, exist_ok=True)
os.chdir(output_dir)
import functools
print = functools.partial(print, flush = True) if comm.rank==0 else lambda *a, **kw: None


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
        # self.gridView.plot() if comm.rank == 0 else None
        # self.gridView.hierarchicalGrid.loadBalance()  # cannot be feed forwarded, because then solution_u is computed on each of the domains -> wrong boundary values???
        # fails without loadbalance
        # does same as self.gridView.hierarchichalGrid.loadBalance?
        # dune.fem.loadBalance(self.gridView.hierarchicalGrid)
        self.setup_space()
        self.max_refinement_level = max_refinement_level
        self.gridView.hierarchicalGrid.loadBalance()  # changes self.solution_u???
        # dune.fem.loadBalance([self.solution_u, self.solution_p, self.u_old, self.p_old])
        #print("before")
        #print(numpy.nonzero(self.solution_u.as_numpy))
        #dune.fem.loadBalance(self.gridView.hierarchicalGrid)
        #print("after")
        #print(numpy.nonzero(self.solution_u.as_numpy))
        self.curl = self.calc_curl()
        self.vtk = self.gridView.sequencedVTK(
            "karman", pointdata=[self.solution_u, self.solution_p, self.curl],
        )
        print(self.gridView.size(0))
        self.vtk()  # stores image

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
        self.p_old = self.P_space.function(name="p_old")
        self.element_storage = fem.space.finiteVolume(self.gridView,dimRange=1)
        if self.init_method == "uzawa":
            self.u_old.as_numpy[:], self.p_old.as_numpy[:] = compute_quasi_stokes_uzawa(self.rho, self.mu, self)
        elif self.init_method == "monolithic":
            try:
                self.u_old.as_numpy[:] = (
                    np.load("../initial_velocity.npy") if comm.rank == 0 else None
                )
                #assign( #apy
                self.p_old.as_numpy[:] = (
                    np.load("../initial_pressure.npy") if comm.rank == 0 else None
                )
            except Exception as e:
                print(f"No initial condition found, computing quasi-stokes, wih error: {e}")
                compute_quasi_stokes(self.rho, self.mu, self, order=self.order) if comm.rank == 0 else None
                self.u_old.as_numpy[:] = (
                    np.load("initial_velocity.npy") if comm.rank == 0 else None
                )
                self.p_old.as_numpy[:] = (
                    np.load("initial_pressure.npy") if comm.rank == 0 else None
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

    def adapt(self):
        curl = self.calc_curl()
        [refined, coarsened] = fem.mark(
            indicator=curl,
            refineTolerance=0.075,
            coarsenTolerance=0.01,
            maxLevel=self.max_refinement_level,
            minLevel=2,
            markNeighbors=True,
            gridView=self.gridView,
        )
        print(f"element counts:{self.gridView.size(0), self.solution_u.size}")
        print(
            f"Refining {refined} cells, coarsening {coarsened} cells")
        fem.adapt([self.solution_u, self.solution_p, self.u_old, self.p_old])

    def calc_curl(self):
        curl = ufl.curl(ufl.as_vector([self.solution_u[0], self.solution_u[1]]))
        curl = ufl.sqrt(ufl.inner(curl, curl))
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
        params = {
            "nonlinear.verbose": False,
            "linear.verbose": False,
            # "linear.tolerance": 1e-6,
            "linear.preconditioning.method": "hypre",
        }
        self.step_one_scheme = fem.scheme.galerkin(
            [step_one_form == 0, *velocity_boundary_condition],
            solver=("petsc","gmres"),
            parameters=params,
        )
        self.step_two_scheme = fem.scheme.galerkin(
            [step_two_form == 0, *pressure_boundary_condition],
            solver=("petsc","gmres"),
            parameters=params,
        )
        self.step_three_scheme = fem.scheme.galerkin(
            [step_three_form == 0, *velocity_boundary_condition],
            solver=("petsc","gmres"),
            parameters=params,
        )

    def perform_one_step(self):
        self.step_one_scheme.solve(target=self.solution_u)
        self.step_two_scheme.solve(target=self.solution_p)
        self.step_three_scheme.solve(target=self.solution_u)
        self.p_old.as_numpy[:] = self.solution_p.as_numpy[:]
        self.u_old.as_numpy[:] = self.solution_u.as_numpy[:]

    def integrate(self, endTime, time_between_plots=None):
        next_plot_time = 0
        progress_bar = tqdm(total=endTime, desc="Integration Progress")
        while self.t.value < endTime:
            self.t.value += self.dt.value
            self.perform_one_step()
            progress_bar.update(self.dt.value) if comm.rank == 0 else None
            
            if time_between_plots is not None and self.t.value >= next_plot_time:
                #print()
                #print(self.t.value/increasingInflowDuration)
                next_plot_time += time_between_plots
                self.curl.as_numpy[:] = self.calc_curl().as_numpy[:]
                self.vtk()
                self.adapt()
                dune.fem.loadBalance([self.solution_p, self.solution_u, self.u_old, self.p_old])


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
    np.save("initial_velocity_4p.npy", solution.as_numpy[:indices_split])
    np.save("initial_pressure_4p.npy", solution.as_numpy[indices_split:])

class Scheme1:
    def __init__(self, scheme, u0):
        self.scheme = scheme
        self.jacobian = scheme.linear(ubar=u0) # obtain a linear operator for the Newton method

    def solve(self, target):
        # create a copy of target for the residual
        res = target.copy(name="residual")
        dh  = target.copy(name="direction")
        n, linIter = 0,0
        while True:
            # res = S[u]
            #     = u - g on boundary
            self.scheme(target, res)
            absF = res.scalarProductDofs(res)
            if absF < 1e-7**2: # this is the same tolerance we set above for the built-in Newton solver
                break
            self.scheme.jacobian(target,self.jacobian)  # assemble the linearization
            # dh = DS[u]^{-1}S[u]
            #    = u - g on boundary
            dh.clear()
            info = self.jacobian.solve(target=dh,rightHandSide=res)
            linIter += info["linear_iterations"]
            # unew = u - DS[u]^{-1}S[u]
            #      = u - (u-g) = g on boundary
            target -= dh
            n += 1
        return {"iterations":n, "linear_iterations":linIter}
    
def compute_quasi_stokes_uzawa(rho, mu, vortex_solver, order=2, H=0.41, L=2.2, r=0.05):
    grid = vortex_solver.gridView
    dim = grid.dimension
    spcU = lagrange(grid, dimRange=grid.dimension, order=order)
    spcP = lagrange(grid, order=order-1)

    u = ufl.TrialFunction(spcU)
    v = ufl.TestFunction(spcU)
    p = ufl.TrialFunction(spcP)
    q = ufl.TestFunction(spcP)

    x = ufl.SpatialCoordinate(spcU)
    exact_u = ufl.as_vector( [x[1] * (1.-x[1]), 0] )
    f = ufl.as_vector( [0,]*dim)
    f += mu*exact_u

    epsilon = lambda w: 0.5 * (ufl.nabla_grad(w) + ufl.nabla_grad(w).T)
    sigma = -p * ufl.Identity(2) + 2 * mu * epsilon(u)

    # boundary conditions
    no_slip_bottom = dune.ufl.DirichletBC(spcU, [0, 0], x[1] < 1e-10)
    no_slip_top = dune.ufl.DirichletBC(spcU, [0, 0], x[1] > H - 1e-10)
    no_slip_cylinder = dune.ufl.DirichletBC(
        spcU,
        [0, 0],
        ufl.sqrt((x[0] - 0.2) ** 2 + (x[1] - 0.2) ** 2) < r + 1e-10,
    )
    velocity_inflow = dune.ufl.DirichletBC(
        spcU,
        [(6 * x[1] * (H - x[1])) / np.power(H, 2), 0],
        x[0] < 1e-10,
    )
    outflow = dune.ufl.DirichletBC(
        spcP, [0], x[0] > L - 1e-10
    )
    #dbc = dune.ufl.DirichletBC(spcU, exact_u)

    A_model = rho * ufl.dot(u,v) * ufl.dx + mu * ufl.inner(ufl.grad(u) + ufl.grad(u).T, ufl.grad(v)) * ufl.dx - ufl.dot(f,v) * ufl.dx
    grad_model = -ufl.inner(p*ufl.Identity(grid.dimension), ufl.grad(v)) * ufl.dx
    B_model = - ufl.inner(q, ufl.div(u)) * ufl.dx
    precondition_model = ufl.inner(ufl.grad(p), ufl.grad(q)) * ufl.dx
    mass_model = p * q * ufl.dx

    A_op = dune.fem.operator.galerkin([A_model, no_slip_bottom, no_slip_top, no_slip_cylinder, velocity_inflow])
    grad_operator = dune.fem.operator.galerkin([grad_model, no_slip_bottom, no_slip_top, no_slip_cylinder, velocity_inflow])
    B_op = dune.fem.operator.galerkin([B_model, outflow])
    precondition_op = dune.fem.operator.galerkin((precondition_model, dune.ufl.DirichletBC(spcP, 0)),spcP)
    mass_op = dune.fem.operator.galerkin(mass_model)

    A = A_op.linear()
    A = A.as_numpy
    G = grad_operator.linear()
    G = G.as_numpy
    B = B_op.linear()
    B = B.as_numpy
    precondition = precondition_op.linear()
    precondition = precondition.as_numpy
    mass_op = mass_op.linear()
    mass_op = mass_op.as_numpy

    velocity = spcU.interpolate([0,0], name = "velocity")
    pressure = spcP.interpolate(0, name = "pressure")
    sol_u = velocity.as_numpy
    sol_p = pressure.as_numpy
    rhsVelo  = velocity.copy()
    rhsPress = pressure.copy()
    rhs_u  = rhsVelo.as_numpy
    rhs_p  = rhsPress.as_numpy
    r2      = np.zeros_like(rhs_p)
    precon = np.zeros_like(rhs_p)
    chi = np.zeros_like(rhs_u)

    A_op(velocity, rhsVelo) # für randwerte
    # rhsVelo.plot()
    rhs_u *= -1
    A_op.setConstraints(rhsVelo)
    sol_u[:] = linalg.spsolve(A, rhs_u) #u = A^-1 * (F-B*p) // but p= 0 //  3.99a
    
    # trial for parallelization
    # scheme = dune.fem.scheme.galerkin(A == rhs_u, solver='cg',  # sind A_model und rhsVelo falsch gewählt? Alle anderen Möglichkeiten auch error
    #               parameters={"linear.preconditioning.method":"jacobi",
    #                           "nonlinear.forcing":"eisenstatwalker",
    #                           # this is the default for this forcing
    #                           "linear.errormeasure":"residualreduction",
    #                           "nonlinear.tolerance":1e-7},
    #              )
    # scheme_cls = Scheme1(scheme,u0=velocity)
    # info = scheme_cls.solve(target=velocity)

    rhs_p[:] = B * sol_u # 3.99b, rhs_p = B*u
    r2 = linalg.spsolve(mass_op, rhs_p) # 3.99b, M * r = rhs_p
    precon[:] = linalg.spsolve(precondition, rhs_p) # 3.99c, P * precon = rhs_p
    r2[:]= mu*r2 + precon * rho # 3.99d r = mu * r + rho * precon
    d = np.copy(r2) # 3.99e, d = r
    delta = np.dot(r2,rhs_p) # 3.99f, delta = mass * r * mass^-1 *rhs_p

    iterations = 0
    
    while delta > 1e-8 and iterations < 1000:
        rhs_u[:] = G * d # 3.99g, rhs_u = G * d
        chi[:] = linalg.spsolve(A, rhs_u) # 3.99g, chi = A^-1 * rhs_u
        rhs_p[:] = B * chi # wie oben 3.99b
        scale = - delta/np.dot(d, rhs_p) # 3.99h, rho = - delta / (d * B * chi)
        sol_p -= scale * d # 3.99i, p = p - rho * d
        sol_u += scale * chi # 3.99i, u = u + rho * chi
        rhs_p[:] = B * sol_u # wie oben 3.99b
        r2 = linalg.spsolve(mass_op, rhs_p) # 3.99k // wie oben 3.99b
        precon[:] = linalg.spsolve(precondition, rhs_p) # 3.99k // wie oben 3.99c
        r2[:] = mu*r2 + precon * rho # wie oben 3.99d
        delta_new = np.dot(r2, rhs_p) # 3.99n, delta_new = r * M * r
        gamma = delta_new / delta # 3.99o, gamma = delta_new / delta
        delta = delta_new
        d[:] = r2 + gamma * d # 3.99p, d = r + gamma * d
        print(f"error at iteration {iterations}: {delta}")
        iterations += 1
    vtk = grid.sequencedVTK("initialization", pointdata=[velocity, pressure])
    vtk()
    print()
    return sol_u, sol_p


order = 2
t_end = 10
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
vortex_solver = NavierStokesSolver(vortex_street_grid, order, mu, rho, dt,max_refinement_level=6)
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
