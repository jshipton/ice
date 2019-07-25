# First I compute the value of A (the constant in Glen's flow law) at temperature 263.15 K
# The I solve the Stokes equation with Newton viscosity and the parameters from the Stokes demo for Firedrake
# This gives the initial guess of eta and tau
# Then I replace the Newton viscosity by tau = eta * epsilon = eta * sym(grad(u))
# Solve (I am getting error message here: TypeError: '<' not supported between instances of 'MeshGeometry' and 'MeshGeometry')(I think it got something to do with eta and sym(grad(u)))
# Update epsilon, eta and tau using the output
# Repeat until eta converges (the difference of new eta and old eta is smaller than some threshold)



import numpy as np
from firedrake import *
import matplotlib.pyplot as plt
from icepack.constants import year, ideal_gas as R, glen_flow_law as n
import icepack
from firedrake.petsc import PETSc

## compute A(T) (the temperature-dependent constant in Glen's flow law)
transition_temperature = 263.15      # K
A0_cold = 3.985e-13 * year * 1.0e18  # mPa**-3 yr**-1
A0_warm = 1.916e3 * year * 1.0e18
Q_cold = 60                          # kJ / mol
Q_warm = 139

T = 263.15 # K
A = icepack.rate_factor(T)
A = Constant(A)

# define the parameters (copied from firedrake stokes tutorial)
parameters = {
    "mat_type": "matfree",
    "ksp_type": "gmres",
    "ksp_monitor_true_residual": None,
    "ksp_view": None,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "diag",
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_pc_type": "python",
    "fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
    "fieldsplit_0_assembled_pc_type": "hypre",
    "fieldsplit_1_ksp_type": "preonly",
    "fieldsplit_1_pc_type": "python",
    "fieldsplit_1_pc_python_type": "firedrake.MassInvPC",
    "fieldsplit_1_Mp_ksp_type": "cg",
    "fieldsplit_1_Mp_pc_type": "none",
    "fieldsplit_1_Mp_mat_type": "matfree"  
}

## initial value for eta and tau
# use the output by solving Stokes equation with Newton viscosity at the moment
n = 16
L = 1
mesh = PeriodicRectangleMesh(n, n, L, L, direction="x", quadrilateral=True)

V1 = VectorFunctionSpace(mesh, "CG", 2)
V2 = FunctionSpace(mesh, "CG", 1)
W = MixedFunctionSpace((V1, V2))

w, phi = TestFunctions(W)
u, p = TrialFunctions(W)
soln = Function(W)
nu = Constant(1.) # newton viscosity

a = -div(w)*p*dx + nu*inner(grad(w), grad(u))*dx + phi*div(u)*dx
L = inner(w, Constant((0,0)))*dx
bcs = [DirichletBC(W.sub(0), Constant((0., 0.)), 1),
       DirichletBC(W.sub(0), Constant((1., 0.)), 2)]
nullspace = MixedVectorSpaceBasis(W, [W.sub(0), VectorSpaceBasis(constant=True)])

soln.assign(0)
solve(a == L, soln, bcs=bcs, nullspace=nullspace, solver_parameters=parameters)

u_out, p_out = soln.split()
## for plotting
#plot(u_out)
#plot(p_out)

# initial guess for eta and tau
epsilon = sym(grad(u_out))
eta = (A*inner(epsilon, epsilon))**(-1/3)
tau = eta * epsilon
print(type(epsilon), type(eta), type(tau))

## now solve Stokes by replacing the Newton viscosity by tau = eta * sym(u)
# similar set up as before
# define a function to do this
def nonnewton_stokes_solver(eta, n=16, L=1):
    """compute u and p on periodic rectangle mesh given tau"""
    # define the mesh
    mesh = PeriodicRectangleMesh(n, n, L, L, direction="x", quadrilateral=True)

    # define the function space
    V1 = VectorFunctionSpace(mesh, "CG", 2)
    V2 = FunctionSpace(mesh, "CG", 1)
    W = MixedFunctionSpace((V1, V2))

    # set up
    w, phi = TestFunctions(W)
    u, p = TrialFunctions(W)
    soln = Function(W)

    # solve
    a = -div(w)*p*dx + eta*inner(grad(w), sym(grad(u)))*dx + phi*div(u)*dx
    L = inner(w, Constant((0,0)))*dx 
    bcs = [DirichletBC(W.sub(0), Constant((0., 0.)), 1),
           DirichletBC(W.sub(0), Constant((1., 0.)), 2)]

    # add null space for pressure
    nullspace = MixedVectorSpaceBasis(W, [W.sub(0), VectorSpaceBasis(constant=True)])

    soln.assign(0)
    solve(a == L, soln, bcs=bcs, nullspace=nullspace, solver_parameters=parameters)

    u_out1, p_out1 = split(soln) # for updating tau
    u_out2, p_out2 = soln.split() # for plotting

    return u_out1, p_out1, u_out2, p_out2

# try the function
u_out1, p_out1, u_out2, p_out2 = nonnewton_stokes_solver(eta)

## keep updating tau until converge
#max_iter = 5
#tolerance = 0.1
#i = 0
#diff_eta = 1
#
#while i < max_iter and diff_eta > tolerance:
#    # solve for u and p
#    u_out, p_out = nonnewton_stokes_solver(eta)
#    # compute epsilon
#    epsilon = sym(grad(u_out))
#    # compute eta
#    eta_new = (A*inner(epsilon,epsilon))**(-1/3)
#    # update tau
#    tau = eta * epsilon
#    # update iteration criteria
#    i += 1
#    diff_eta = norms.errornorm(eta_new-eta, eta_new-eta) # not sure 
#    eta = eta_new
#   
#print(i) # check number of iterations
#print(diff_tau)
    

