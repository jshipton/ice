from firedrake import *
import matplotlib.pyplot as plt
import numpy as np
import scipy

n = 16
L = 1

def ice_solve(n=n, L=L, shear=1, A=1., Newtonian=False, pointsolve=False):
    """Solve the Stokes equation for Newtonian flow or the ice flow with or without point_solve"""
    
    # define the mesh
    mesh = PeriodicRectangleMesh(n, n, L, L, direction="x", quadrilateral=True)
    
    # function spaces
    V1 = VectorFunctionSpace(mesh, "CG", 2) # velocity
    V2 = FunctionSpace(mesh, "CG", 1) # pressure
    V3 = TensorFunctionSpace(mesh, "CG", 2) # stress tensor

    # mixed function space
    if pointsolve:
        W = MixedFunctionSpace((V1, V2))
        w, phi = TestFunctions(W)
        soln = Function(W)
        u, p = split(soln)
    else:
        W = MixedFunctionSpace((V1, V2, V3))
        w, phi, z = TestFunctions(W)
        soln = Function(W)
        u, p, tau = split(soln)

    nu = Constant(1.)
    
    nullspace = MixedVectorSpaceBasis(
        W, [W.sub(0), VectorSpaceBasis(constant=True), W.sub(2)])

    # solve for the Newtonian flow
    # tau = nu*grad(u)
    F = div(w)*p*dx - inner(grad(w), tau)*dx - phi*div(u)*dx + inner(z, tau)*dx - nu*inner(z, grad(u))*dx

    # boundary conditions
    upper_bc = Constant((shear, 0.))
    bcs = [DirichletBC(W.sub(0), Constant((0., 0.)), 1),
           DirichletBC(W.sub(0), upper_bc, 2)]

    # solver parameters
    direct_solver = {
        "ksp_type": "preonly", 
        "pc_type": "lu",
        "mat_type": "aij",
        "pc_factor_mat_solver_type": "mumps"}
    
    # solve
    solve(F==0, soln, bcs=bcs, nullspace=nullspace, solver_parameters=direct_solver)

    # Newtonian or not
    if Newtonian:
        u_l, p_l, tau_l = soln.split()
        return u_l, p_l, tau_l
    else:
        if pointsolve:
            ps = point_solve(lambda x, y: inner(x,x)*x - y)
            tau = ps(sym(grad(u)), function_space=V3, shape=(2, 2))
            F = div(w)*p*dx - nu*inner(grad(w), tau)*dx - phi*div(u)*dx
        else:
            # use the soln to linear problem as the initial guess
            # u, p, tau = split(soln)
            A = Constant(A)
            F = div(w)*p*dx - inner(grad(w), tau)*dx - phi*div(u)*dx + inner(tau, tau)*inner(z, tau)*dx -1/A*inner(z, sym(grad(u)))*dx
            solve(F==0, soln, bcs=bcs, nullspace=nullspace, solver_parameters=direct_solver)
    
            u_out, p_out, tau_out = soln.split()
    
        return u_out, p_out, tau_out


def velocity_profile(u, plotfig=True):
    """plot the velocity profile for the flow"""

    # evaluate u at some points
    x = np.linspace(0,1,n+1)
    eval_points = [[0, i] for i in x]
    u_eval = u.at(eval_points)
    speed = [np.linalg.norm(i) for i in u_eval]

    if plotfig:
        plt.figure()
        plt.plot(x, speed, '.')

    return speed



def constitutive_law_plot(A=1., plotfig=True):
    """compute tau for the nonlinear case when varying the shear (upper boundary condition)"""
    upper_bc = np.linspace(1, 100, 10)
    tau_list = []
    for s in upper_bc:
        # compute tau
        u_out, p_out, tau_out = ice_solve(shear=s, A=A, Newtonian=False)
        ## above may diverge when shear is too large given constant A
        ## for example, when A=15 and shear > 82
        # find the square root of \int \tau \cdot \tau dx
        tau_square = assemble(inner(tau_out, tau_out)*dx)
        tau = np.sqrt(tau_square)
        # update
        tau_list.append(tau)

    # make the plot
    if plotfig:
        plt.figure()
        plt.plot(upper_bc, tau_list)

    return upper_bc, np.array(tau_list)
# plotting tau_list**3 (as y-axis) against upper_bc (as x-axis) gives a straight line through (0,0)
# the gradient of the straight line depends on the value of constant A
        

# now investigate how the slope depends on the value of A
def slope_relation(plotfig=True):
    A = np.linspace(1, 10, 10)
    slope_list = []
    for a in A:
        x, y = constitutive_law_plot(A=a, plotfig=False)
        slo, inter, r, p, std = scipy.stats.linregress(x, y**3)
        slope_list.append(slo)

    if plotfig:
        plt.figure()
        plt.plot(A, slope_list)

    return A, np.array(slope_list)
