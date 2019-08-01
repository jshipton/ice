from firedrake import *
import matplotlib.pyplot as plt
import numpy as np

def ice_solve(n=16, L=1, Newtonian=False, pointsolve=False):
    """Solve the Stokes equation for Newtonian flow or the ice flow with or without point_solve"""
    
    # define the mesh
    mesh = PeriodicRectangleMesh(n, n, L, L, direction="x", quadrilateral=True)
    x, y = SpatialCoordinate(mesh)
    
    # function spaces
    V0 = FunctionSpace(mesh,"CG",3)
    V1 = VectorFunctionSpace(mesh, "CG", 2) # velocity
    V2 = FunctionSpace(mesh, "CG", 1) # pressure
    V3 = TensorFunctionSpace(mesh, "DG", 0) # stress tensor

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
        
    # Newtonian or not
    if Newtonian: # tau = nu*grad(u)
        F = div(w)*p*dx - nu*inner(grad(w), tau)*dx - phi*div(u)*dx
        + inner(z, tau)*dx - inner(z, grad(u))*dx
    else:
        if pointsolve:
            ps = point_solve(lambda x, y: inner(x,x)*x - y)
            tau = ps(sym(grad(u)), function_space=V3, shape=(2, 2))
            F = div(w)*p*dx - nu*inner(grad(w), tau)*dx - phi*div(u)*dx
        else:
            F = div(w)*p*dx - nu*inner(grad(w), tau)*dx - phi*div(u)*dx
            + inner(z, tau)*dx - inner(z, sym(grad(u)))*dx

    # boundary conditions
    bcs = [DirichletBC(W.sub(0), Constant((0., 0.)), 1),
           DirichletBC(W.sub(0), Constant((1., 0.)), 2)]

    # solve
    solve(F==0, soln, bcs=bcs)
    u_out, p_out, tau_out = soln.split()
    
    return u_out, p_out, tau_out


def velocity_profile(u, plotfig=True):
    """plot the velocity profile for the flow"""

    # evaluate u at some points
    x = np.linspace(0,1,17)
    eval_points = [[0, i] for i in x]
    u_eval = u.at(eval_points)
    speed = [np.linalg.norm(i) for i in u_eval]

    if plotfig:
        # plot
        plt.plot(x, speed, '.')

    return speed
