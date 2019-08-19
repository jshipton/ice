from firedrake import *
# import matplotlib.pyplot as plt
# import numpy as np
# import scipy


n = 16
L = 1

# define the mesh
mesh = PeriodicRectangleMesh(n, n, L, L, direction="x", quadrilateral=True)
    
# function spaces
V1 = VectorFunctionSpace(mesh, "CG", 2) # velocity
V2 = FunctionSpace(mesh, "CG", 1) # pressure
V3 = TensorFunctionSpace(mesh, "CG", 2) # stress tensor

# the constants
nu = Constant(1.)
A = Constant(1.)

W = MixedFunctionSpace((V1, V2))
w, phi = TestFunctions(W)
soln = Function(W)
u, p = split(soln)

#nullspace = MixedVectorSpaceBasis(
#        W, [W.sub(0), VectorSpaceBasis(constant=True)])

# boundary conditions
bcs = [DirichletBC(W.sub(0), Constant((0., 0.)), 1),
       DirichletBC(W.sub(0), Constant((1., 0.)), 2)]

Newtonian = True

if Newtonian:
    pe = point_expr(lambda x: nu*grad(x))
    tau = pe(u, function_space=V3)

    #pe = point_expr(lambda x: nu*x)
    #tau = pe(grad(u), function_space=V3)

else:
    ps = point_solve(lambda x, y: inner(x,x)*x - y)
    #ps = point_solve(lambda x, y: abs(dot(x, x))*x-y)
    
    tau = ps(sym(grad(u)), function_space=V3)


F = div(w)*p*dx - inner(grad(w), tau)*dx - phi*div(u)*dx

solve(F==0, soln, bcs=bcs)

u_out, p_out = soln.split()
