from firedrake import *
import matplotlib.pyplot as plt
import numpy as np

n = 16
L = 1

mesh = PeriodicRectangleMesh(n, n, L, L, direction="x", quadrilateral=True)

V1 = VectorFunctionSpace(mesh, "CG", 2)
V2 = FunctionSpace(mesh, "CG", 1)
V3 = TensorFunctionSpace(mesh, "CG", 2) # replace DG,0 by CG, 2

W = MixedFunctionSpace((V1, V2, V3))
w, phi, z = TestFunctions(W)
soln = Function(W)
u, p, tau = split(soln)

nu = Constant(1.)

nullspace = MixedVectorSpaceBasis(
    W, [W.sub(0), VectorSpaceBasis(constant=True), W.sub(2)])

bcs = [DirichletBC(W.sub(0), Constant((0., 0.)), 1),
       DirichletBC(W.sub(0), Constant((1., 0.)), 2)]

direct_solver = {
    "ksp_type": "preonly", 
    "pc_type": "lu",
    "mat_type": "aij",
    "pc_factor_mat_solver_type": "mumps"}


F1 =  div(w)*p*dx - nu*inner(grad(w), grad(u))*dx - phi*div(u)*dx + inner(z, tau)*dx - inner(z, grad(u))*dx

solve(F1==0, soln, bcs=bcs, nullspace=nullspace, solver_parameters=direct_solver)

u1, p1, tau1 = soln.split()
plot(u1)
plt.show()


# replace nu*grad(u) by tau
F2 =  div(w)*p*dx - inner(grad(w), tau)*dx - phi*div(u)*dx + inner(z, tau)*dx - inner(z, grad(u))*dx

solve(F2==0, soln, bcs=bcs, nullspace=nullspace, solver_parameters=direct_solver)

u2, p2, tau2 = soln.split()
