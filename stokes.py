from firedrake import *
import matplotlib.pyplot as plt

n = 16
L = 1
mesh = PeriodicRectangleMesh(n, n, L, L, direction="x", quadrilateral=True)

V1 = VectorFunctionSpace(mesh, "CG", 2)
V2 = FunctionSpace(mesh, "CG", 1)
W = MixedFunctionSpace((V1, V2))

w, phi = TestFunctions(W)
u, p = Function(W).split()
soln = Function(W)
nu = Constant(1.)

F = div(w)*p*dx - nu*inner(grad(w), grad(u))*dx - phi*div(u)*dx
solve(F == 0, soln)
u_out, p_out = soln.split()
plot(p_out)
plt.show()
