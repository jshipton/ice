from firedrake import *
import matplotlib.pyplot as plt

n = 16
L = 1
mesh = PeriodicRectangleMesh(n, n, L, L, direction="x", quadrilateral=True)

V1 = VectorFunctionSpace(mesh, "CG", 2)
V2 = FunctionSpace(mesh, "CG", 1)
V3 = TensorFunctionSpace(mesh, "DG", 0)
W = MixedFunctionSpace((V1, V2, V3))

w, phi, z = TestFunctions(W)
# u, p = Function(W).split()
soln = Function(W)
u, p, tau = split(soln)
nu = Constant(1.)

# tau  = grad(u)

Newtonian=False
pointsolve=True
if Newtonian:
    F = div(w)*p*dx - nu*inner(grad(w), tau)*dx - phi*div(u)*dx + inner(z, tau)*dx - inner(z, grad(u))*dx
else:
    if pointsolve:
        ps = point_solve(lambda x, y: inner(x,x)*x - y)
        tau = ps(sym(grad(u)), function_space=V3, shape=(2, 2))
        F = div(w)*p*dx - nu*inner(grad(w), tau)*dx - phi*div(u)*dx
    else:
        F = div(w)*p*dx - nu*inner(grad(w), tau)*dx - phi*div(u)*dx + inner(z, tau)*dx - inner(z, sym(grad(u)))*dx
bcs = [DirichletBC(W.sub(0), Constant((0., 0.)), 1),
       DirichletBC(W.sub(0), Constant((1., 0.)), 2)]

solve(F == 0, soln, bcs=bcs)

u_out, p_out, tau_out = soln.split()
plot(p_out)
plt.show()
plot(u_out)
plt.show()
