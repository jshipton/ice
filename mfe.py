from firedrake import *
import matplotlib.pyplot as plt

n = 16
L = 1
mesh = PeriodicRectangleMesh(n, n, L, L, direction="x", quadrilateral=True)

V1 = VectorFunctionSpace(mesh, "CG", 2)
V3 = TensorFunctionSpace(mesh, "DG", 0)

w = TestFunction(V1)
u = Function(V1)

f = inner(grad(w), sym(grad(u)))*dx

a = ufl_expr.derivative(f, u)
assemble(a)

ps = point_solve(lambda x, y: inner(x,x)*x - y)
tau = ps(sym(grad(u)), function_space=V3, shape=(2, 2))

f = inner(grad(w), tau)*dx

a = ufl_expr.derivative(f, u)
assemble(a)
