import pyomo.environ as pyo

model = pyo.ConcreteModel()

model.x = pyo.Var([1,2], domain=pyo.NonNegativeReals)

model.OBJ = pyo.Objective(expr = 2*model.x[1] + 3*model.x[2])

model.Constraint1 = pyo.Constraint(expr = 3*model.x[1] + 4*model.x[2] >= 1)

opt = pyo.SolverFactory('glpk')
opt.solve(model) 

print('x1 = ', pyo.value(model.x[1]), ' x2 = ', pyo.value(model.x[2]), ' obj = ', pyo.value(model.OBJ))