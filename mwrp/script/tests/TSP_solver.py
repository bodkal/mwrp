from docplex.mp.model import Model
import docplex.mp.solution as mp_sol


def TSP_solver(citys,all_cell_location,distance_dict,coord_x,coord_y):

    mdl = Model('TSP')

    x = mdl.binary_var_dict(all_cell_location, name='x')
    u = mdl.continuous_var_dict(citys, name='u')

    mdl.minimize(mdl.sum(distance_dict[e] * x[e] for e in all_cell_location))
    for c in citys:
        mdl.add_constraint(mdl.sum(x[(i, j)] for i, j in all_cell_location if i == c) == 1, ctname='out_%d' % c)

    for c in citys:
        mdl.add_constraint(mdl.sum(x[(i, j)] for i, j in all_cell_location if j == c) == 1, ctname='in_%d' % c)

    for i, j in all_cell_location:
        if j != 0:
            mdl.add_indicator(x[(i, j)], u[i] + 1 == u[j], name='order_(%d,_%d)' % (i, j))

    solucion = mdl.solve(log_output=False)

    import matplotlib.pyplot as plt

    arcos_activos = [e for e in all_cell_location if x[e].solution_value > 0.9]
    for i, j in arcos_activos:
        plt.plot([coord_x[i], coord_x[j]], [coord_y[i], coord_y[j]], color='b', alpha=0.4, zorder=0)
    plt.scatter(x=coord_x, y=coord_y, color='r', zorder=1)
    for i in citys:
        plt.annotate(i, (coord_x[i] + 1, coord_y[i] + 1))
    plt.show()

    return solucion.objective_value

# print(f'solve in  {time()-t} sec')
# mdl.solve_details
# solucion.display()
#
# import matplotlib.pyplot as plt
#
# arcos_activos = [e for e in all_cell_location if x[e].solution_value > 0.9]
# for i,j in arcos_activos:
#     plt.plot([coord_x[i],coord_x[j]],[coord_y[i],coord_y[j]],color='b', alpha=0.4, zorder=0)
# plt.scatter(x=coord_x, y=coord_y, color='r', zorder=1)
# for i in citys:
#     plt.annotate(i,(coord_x[i]+1,coord_y[i]+1))
# plt.show()


# tour = [0]
# while len(tour) < n:
#     k = tour[-1]
#     new_dist = {(i, j): d for (i,j), d in distance_dict.items() if i==k and j not in tour}
#     (i, j) = min(new_dist, key=new_dist.get)
#     tour.append(j)
# tour


# sol_inicial = mp_sol.SolveSolution(mdl)
# # for c in x.keys():
# #     sol_inicial.add_var_value(x[c], 1)
#
# for g in range(n):
#     i = tour[g - 1]
#     j = tour[g]
#     sol_inicial.add_var_value(x[(i,j)], 1)
# print(sol_inicial)


# mdl.add_mip_start(sol_inicial)
# mdl.parameters.timelimit = 60
# mdl.parameters.mip.tolerances.mipgap = 0.2
# mdl.parameters.mip.strategy.branch = 1
