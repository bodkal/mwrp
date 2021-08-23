import numpy as np
from docplex.mp.model import Model
import docplex.mp.solution as mp_sol
from time import time


n = 7
m=3
p=[1,2,3]
n=n+p.__len__()

# k=2
L=9999999

citys = [i for i in range(n)]

ind=[0]*citys[1:].__len__()

all_cell_location_no_start = [(i, j) for i in citys for j in citys if i != j and i !=0 and j != 0]

#all_cell_location = list(zip([0]*m,p))+list(zip(range(1,n),[0]*(n-1)))+all_cell_location_no_start

all_cell_location = [(i, j) for i in citys for j in citys if i != j]

rnd = np.random
rnd.seed(1)
coord_x = rnd.rand(n) * 100
coord_y = rnd.rand(n) * 100

distance_dict = {(i, j): np.hypot(coord_x[i] - coord_x[j] , coord_y[i] - coord_y[j]) if (i!=0 and j!=0) else 0 for i, j in all_cell_location}

t=time()
mdl = Model('SD-MTSP')

x = mdl.binary_var_dict(all_cell_location,name='x')
u = mdl.continuous_var_list(citys,name='u')

mdl.minimize(mdl.sum(distance_dict[e]*x[e] for e in all_cell_location))

for c in citys[1:]:
    mdl.add_constraint(mdl.sum(x[(i,j)] for i,j in all_cell_location if i == c) == 1, ctname='out_%d'%c)
    mdl.add_constraint(mdl.sum(x[(i,j)] for i,j in all_cell_location if j == c) == 1, ctname='in_%d'%c)

for c in p:
    mdl.add_constraint(x[(0,c)] == 1, ctname='out_dipot_%d'%c)

#mdl.add_constraint(mdl.sum(x[(0,j)] for j in range(n) if j != 0) == m, ctname='out_start')
mdl.add_constraint(mdl.sum(x[(j,0)] for j in range(n) if j != 0) == m, ctname='in_start')

for i, j in all_cell_location:
    if j!=0:
        mdl.add_indicator(x[(i, j)], u[j] - u[i] >= distance_dict[(i, j)] , name='order_(%d,_%d)'%(i, j))


for i in citys[1:]:
        ind[i-citys[1]] = mdl.add_indicator(x[(i, 0)], u[i] <= L, name=f'HB_2_{i}')

solucion = mdl.solve(log_output=False)
max_u = solucion.get_all_values()[-citys.__len__():]


#print(solucion.display())

# t1=time()


import matplotlib.pyplot as plt
plt.figure('0')
arcos_activos = [e for e in all_cell_location if x[e].solution_value > 0.9]
for i, j in arcos_activos:
    plt.plot([coord_x[i], coord_x[j]], [coord_y[i], coord_y[j]], color='b', alpha=0.4, zorder=0)
plt.scatter(x=coord_x, y=coord_y, color='r', zorder=1)
for i in citys:
    plt.annotate(i, (coord_x[i], coord_y[i]))
#plt.show()

# mdl.remove_constraints(ind)
# for i in citys[1:]:
#     ind[i - citys[1]] = mdl.add_indicator(x[(i, 0)], u[i] <= L, name=f'HB_2_{i}')

# # print(mdl.export_to_string())

print(f' min_sum  {solucion.objective_value} HB : {L}  min_max : {max(max_u)}')

solucion = mdl.solve(log_output=False)
L = max(max_u) - 1


index=1
while solucion:
    mdl.remove_constraints(ind)
    for i in citys[1:]:
        ind[i - citys[1]] = mdl.add_indicator(x[(i, 0)], u[i] <= L, name=f'HB_2_{i}')
    solucion = mdl.solve(log_output=False)

    # solucion.display()
    if solucion:
        plt.figure(index.__str__())
        index+=1
        arcos_activos = [e for e in all_cell_location if x[e].solution_value > 0.9]
        for i, j in arcos_activos:
            plt.plot([coord_x[i], coord_x[j]], [coord_y[i], coord_y[j]], color='b', alpha=0.4, zorder=0)
        plt.scatter(x=coord_x, y=coord_y, color='r', zorder=1)
        for i in citys:
            plt.annotate(i, (coord_x[i], coord_y[i]))
        max_u = solucion.get_all_values()[-citys.__len__():]
        # print(f'solve in  {L} sec len of : #{max(max_u)}')

        print(f' min_sum  {solucion.objective_value} HB : {L}  min_max : {max(max_u)}')
        L = max(max_u) - 1

plt.show()

#print(f'solve in  {time()-t} sec len of : {max(max_u)}')





















