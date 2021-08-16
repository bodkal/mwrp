import numpy as np
from docplex.mp.model import Model
import docplex.mp.solution as mp_sol
from time import time
from TSP_solver import TSP_solver

#########
#MIN SUM(C*X)

# constraint FOR XI SUM(XI->XJ)==1 (J,I=0->N  , I=/=J) IN AND OUT
# add_indicator FOR i j  x[(i, j)], u[i] + 1 == u[j]

########

n = 5
m=2

D = m

citys = [i for i in range(n)]
dipot = [i for i in range(n,n+D)]

all_cell_location_no_dipot = [(i, j, k) for i in citys for j in citys for k in range(n,n+m) if i != j]
citys_to_dipot = [(i, j, k) for i in citys for j in dipot for k in range(n,n+m) if i != j]
dipot_to_citys = [(i, j, k) for i in dipot for j in citys for k in range(n,n+m) if i != j]
all=all_cell_location_no_dipot+citys_to_dipot+dipot_to_citys


rnd = np.random
rnd.seed(1)

coord_x = rnd.rand(n+D) * 100
coord_y = rnd.rand(n+D) * 100

distance_all_cell_location_no_dipot = {(i, j): np.hypot(coord_x[i] - coord_x[j] , coord_y[i] - coord_y[j]) for i, j, k\
                                       in  all_cell_location_no_dipot}

distance_citys_to_dipot = {(i, j): np.hypot(coord_x[i] - coord_x[j] , coord_y[i] - coord_y[j]) for i, j, k in \
                           citys_to_dipot}

distance_dipot_to_citys = {(i, j): np.hypot(coord_x[i] - coord_x[j] , coord_y[i] - coord_y[j]) for i, j, k in \
                           dipot_to_citys}
all_distance={}
all_distance.update(distance_all_cell_location_no_dipot)
all_distance.update(distance_citys_to_dipot)
all_distance.update(distance_dipot_to_citys)
t=time()



max_d=max(distance_dipot_to_citys.values())
TSP_best_N=TSP_solver(citys, [(i, j) for i in citys for j in citys  if i != j],distance_all_cell_location_no_dipot,coord_x[:n],coord_y[:n])
L_max_UB=TSP_best_N/m+2*max_d

all_tmp_cytis=[(i, j) for i in citys+dipot for j in citys+dipot if i != j]
TSP_best_N_D=TSP_solver(citys+dipot,all_tmp_cytis,{(i, j): np.hypot(coord_x[i] - coord_x[j] , coord_y[i] - coord_y[j]) for i, j\
                                       in  all_tmp_cytis},coord_x,coord_y)
L_max_LB_1=(TSP_best_N_D+TSP_best_N)/m

L_max_LB_2=2*max_d

L_max_LB=max(L_max_LB_1,L_max_LB_2)


while (L_max_UB/L_max_LB-1)>=0.01:
    L_max=(L_max_UB-L_max_LB)/2
    mdl = Model('MD-MTSP')

    x_mid = mdl.binary_var_dict(all_cell_location_no_dipot,name='x')
    x_in = mdl.binary_var_dict(citys_to_dipot,name='x')
    x_out = mdl.binary_var_dict(dipot_to_citys,name='x')

    L_in=mdl.sum(distance_citys_to_dipot[e[:2]]*x_in[e] for e in citys_to_dipot)
    L_out=mdl.sum(distance_dipot_to_citys[e[:2]]*x_out[e] for e in dipot_to_citys)
    L_mid=mdl.sum(distance_all_cell_location_no_dipot[e[:2]]*x_mid[e] for e in all_cell_location_no_dipot)

    L_k=mdl.sum(L_in+L_out+L_mid)

    mdl.minimize(L_k)

    for city in citys:

        out_sum=mdl.sum(x_out[cell] for cell in dipot_to_citys if cell[0] == cell[2] and cell[1]==city)
        mid_sum=mdl.sum(x_mid[cell] for cell in all_cell_location_no_dipot if cell[1] == city)
        mdl.add_constraint(mid_sum + out_sum== 1, ctname='out_%d'%city)

    # for city in citys:
    #
    #     mid_sum = mdl.sum(x_mid[cell[::-1]]-x_mid[cell] for cell in all_cell_location_no_dipot if cell[0] == city)
    #     out_in_sum = mdl.sum(x_out[dipot[::-1]]-x_in[ dipot] for dipot in citys_to_dipot if dipot[0]==city)
    #     mdl.add_constraint(mid_sum + out_in_sum == 0 , ctname=f'repet_{city}')

    for city in dipot:
        out_dipot_sum=mdl.sum(x_out[cell] for cell in dipot_to_citys if cell[0] == cell[2])
        mdl.add_constraint(out_dipot_sum<=1 , ctname=f'out_dipot_{city}')
    print(mdl.export_to_string())


    for city in dipot:
        out_dipot_sum=mdl.sum(x_out[cell]-x_in[(cell[1],cell[0],cell[2])] for cell in dipot_to_citys if cell[0] == cell[2])
        mdl.add_constraint(out_dipot_sum<=1 , ctname=f'out_diff_in_dipot_{city}')

    for city in dipot:
        out_in_dipot_sum=mdl.sum(x_out[cell] - x_in[cell[::-1]] for cell in dipot_to_citys if cell[0] == city)
        mdl.add_constraint(out_in_dipot_sum == 0 , ctname=f'out_in_dipot_{city}')

    for city in dipot:
        L_in = mdl.sum(distance_citys_to_dipot[cell] * x_in[cell] for cell in citys_to_dipot)
        L_out = mdl.sum(distance_dipot_to_citys[cell] * x_out[cell] for cell in dipot_to_citys)
        L_mid = mdl.sum(distance_all_cell_location_no_dipot[cell] * x_mid[cell] for cell in all_cell_location_no_dipot)

        L_k = mdl.sum(L_in + L_out + L_mid)
        mdl.add_constraint(L_k <= L_max, ctname=f'repet_{city}')

    print(mdl.export_to_string())



    solucion = mdl.solve(log_output=False)
    if solucion :
        L_max_LB=L_max
        print(f'solve in  {time()-t} sec')
        #mdl.solve_details
        solucion.display()


        import matplotlib.pyplot as plt

        arcos_activos = [e for e in all_cell_location_no_dipot if x_mid[e].solution_value > 0.9]
        for i,j in arcos_activos:
            plt.plot([coord_x[i],coord_x[j]],[coord_y[i],coord_y[j]],color='b', alpha=0.4, zorder=0)
        arcos_activos = [e for e in citys_to_dipot if x_in[e].solution_value > 0.9]
        for i,j in arcos_activos:
            plt.plot([coord_x[i],coord_x[j]],[coord_y[i],coord_y[j]],color='b', alpha=0.4, zorder=0)
        arcos_activos = [e for e in dipot_to_citys if x_out[e].solution_value > 0.9]
        for i,j in arcos_activos:
            plt.plot([coord_x[i],coord_x[j]],[coord_y[i],coord_y[j]],color='b', alpha=0.4, zorder=0)

        plt.scatter(x=coord_x, y=coord_y, color='r', zorder=1)
        for i in citys:
            plt.annotate(i,(coord_x[i]+1,coord_y[i]+1))

        for i in dipot:
            plt.annotate(i,(coord_x[i]+1,coord_y[i]+1))
        plt.show()


