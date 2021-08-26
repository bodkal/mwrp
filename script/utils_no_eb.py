import numpy as np
from random import randint
from time import time
import matplotlib.pyplot as plt
import csv


class Utils:

    @staticmethod
    def n_dim_distance(point_a, point_b):
        new_vec = point_a - point_b
        return np.sqrt(np.dot(new_vec, new_vec))

    @staticmethod
    def distance_all(point_a, points):
        new_vec = points - point_a
        return np.sqrt(np.einsum('ij,ij->i', new_vec, new_vec))
        # return cdist([point_a], points)

    @staticmethod
    def get_random_map(row, col, start):
        random_map = np.zeros((row, col))
        random_map[row - 1, :] = 1
        random_map[0, :] = 1
        random_map[:, col - 1] = 1
        random_map[:, 0] = 1
        for i in range(int(row * col / 10)):
            random_map[randint(1, row - 1), randint(1, col - 1)] = 1

        for i in start:
            random_map[i] = 0
        return random_map

    @staticmethod
    def centrality_list(dist_dict, grid_map):
        centrality_list = []
        for index, cell in enumerate(dist_dict):
            tmp_cell_all_dist = sum(cell)
            if tmp_cell_all_dist:
                centrality_list.append(tuple((tmp_cell_all_dist, tuple((divmod(index, grid_map.shape[1]))))))
        centrality_list = sorted(centrality_list, reverse=True)
        return centrality_list

    @staticmethod
    def centrality_dict(dist_dict, dist_map):
        centrality_dict = dict()
        for index, cell in enumerate(dist_dict):
            #tmp_cell_all_dist = sum(cell)
            if cell:
                centrality_dict[dist_map[index]] = cell
        return centrality_dict

    @staticmethod
    def centrality_dict_wachers(dist_dict, dist_map):
        centrality_dict = dict()
        for index, cell in enumerate(dist_dict):
            tmp_cell_all_dist = sum(cell)
            if tmp_cell_all_dist:
                centrality_dict[dist_map[index]] = tmp_cell_all_dist
        return centrality_dict

    @staticmethod
    def centrality_tie_see(dist_dict, grid_map, dist_wachers):
        centrality_list = []
        for index, cell in enumerate(dist_dict):
            tmp_cell_all_dist = sum(cell)
            tmp_index = 0
            if tmp_cell_all_dist:
                new_cell = tuple((divmod(index, grid_map.shape[1])))
                for i in range(centrality_list.__len__()):
                    if tmp_cell_all_dist > centrality_list[i][0]:
                        break
                    elif tmp_cell_all_dist == centrality_list[i][0]:
                        if dist_wachers[new_cell].__len__() < dist_wachers[centrality_list[i][1]].__len__():
                            break
                    tmp_index += 1

                centrality_list.insert(tmp_index, tuple((tmp_cell_all_dist, new_cell)))
        return centrality_list

    @staticmethod
    def centrality_tie_wahcers(dist_dict, grid_map, dist_wachers):
        centrality_list = []
        for index, cell in enumerate(dist_dict):
            tmp_cell_all_dist = sum(cell)
            tmp_index = 0
            if tmp_cell_all_dist:
                new_cell = tuple((divmod(index, grid_map.shape[1])))
                for i in range(centrality_list.__len__()):
                    if dist_wachers[new_cell].__len__() < dist_wachers[centrality_list[i][1]].__len__():
                        break
                    elif dist_wachers[new_cell].__len__() < dist_wachers[centrality_list[i][1]].__len__():
                        if tmp_cell_all_dist > centrality_list[i][0]:
                            break
                    tmp_index += 1

                centrality_list.insert(tmp_index, tuple((tmp_cell_all_dist, new_cell)))
        return centrality_list


    @staticmethod
    def centrality_meen_see(dist_wachers):
        centrality_list = []
        for cell in dist_wachers:
            centrality_list.append(tuple((dist_wachers[cell].__len__(), cell)))
        centrality_list = sorted(centrality_list)
        return centrality_list

    @staticmethod
    def print_pivot(world,pivot):
        tmp = np.copy(world.grid_map)
        for cell in pivot.keys():
            tmp[cell] = 3
        #plt.figure(1)
        plt.pcolormesh(tmp, edgecolors='grey', linewidth=0.01)
        plt.gca().set_aspect('equal')
        plt.gca().set_ylim(plt.gca().get_ylim()[::-1])
       # plt.show()

    @staticmethod
    def print_whacer(world, tmp_cell):
        for pivot_cell in tmp_cell:
            tmp = np.copy(world.grid_map)
            for cell in world.dict_wachers[pivot_cell]:
                tmp[cell] = 3
            tmp[pivot_cell] = 2

           # plt.figure(pivot_cell.__str__())
            plt.pcolormesh(tmp, edgecolors='grey', linewidth=0.01)
            plt.gca().set_aspect('equal')
            plt.gca().set_ylim(plt.gca().get_ylim()[::-1])

        plt.show()

    @staticmethod
    def print_all_whacers(world, tmp_cell):
        tmp = np.copy(world.grid_map)

        for pivot_cell in tmp_cell:
            for cell_2 in pivot_cell:
                for cell in world.dict_wachers[cell_2]:
                    if not tmp[cell]==2:
                        tmp[cell] = 3
                tmp[cell_2] = 2


            plt.pcolormesh(tmp, edgecolors='grey', linewidth=0.01)
            plt.gca().set_aspect('equal')
            plt.gca().set_ylim(plt.gca().get_ylim()[::-1])
        plt.draw()
        plt.pause(0.001)
        plt.clf()
        #plt.show()

    @staticmethod
    def print_fov(grid_map,all_cell,main_cell):
        tmp = np.copy(grid_map)

        for cell in all_cell:
            if not tmp[cell]==2:
                tmp[cell] = 3
            tmp[main_cell] = 2

        plt.pcolormesh(tmp, edgecolors='grey', linewidth=0.01)
        plt.gca().set_aspect('equal')
        plt.gca().set_ylim(plt.gca().get_ylim()[::-1])
        plt.show()


    @staticmethod
    def map_to_sets(cell):
        return set(map(tuple,[cell]))

    @staticmethod
    def convert_map(map_config):
        #row_map = []

        with open(map_config, newline='') as txtfile:
            #all_row = []
            row_map=[ [0 if cell == '.' else 1 for cell in row[:-1]] for row in txtfile.readlines()]

        return row_map

class Node:

    def __init__(self, parent, location, unseen,dead_agent,cost=0, heuristics=0):
        self.parent = parent
        self.location = location
        self.unseen = unseen
        self.cost = cost
        self.heuristics = heuristics
        self.dead_agent=dead_agent

    def __sort__(self):
        return tuple(sorted((self.location)))

class Bresenhams:
    def __init__(self, grid_map):
        self.grid_map = grid_map

        self.initial = True
        self.end = True
        self.x0 = 0
        self.x1 = 0
        self.y0 = 0
        self.y1 = 0
        self.dx = 0
        self.dy = 0
        self.sx = 0
        self.sy = 0
        self.err = 0

    def get_next(self):
        if self.initial:
            self.initial = False
            return tuple((self.x0, self.y0))
        if self.x0 == self.x1 and self.y0 == self.y1 or self.grid_map[self.x0, self.y0] == 1:
            self.end = True
            return False

        e2 = 1 * self.err
        if e2 > -self.dy:
            self.err = self.err - self.dy
            self.x0 = self.x0 + self.sx

        if e2 < self.dx:
            self.err = self.err + self.dx
            self.y0 = self.y0 + self.sy

        return tuple((self.x0, self.y0))

    def get_line(self, start_cell, end_cell):
        self.initial = True
        self.end = False
        self.x0 = start_cell[0]
        self.y0 = start_cell[1]
        self.x1 = end_cell[0]
        self.y1 = end_cell[1]

        self.dx = abs(self.x1 - self.x0)
        self.dy = abs(self.y1 - self.y0)

        self.sx = 1 if self.x0 < self.x1 else -1
        self.sy = 1 if self.y0 < self.y1 else -1
        self.err = self.dx - self.dy

        all_seen = []
        while not self.end:
            new_seen = self.get_next()
            if new_seen:
                all_seen.append(new_seen)
        return set(map(tuple, all_seen[:-1]))


class FloydWarshall:

    def __init__(self, grid_map):
        self.grid_map = grid_map
        self.free_cell = np.array(np.where(self.grid_map == 0)).T

        self.row = grid_map.shape[0]
        self.col = grid_map.shape[1]
        self.nV = self.free_cell.__len__()
        self.dict_dist = dict()
        self.INF = (self.row * self.col) * 2

    def from_grid_map_to_cost_map(self):

        cost_map = []
        for main_cell in self.free_cell:
            cost_map.append([abs(x[0] - main_cell[0]) + abs(x[1] - main_cell[1]) if abs(x[0] - main_cell[0]) + abs(
                x[1] - main_cell[1]) < 2 else self.INF for x in self.free_cell])

        return cost_map

    def floyd_warshall(self):
        cost_map = self.from_grid_map_to_cost_map()
        for k in range(self.nV):
            for i in range(self.nV):
                for j in range(self.nV):
                    cost_map[i][j] = min(cost_map[i][j], cost_map[i][k] + cost_map[k][j])

        for ii in range(self.nV):
            for jj in range(self.nV):

                cell_a = tuple(self.free_cell[ii])
                cell_b = tuple(self.free_cell[jj])
                key = tuple(sorted([cell_a, cell_b]))
                if key not in self.dict_dist:
                    self.dict_dist[key] = cost_map[ii][jj]

        centrality_dict = {tuple(self.free_cell[index]) : sum(row) for index, row in enumerate(cost_map)}
        return self.dict_dist, centrality_dict


from docplex.mp.model import Model
import docplex.mp.solution as mp_sol

class lp_mtsp():

    def __init__(self,agent_number,pivot,distance_dict):
        self.mdl = Model('SD-MTSP')
        self.m = agent_number
        self.mdl.set_time_limit(50)
        self.mdl.parameters.threads = 1
        self.mdl.parameters.mip.cuts.flowcovers = 1
        self.mdl.parameters.mip.cuts.mircut = 1
        self.mdl.parameters.mip.strategy.probe = 1
        self.mdl.parameters.mip.strategy.variableselect = 4
        self.mdl.parameters.mip.limits.cutpasses=-1

        #self.mdl.parameters.mip.strategy.presolvenode = 2
        #self.mdl.parameters.mip.strategy.heuristicfreq: 100
        #self.mdl.parameters.mip.strategy.backtrack: 0.1


        # self.distance_out = {(0, i): 0 for i in range(1,self.m+1)}
       # self.distance_in_agent = {(i, 0): 0 for i in range(1,self.m+1)}

        self.k = self.mdl.continuous_var_dict(1, lb=0, name='k')

        # self.u_start_and_agent=self.mdl.continuous_var_dict(range(self.m+1), lb=0,ub=0, name='u_start_and_agent')
        # self.u_pivot=self.mdl.continuous_var_dict(pivot.keys(), lb=0, name='u_pivot')

        self.u_start_and_agent=self.mdl.continuous_var_dict(range(self.m+1), lb=0,ub=0, name='u_start_and_agent')
        self.u_pivot=self.mdl.continuous_var_dict(pivot.keys(), lb=0, name='u_pivot')

        self.x = self.mdl.binary_var_dict(list(distance_dict.keys()), name='x')

        self.mdl.minimize(self.k[0])

        #print(self.mdl.export_to_string())


    def add_var(self,pivot,distance_dict):

        for i in set(pivot.keys())-set(self.u_pivot.keys()):
                self.u_pivot={**self.mdl.continuous_var_dict([i], lb=0, name='u_pivot'),**self.u_pivot}

        for i in set(distance_dict.keys())-set(self.x.keys()):
                self.x={**self.mdl.binary_var_dict([i], lb=0, name='x'),**self.x}

    def get_makespan(self, for_plot, w, pivot, n, citys,distance_dict):
        t=time()
        self.mdl.clear_constraints()

        self.add_var(pivot, distance_dict)
        # print(len(distance_dict))
        # distance_dict = {**self.distance_in_agent,**self.distance_out, **distance_dict}
        # print(len(distance_dict))

        all_directed_edges = list(distance_dict.keys())

        all_u={**self.u_start_and_agent, **{key : self.u_pivot[key] for key in pivot.keys()}}

        #max_subtoor
        self.mdl.add_constraints_([self.k[0]>=self.u_pivot[c] for c in pivot.keys()])

        # 'out'
        self.mdl.add_constraints_([self.mdl.sum(self.x[(i, j)] for i, j in all_directed_edges if i == c) == 1
                                                                                for c in list(all_u.keys())[1:]])
        # in
        self.mdl.add_constraints_([self.mdl.sum(self.x[(i, j)] for i, j in all_directed_edges if j == c) == 1
                                                                                for c in list(all_u.keys())[1:]])

        self.mdl.add_constraint_(self.mdl.sum(self.x[(j, 0)] for j in all_u if j != 0) == self.m)

        a, b = zip(*[(self.x[c], all_u[c[1]] == all_u[c[0]] + distance_dict[c])
                     for c in distance_dict.keys() if c[0] != 0 and c[1] != 0])

        self.mdl.add_indicators(a,b,[1]*a.__len__())
        #print(f'genarate cplax - {time() -t}')
        #t=time()

        solucion = self.mdl.solve(log_output=False)
        #print(f'solve - {time() -t}')

        # cpx = self.mdl.get_engine().get_cplex()
        # status = cpx.parameters.tune_problem()
        # if status == cpx.parameters.tuning_status.completed:
        #     print("tuned parameters:")
        #     for param, value in cpx.parameters.get_changed():
        #         print("{0}: {1}".format(repr(param), value))
        # else:
        #     print("tuning status was: {0}".format(cpx.parameters.tuning_status[status]))


        if self.mdl.solve_status.name != 'OPTIMAL_SOLUTION':
            print('-1')
            return -1

        max_u = solucion.get_objective_value()


        return round(max_u)

    def print_SD_MTSP_on_map(self,for_plot,all_cell_location,best_x,w,p):

        import matplotlib.pyplot as plt
        plt.figure('0')
        arcos_activos = [e for e in all_cell_location if best_x[e].solution_value > 0.9]
        for i, j in arcos_activos:
            plt.plot([for_plot[i][1]+0.5, for_plot[j][1]+0.5], [for_plot[i][0]+0.5, for_plot[j][0]+0.5], color='r', alpha=0.6,linewidth=2, marker='o')
        # for i, j in for_plot:
        #     plt.scatter(j, i, color='r', zorder=1)
        for i in self.citys:
            plt.annotate(i, xy=(for_plot[i][1]+0.5, for_plot[i][0]+0.5),color ='k',weight='bold')
        Utils.print_pivot(w,p)
        plt.show()

    def get_subtoor(self,x,all_cell_location,distance_dict):
        ac0 = []
        ac1 = []
        for e in all_cell_location:
            if x[e].solution_value > 0.9:
                ac0.append(e[0])
                ac1.append(e[1])

        val = []
        while ac1.__len__():
            tmp_x = ac1.pop(0)
            del ac0[0]
            val.append(0)

            while tmp_x and tmp_x in ac0:
                i = ac0.index(tmp_x)
                val[-1] += distance_dict[(ac0[i], ac1[i])]

                tmp_x = ac1[i]
                del ac0[i]
                del ac1[i]
        return val