import numpy as np
from random import randint
from time import time
import matplotlib.pyplot as plt
from matplotlib import colors as c
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall
from docplex.mp.model import Model
from bresenham import bresenham


class Utils:
    '''
    Holds all static functions that do not belong to any specific class
    '''

    @staticmethod
    def get_random_map(row: int, col: int) -> np.array:
        """
        Produces a map of desired size with obstacles in a random location
        :param row: map row size
        :param col: ap row size
        :return: new map white random obstacle
        """
        random_map = np.zeros((row, col))
        random_map[:] = '.'
        random_map[row - 1, :] = '@'
        random_map[0, :] = '@'
        random_map[:, col - 1] = '@'
        random_map[:, 0] = '@'

        for i in range(int(row * col / 10)):
            random_map[randint(1, row - 1), randint(1, col - 1)] = '@'

        return random_map

    # @staticmethod
    # def centrality_dict(dist_dict, dist_map):
    #     centrality_dict = dict()
    #     for index, cell in enumerate(dist_dict):
    #         # tmp_cell_all_dist = sum(cell)
    #         if cell:
    #             centrality_dict[dist_map[index]] = cell
    #     return centrality_dict

    # @staticmethod
    # def centrality_dict_wachers(dist_dict, dist_map):
    #     centrality_dict = dict()
    #     for index, cell in enumerate(dist_dict):
    #         tmp_cell_all_dist = sum(cell)
    #         if tmp_cell_all_dist:
    #             centrality_dict[dist_map[index]] = tmp_cell_all_dist
    #     return centrality_dict

    # @staticmethod
    # def print_pivot(world, pivot):
    #     tmp = np.copy(world.grid_map)
    #     for cell in pivot.keys():
    #         tmp[cell] = 3
    #     # plt.figure(1)
    #     plt.pcolormesh(tmp, edgecolors='grey', linewidth=0.01)
    #     plt.gca().set_aspect('equal')
    #     plt.gca().set_ylim(plt.gca().get_ylim()[::-1])
    # # plt.show()

    # @staticmethod
    # def print_whacer(world, tmp_cell):
    #     for pivot_cell in tmp_cell:
    #         tmp = np.copy(world.grid_map)
    #         for cell in world.dict_wachers[pivot_cell]:
    #             tmp[cell] = 3
    #         tmp[pivot_cell] = 2
    #
    #         # plt.figure(pivot_cell.__str__())
    #         plt.pcolormesh(tmp, edgecolors='grey', linewidth=0.01)
    #         plt.gca().set_aspect('equal')
    #         plt.gca().set_ylim(plt.gca().get_ylim()[::-1])
    #
    #     plt.show()

    @staticmethod
    def print_all_whacers(world: object, whacers_cell: tuple) -> None:
        """
        Displays the map with desired cell and their whacers
        :param world: map object
        :param tmp_cell:
        """
        tmp = np.copy(world.grid_map)

        # Changes the colors of all whacers
        for pivot_cell in whacers_cell:
            for cell_2 in pivot_cell:
                for cell in world.dict_wachers[cell_2]:
                    if not tmp[cell] == 2:
                        tmp[cell] = 3
                tmp[cell_2] = 2

        colors = [(1, 1, 1), (0, 0, 0), (0.2, 0.2, 1), (1, 0, 0), (1, 1, 0), (0, 0.6, 0.05), (0.5, 0.5, 0.5)]
        # white    black       blue            red     yellow      green           grey
        cmap = c.ListedColormap([colors[i] for i in [0, 1, -2, -1]])
        plt.pcolormesh(tmp, cmap=cmap, edgecolors='black', linewidth=0.01)
        plt.gca().set_aspect('equal')
        plt.gca().set_ylim(plt.gca().get_ylim()[::-1])
        plt.show()

    @staticmethod
    def print_map(world: object) -> None:
        """
        Displays only  the map
        :param world: map object
        """
        colors = [(1, 1, 1), (0, 0, 0), (0.2, 0.2, 1), (1, 0, 0), (1, 1, 0), (0, 0.6, 0.05), (0.5, 0.5, 0.5)]
        # white    black       blue            red     yellow      green           grey
        cmap = c.ListedColormap([colors[i] for i in [0, 1]])
        plt.pcolormesh(world.grid_map, cmap=cmap, edgecolors='black', linewidth=0.01)
        plt.gca().set_aspect('equal')
        plt.gca().set_ylim(plt.gca().get_ylim()[::-1])

        plt.show()

    @staticmethod
    def print_serch_status(world: object, node: object, start_time: float,
                           expend: int, genrate: int, move: bool, pivot: list = []) -> None:
        """
        Displays the map with all state
        :param world: map object
        :param node: the new node we need to present
        :param start_time: time froom the start
        :param expend: number of expend node
        :param genrate: number of genrate node
        :param move: flag if move == True The image will change as the algorithm runs
        :param pivot: the pivot cell
        """
        colors = [(1, 1, 1), (0, 0, 0), (0.2, 0.2, 1), (1, 0, 0), (1, 1, 0), (0, 0.6, 0.05), (0.5, 0.5, 0.5)]
        # white    black       blue            red     yellow      green           grey

        tmp = np.copy(world.grid_map)
        plt.rcParams["figure.figsize"] = (11, 7)

        # Paint the cell we have already seen
        seen = set(world.dict_wachers.keys()) - node.unseen
        for cell in seen:
            tmp[cell] = 3

        # Paint the cell with the location of the agents
        for cell in node.location:
            tmp[cell] = 2

        # Paint the cell with the pivot and ther whacers
        if pivot.__len__() > 0:
            cmap = c.ListedColormap([colors[i] for i in [0, 1, -2, -1, 4, 3]])

            for cell in pivot:
                for whacer in world.dict_wachers[cell]:
                    tmp[whacer] = 4
            for cell in pivot:
                tmp[cell] = 5
        else:
            cmap = c.ListedColormap([colors[i] for i in [0, 1, -2, -1]])

        plt.pcolormesh(tmp, cmap=cmap, edgecolors='black', linewidth=0.01)
        plt.gca().set_aspect('equal')
        plt.gca().set_ylim(plt.gca().get_ylim()[::-1])
        # self.col_max,

        plt.text(-4.8, 1, f'time - {round(time() - start_time, 3)} sec')
        plt.text(-4.8, 2, f'Expend - {expend}')
        plt.text(-4.8, 3, f'genarate - {genrate}')
        plt.text(-4.8, 4, f'g - {node.cost}')
        plt.text(-4.8, 5, f'h - {[node.f - i for i in node.cost]}')
        plt.text(-4.8, 6, f'f - {node.f}')
        plt.text(-4.8, 7, f'terminate - {node.dead_agent}')
        plt.text(-4.8, 8, f'coverge - {round(seen.__len__() / world.free_cell * 100, 3)} % ')
        if move == True:
            plt.draw()
            plt.pause(0.001)
            plt.clf()
        else:
            plt.show()

    @staticmethod
    def print_fov(grid_map: list, all_cell: tuple, main_cell: tuple) -> None:
        """
        print the fov of a spsefic cell
        :param grid_map: the map
        :param all_cell:
        :param main_cell:
        """
        tmp = np.copy(grid_map)

        for cell in all_cell:
            if not tmp[cell] == 2:
                tmp[cell] = 3
            tmp[main_cell] = 2

        plt.pcolormesh(tmp, edgecolors='black', linewidth=0.01)
        plt.gca().set_aspect('equal')
        plt.gca().set_ylim(plt.gca().get_ylim()[::-1])
        plt.show()

    @staticmethod
    def map_to_sets(cell: tuple) -> set:
        """
        convert all cell from tupel to set
       :param cell: all cell that need to convert
       :return: converted cell
       """
        return set(map(tuple, [cell]))

    @staticmethod
    def convert_map(map_config: str) -> list:
        """
        convert map from str(@,.) format to int format (1,0)
        :rtype: the converted map
        """
        with open(map_config, newline='') as txtfile:
            row_map = [[0 if cell == '.' else 1 for cell in row[:-1]] for row in txtfile.readlines()]
        return row_map

    @staticmethod
    def sort_list(list_a: list) -> tuple:
        """
        sort a list and retorn the sorted order and the sort list
        :param list_a:
        :return: sorted list , sorted index as dict
        """
        sorted_list_a = sorted(list_a)
        sort_dick = {data: i for i, data in enumerate(sorted(range(len(list_a)), key=list_a.__getitem__))}
        return tuple(sorted_list_a), sort_dick

    # @staticmethod
    # def print_path(all_path, see_agent_walk,world):
    #     tmp_location = []
    #     for cell in all_path:
    #
    #         # print(f'L = {cell.location} \t h = {cell.heuristics} ')
    #         tmp_location.append(cell.location)
    #         tmp_word = np.copy(world.grid_map)
    #         for k in cell.unseen:
    #             tmp_word[k] = 2
    #         for j in cell.location:
    #             tmp_word[j] = 3
    #         if see_agent_walk:
    #             plt.figure(1)
    #             plt.pcolormesh(tmp_word, edgecolors='black', linewidth=0.01)
    #             plt.gca().set_aspect('equal')
    #             plt.gca().set_ylim(plt.gca().get_ylim()[::-1])
    #             plt.draw()
    #             plt.pause(0.001)
    #             plt.clf()
    #             time.sleep(0.5)
    #     plt.close('all')
    #     Utils.print_all_whacers(world, tmp_location)

    # @staticmethod
    # def replace_in_heap(heap, index_in_heap, item):
    #     # Replace in_closed with new_node in open
    #     parent_index = (index_in_heap - 1) >> 1
    #     while index_in_heap != 0:
    #         heap[parent_index], heap[index_in_heap] = (
    #             heap[index_in_heap],
    #             heap[parent_index], )
    #         index_in_heap = parent_index
    #         parent_index = (index_in_heap - 1) >> 1
    #     # in_closed is now at 0
    #     heapq.heapreplace(heap, item)


class Node:

    def __init__(self, parent: object, location: tuple, unseen: set, dead_agent: list,
                 cost: list, minimize: bool, f: int = 0) -> object:
        """
        :param parent: the parent node (this node are genarate from parent node)
        :param location: hold the robots location tupel((x1,y1),(x2,y2),...)
        :param unseen: set of cells that need to see
        :param dead_agent: list of dead agent that doesn't move anymore
        :param cost: the cost untel thes node
        :param minimize: SOC(0) OR MKSP(1)
        :param f: cost + heuristic
        """

        self.parent = parent
        self.location = location
        self.unseen = unseen
        self.cost = cost  # can work only white the cost_map idn if worthe the hedic self._cost_map().values()
        self.f = f
        self.dead_agent = dead_agent
        self.first_genarate = False
        self.cost_map = self._cost_map()

        # lode the cost for the __lt__ baste on the minimize
        if minimize == 0:
            self.get_cost = self.get_max_cost
        else:
            self.get_cost = self.get_sum_cost

    def get_max_cost(self, node: object) -> int:
        """
        max cost for the MKSP wan comparing 2
        :param node: the node we need comper to self
        :return: diff bitween max node and max self
        """
        return max(node.cost) - max(self.cost)

    def get_sum_cost(self, node: object) -> int:
        """
        sum cost for the MKSP wan comparing 2
        :param node: the node we need comper to self
        :return: diff bitween sum node and sum self
        """
        return sum(node.cost) - sum(self.cost)

    def _cost_map(self) -> dict:
        """
        bild a cost map that map spsific cost to spsefic agent
        :return: dict that hold all agent and ther cost
        """

        cost_map = dict()
        for index, cell in enumerate(self.location):
            # handel duplicate location
            if cell not in cost_map:
                cost_map[cell] = [self.cost[index]]
            else:
                cost_map[cell].append(self.cost[index])

        # sort all duplicate location for maintaining the same cost to the same agent
        for cell in cost_map.keys():
            cost_map[cell] = sorted(cost_map[cell])
        return cost_map

    def __lt__(self, oder: object) -> bool:
        """
        compere between 2 nodes for the heapq insert metod
        :param oder: node
        :return: True if need to insert on higher in open False otherwise
        """
        abs_oder_f = abs(oder.f)
        abs_self_f = abs(self.f)
        if abs_self_f == abs_oder_f:
            # first tiebreaker smallest unseen is wining
            if self.unseen.__len__() == oder.unseen.__len__():
                diff_cost = self.get_cost(oder)

                # second tiebreaker bisest cost is wining
                if diff_cost == 0:
                    # Third tiebreaker according to location in memory so as not to be random
                    if id(self) < id(oder):
                        return False
                    else:
                        return True
                elif diff_cost > 0:
                    return False
                else:
                    return True

            elif self.unseen.__len__() > oder.unseen.__len__():
                return False
            else:
                return True
        elif abs_self_f > abs_oder_f:
            return False
        else:
            return True


class Vision:
    def __init__(self, grid_map: list):
        """
        hendel the Vision for the prablom bast on bresenham pkg
        """

        self.grid_map = grid_map

        # bild the obstical that srund the map for the fov
        self.wall_box = []
        for y in range(self.grid_map.shape[1]):
            self.wall_box.append((0, y))
            self.wall_box.append((self.grid_map.shape[0] - 1, y))

        for x in range(1, self.grid_map.shape[0] - 1):
            self.wall_box.append((x, 0))
            self.wall_box.append((x, self.grid_map.shape[1] - 1))

    def _get_line(self, start_cell: tuple, end_cell: tuple) -> set:
        """
        bild the line of site untel he bump in obstical
        :param start_cell: whare the agent stend
        :param end_cell: one of the cell wall
        :return: set of cell that can see from start cell
        """

        all_line = set()
        # rum on all cell between start_cell and end_cell untell he stack at obstecal
        for cell in bresenham(start_cell[0], start_cell[1], end_cell[0], end_cell[1]):
            if self.grid_map[cell] == 0:
                all_line.add(cell)
            else:
                break

        return all_line

    def get_fov(self, start_cell: tuple) -> set:
        """
        run from spsefic cell to all the cell in the wall arund the map
        :param start_cell: whare the agent stend
        :return: fov from spsefic cell
        """
        all_seen = set()
        for end_cell in self.wall_box:
            all_seen = all_seen | self._get_line(start_cell, end_cell)

        return all_seen


class FloydWarshall:

    def __init__(self, grid_map):
        self.free_cell = np.array(np.where(grid_map == 0)).T
        self.free_cell_size = self.free_cell.__len__()

    def _get_cost_grid(self) -> csr_matrix:
        """
        convert the grid mat to csr_matrix (grath that hold the cost bitwine all cells)
        :return: all cost bitwine all cells
        """
        INF = self.free_cell_size ** 2
        cost_map = []
        #bild the list that hold the cost bitwine all cells if no path cost = INF
        for main_cell in self.free_cell:
            cost_map.append([abs(x[0] - main_cell[0]) + abs(x[1] - main_cell[1]) if abs(x[0] - main_cell[0]) + abs(
                x[1] - main_cell[1]) < 2 else INF for x in self.free_cell])

        cost_graph = csr_matrix(cost_map)
        return cost_graph

    def run(self) -> tuple:
        """
        calculate all  dict_dist and centrality_dict
        :return: all distance and centalty for all cells
        """

        # calculate distance
        cost_map = floyd_warshall(csgraph=self._get_cost_grid(), directed=True, return_predecessors=False)
        dict_dist = dict()

        # covert distance from NXN matrix  to dict {((x1,y1),(x2,y2)) : distance}
        for ii in range(self.free_cell_size):
            for jj in range(self.free_cell_size):
                key = tuple(sorted([tuple(self.free_cell[ii]), tuple(self.free_cell[jj])]))
                if key not in dict_dist:
                    dict_dist[key] = cost_map[ii][jj]

        # calculate centrality
        centrality_dict = {tuple(self.free_cell[index]): sum(row) for index, row in enumerate(cost_map)}
        return dict_dist, centrality_dict



class lp_mtsp():

    def __init__(self, agent_number, pivot, distance_dict):
        self.mdl = Model('SD-MTSP')
        self.m = agent_number
        self.mdl.set_time_limit(50)
        self.mdl.parameters.threads = 1  # agent_number//2
        # self.mdl.parameters.threads = 1

        self.mdl.parameters.mip.cuts.flowcovers = 1
        self.mdl.parameters.mip.cuts.mircut = 1
        self.mdl.parameters.mip.strategy.probe = 1
        self.mdl.parameters.mip.strategy.variableselect = 4
        self.mdl.parameters.mip.limits.cutpasses = -1

        # self.mdl.parameters.mip.strategy.presolvenode = 2
        # self.mdl.parameters.mip.strategy.heuristicfreq: 100
        # self.mdl.parameters.mip.strategy.backtrack: 0.1
        # self.distance_out = {(0, i): 0 for i in range(1,self.m+1)}
        # self.distance_in_agent = {(i, 0): 0 for i in range(1,self.m+1)}

        self.k = self.mdl.continuous_var_dict(1, lb=0, name='k')

        # self.u_start_and_agent=self.mdl.continuous_var_dict(range(self.m+1), lb=0,ub=0, name='u_start_and_agent')
        # self.u_pivot=self.mdl.continuous_var_dict(pivot.keys(), lb=0, name='u_pivot')

        self.u_start_and_agent = self.mdl.continuous_var_dict(range(self.m + 1), lb=0, name='u_start_and_agent')
        self.mdl.set_var_ub(self.u_start_and_agent[0], 0)
        # self.u_start_and_agent=self.mdl.continuous_var_dict(range(self.m+1), lb=0,ub=0, name='u_start_and_agent')

        self.u_pivot = self.mdl.continuous_var_dict(pivot.keys(), lb=0, name='u_pivot')

        self.x = self.mdl.binary_var_dict(list(distance_dict.keys()), name='x')

        self.mdl.minimize(self.k[0])

        # print(self.mdl.export_to_string())

    def add_var(self, pivot, distance_dict):

        for i in set(pivot.keys()) - set(self.u_pivot.keys()):
            self.u_pivot = {**self.mdl.continuous_var_dict([i], lb=0, name='u_pivot'), **self.u_pivot}

        for i in set(distance_dict.keys()) - set(self.x.keys()):
            self.x = {**self.mdl.binary_var_dict([i], lb=0, name='x'), **self.x}

    # def add_var1(self, pivot, distance_dict):
    #
    #     for i in pivot - set(self.u_pivot.keys()):
    #         self.u_pivot = {**self.mdl.continuous_var_dict([i], lb=0, name='u_pivot'), **self.u_pivot}
    #
    #     for i in set(distance_dict.keys()) - set(self.x.keys()):
    #         self.x = {**self.mdl.binary_var_dict([i], lb=0, name='x'), **self.x}

    def get_makespan(self, for_plot, w, pivot, n, citys, distance_dict, cost):

        self.mdl.clear_constraints()
        self.add_var(pivot, distance_dict)
        all_directed_edges = list(distance_dict.keys())

        all_u = {**self.u_start_and_agent, **{key: self.u_pivot[key] for key in pivot.keys()}}

        # for u in range(1, self.u_start_and_agent.__len__()):
        #     self.mdl.set_var_lb(self.u_start_and_agent[u], cost[u - 1])
        #     self.mdl.set_var_ub(self.u_start_and_agent[u], cost[u - 1])

        # max_subtoor
        self.mdl.add_constraints_([self.k[0] >= self.u_pivot[c] for c in pivot.keys()])

        # 'out'
        self.mdl.add_constraints_([self.mdl.sum(self.x[(i, j)] for i, j in all_directed_edges if i == c) == 1 for c in
                                   list(all_u.keys())[1:]])

        # in
        self.mdl.add_constraints_([self.mdl.sum(self.x[(i, j)] for i, j in all_directed_edges if j == c) == 1 for c in
                                   list(all_u.keys())[1:]])

        self.mdl.add_constraint_(self.mdl.sum(self.x[(j, 0)] for j in all_u if j != 0) == self.m)

        a, b = zip(*[(self.x[c], all_u[c[1]] == all_u[c[0]] + distance_dict[c])
                     for c in distance_dict.keys() if c[1] != 0])

        self.mdl.add_indicators(a, b, [1] * a.__len__())

        solucion = self.mdl.solve(log_output=False)

        # print(f'solve - {time() -t}')
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

        # self.print_SD_MTSP_on_map(for_plot, all_directed_edges, self.x, w, pivot)

        max_u = solucion.get_objective_value()
        # self.print_SD_MTSP_on_map(for_plot, all_directed_edges, self.x, w, pivot)

        return max([round(max_u)] + cost)

    def get_soc(self, for_plot, w, pivot, n, citys, distance_dict, cost):

        self.mdl.clear_constraints()
        self.add_var(pivot, distance_dict)
        all_directed_edges = list(distance_dict.keys())

        all_u = {**self.u_start_and_agent, **{key: self.u_pivot[key] for key in pivot.keys()}}

        # for u in range(1, self.u_start_and_agent.__len__()):
        #     self.mdl.set_var_lb(self.u_start_and_agent[u], cost[u - 1])
        #     self.mdl.set_var_ub(self.u_start_and_agent[u], cost[u - 1])

        # max_subtoor
        # self.mdl.add_constraint(self.k[0] == self.mdl.sum(self.x[c]*distance_dict[c] for c in all_directed_edges))
        self.mdl.minimize(self.mdl.sum(self.x[c] * distance_dict[c] for c in all_directed_edges))

        # 'out'
        self.mdl.add_constraints_([self.mdl.sum(self.x[(i, j)] for i, j in all_directed_edges if i == c) == 1 for c in
                                   list(all_u.keys())[1:]])

        # in
        self.mdl.add_constraints_([self.mdl.sum(self.x[(i, j)] for i, j in all_directed_edges if j == c) == 1 for c in
                                   list(all_u.keys())[1:]])

        self.mdl.add_constraint_(self.mdl.sum(self.x[(j, 0)] for j in all_u if j != 0) == self.m)

        # a, b = zip(*[(self.x[c], all_u[c[1]] == all_u[c[0]] + distance_dict[c])
        #              for c in distance_dict.keys() if c[0] != 0 and c[1] != 0])
        # self.mdl.add_indicators(a, b, [1] * a.__len__())

        solucion = self.mdl.solve(log_output=False)

        # print(f'solve - {time() -t}')
        # cpx = self.mdl.get_engine().get_cplex()
        # status = cpx.parameters.tune_problem()
        # if status == cpx.parameters.tuning_status.completed:
        #     print("tuned parameters:")
        #     for param, value in cpx.parameters.get_changed():
        #         print("{0}: {1}".format(repr(param), value))
        # else:
        #     print("tuning status was: {0}".format(cpx.parameters.tuning_status[status]))

        if self.mdl.solve_status.name != 'OPTIMAL_SOLUTION':
            print('is not a OPTIMAL_SOLUTION')
            return None

        # print((solucion.display()))
        # self.print_SD_MTSP_on_map(for_plot, all_directed_edges, self.x, w, pivot)

        max_u = solucion.get_objective_value()
        # self.print_SD_MTSP_on_map(for_plot, all_directed_edges, self.x, w, pivot)

        return max_u

    def print_SD_MTSP_on_map(self, for_plot, all_cell_location, best_x, w, p):
        for_plot = [(0, 0)] + for_plot
        import matplotlib.pyplot as plt
        plt.figure()
        arcos_activos = [e for e in all_cell_location if best_x[e].solution_value > 0.9]
        for i, j in arcos_activos:
            if type(i) == int and type(j) == int:
                plt.plot([for_plot[i][1] + 0.5, for_plot[j][1] + 0.5], [for_plot[i][0] + 0.5, for_plot[j][0] + 0.5],
                         color='r', alpha=0.6, linewidth=2, marker='o')
            elif type(i) == int:
                plt.plot([for_plot[i][1] + 0.5, j[1] + 0.5], [for_plot[i][0] + 0.5, j[0] + 0.5], color='r', alpha=0.6,
                         linewidth=2, marker='o')
            elif type(j) == int:
                plt.plot([i[1] + 0.5, for_plot[j][1] + 0.5], [i[0] + 0.5, for_plot[j][0] + 0.5], color='r', alpha=0.6,
                         linewidth=2, marker='o')
            else:
                plt.plot([i[1] + 0.5, j[1] + 0.5], [i[0] + 0.5, j[0] + 0.5], color='r', alpha=0.6, linewidth=2,
                         marker='o')

        # for i, j in for_plot:
        #     plt.annotate(f'{i, j}', xy=(j + 1, i + 0.5), color='k', weight='bold')

        # Utils.print_pivot(w, p)
        # plt.show()
