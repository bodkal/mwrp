import numpy as np
from script.utils import Node, Utils, FloydWarshall, lp_mtsp
from script.world import WorldMap
import pickle
import matplotlib.pyplot as plt
from operator import add
import ast
import itertools
from time import time, sleep

from random import randint


class Mwrp:

    def __init__(self, world: WorldMap, start_pos: tuple, huristic_index: int, max_pivot: int, map_name: str, minimize: int) -> None:
        """
        @param world:           An WorldMap  object from script.world that contains the parameters of the world
        @param start_pos:       The starting position of the agents as tupel . For example for two agents : ((1,1),(4,6))
        @param huristic_index:  heuristic activated in this session ( {0: 'singlton', 1: 'max', 2: 'mtsp',3 : 'laze max',4:'BFS'})
        @param max_pivot:       The maximum number of pivots for calculating heuristics . From experience 5 and 6 were the best
        @param map_name:        The name of the map. For example : maze_11_11
        @param minimize:        What we want to minimize (soc, mksp)
        """
        self.huristic_index = huristic_index
        self.minimize = minimize
        self.start_time = 0
        self.number_of_agent = start_pos.__len__()
        self.world = world

        # we do not want to limit the number of pivot now we well limit it later in the code
        self.max_pivot = max_pivot * 10

        # Need it only for the experiments
        self.node_expend_index = 1
        self.genrate_node = 0
        self.expend_node = 0
        self.H_genrate = 0
        self.H_expend = 0
        self.open_is_beter = 0
        self.new_is_beter = 0

        # set which contains all the free cell on the map
        unseen_all = set(map(tuple, np.transpose(np.where(self.world.grid_map == 0))))

        # Filters all the cells that can be seen from the starting position
        unseen_start = unseen_all - self.world.get_all_seen(start_pos)

        # Produces the initial NODE
        start_node = Node(Node(None, start_pos, [0] * self.number_of_agent, 0, [0] * start_pos.__len__()), start_pos,
                          unseen_start, [], [0] * start_pos.__len__())

        # open sort from top to bottom (best down)
        self.open_list = [start_node]

        # open and close list
        self.visit_list_dic = {tuple(sorted(start_pos)): [start_node]}

        # Checks whether we have previously calculated the distance between all the cell on the map and the centrality
        try:
            self.real_dis_dic = pickle.load(open(f"config/real_dis_dic_{map_name}.p", "rb"))
            self.centrality_dict = pickle.load(open(f"config/centrality_dict_{map_name}.p", "rb"))
        except:
            # calculated the distance between all the cell on the map and the centrality and save to file
            print('start FloydWarshall')
            self.real_dis_dic, self.centrality_dict = FloydWarshall(self.world.grid_map).floyd_warshall()
            print('end FloydWarshall')
            pickle.dump(self.real_dis_dic, open(f"config/real_dis_dic_{map_name}.p", "wb"))
            pickle.dump(self.centrality_dict, open(f"config/centrality_dict_{map_name}.p", "wb"))

        # Arranges the centrality_dict according to a given function and filters what you see from the starting point
        self.centrality_dict = self.centrality_list_wachers(unseen_start)

        # Pre-calculate suspicious points that will be PIVOT to save time during the run
        self.pivot = self.get_pivot(unseen_start)
        self.old_pivot = {tuple(sorted((i, j))): self.get_closest_wachers(i, j) for i in self.pivot.keys()
                          for j in self.pivot.keys() if i != j}

        # limit the number of pivot
        self.max_pivot = max_pivot

        # Calculate all the distances between the PIVOT to calculate heuristics The goal is to try to understand if
        # there are PIVOT points that lower the value of the heuristics and filter them
        # TODO  not the best method requires improvement
        distance_pivot_pivot = {(i, j): self.get_closest_wachers(i, j) for j in self.pivot.keys()
                                for i in self.pivot.keys() if i != j}
        distance_in_pivot = {(i, 0): 0 for i in list(self.pivot.keys()) + list(range(1, self.number_of_agent + 1))}
        distance_agent_pivot = {(i, j): 0 for j in self.pivot for i in range(1, self.number_of_agent + 1)}
        all_dist = {**distance_in_pivot, **distance_agent_pivot,
                    **{(0, i): 0 for i in range(1, self.number_of_agent + 1)},
                    **distance_pivot_pivot}

        # Initializes the CPLEX object
        self.lp_model = lp_mtsp(self.number_of_agent, self.pivot, all_dist)

        # List of PIVOT that lower the heuristic
        self.pivot_black_list = self.get_pivot_black_list(start_node)

        # Need it only for the experiments
        if self.huristic_index == 3:
            self.H_start = 0
        else:
            self.H_start = self.get_heuristic(start_node)

    def goal_test(self, unseen: set) -> bool:
        """
        Checking for a solution to a problem (unseen is empty)
        :param unseen: the node unseen list (set)
        :return:  True if unseen is empty (bool)
        """
        if not unseen.__len__():
            return True
        return False

    def get_pivot_black_list(self, start_node: tuple) -> set:
        """
        Checking List of pivot that lower the heuristic
        :param start_node:  the start node (Node)
        :return: List of pivot that lower the heuristic (set)
        """
        old_u = 0
        pivot_black_list = []

        # get all unseen cell sort by the centrality . lower is in the center
        tmp_pivot_key_sorted = sorted([(self.centrality_dict[key], key) for key in self.pivot.keys()])
        tmp_pivot_key = [val[1] for val in tmp_pivot_key_sorted]

        # Runs on all PIVOT points and saves the points that cause the heuristics to go down until he
        # finds no more such points
        for i in range(self.pivot.__len__()):
            # calculate the heuristics and each time remove different PIVOT
            tmp_u = [self.mtsp_makespan_heuristic_start(start_node,
                    {k: self.pivot[k] for k in tmp_pivot_key if k != cell}) for cell in tmp_pivot_key]

            # Finds the index with the highest heuristics because it means we downloaded this
            # PIVOT went up the heuristics
            index_min = np.argmax(tmp_u)
            u_max = max(tmp_u)

            # If the heuristic has dropped from a previous calculation then there are no additional
            # PIVOT points that can be removd without affect the heuristic
            if (old_u >= u_max):
                break
            else:
                # Save the problematic PIVOT and remove it from the list
                old_u = u_max
                bad_index = tmp_pivot_key[index_min]
                pivot_black_list.append(bad_index)
                del tmp_pivot_key[index_min]

        return set(pivot_black_list)

    def centrality_list_wachers(self, unseen: set) -> dict:
        """
        sort the unseen based on the each cell centrality wachers , number of wachers and cell centrality higher it is the better
        :param unseen: the unseen set of a node
        :return: Dictionary of each cell centrality
        """
        centrality_dict=dict()
        #list_sort_by_centrality = []
        # Goes through every cell in unseen and And calculates its centrality
        for index, cell in enumerate(unseen):

            # sum all the cell wachers centrality
            centrality_value=sum(self.centrality_dict[cell_whacers] for cell_whacers in self.world.dict_wachers[cell])

            # calculate all cells centrality
            centrality_dict[cell] = centrality_value / (self.world.dict_wachers[cell].__len__() ** 2) * self.centrality_dict[cell]

        return centrality_dict

    def get_centrality_list_wachers(self, unseen: set) -> list:
        """
        returt sorted list by centrality sort from the bottom up
        :param unseen: the unseen set of a node
        :return: sorted list of each cell centrality
        """

        # create sorted list of each cell centrality based on unseen
        list_sort_by_centrality = sorted([(self.centrality_dict[cell], cell) for cell in unseen])

        return list_sort_by_centrality

    def get_min_list_wachers(self, unseen: set) -> list:
        """
             returt sorted list by centrality sort from the bottom up
             :param unseen: the unseen set of a node
             :return: sorted list of each cell number wachers
             """

        # create sorted list of each cell number wachers based on unseen
        list_sort_by_centrality = sorted([(len(self.world.dict_wachers[cell]), cell) for cell in unseen],reverse=True)
        return list_sort_by_centrality


    def get_pivot(self, unseen: set) -> dict:
        """
        calculate the pivot location for the mtsp huristic
        :param unseen: the unseen set of a node
        :return: Dictionary of selected pivot
        """

        pivot = dict()
        remove_from_unseen_set = set()

        # TODO stop here ------------!!!!!!!!!!!!!!!!!!!!!
        sort_unseen = self.get_centrality_list_wachers(unseen)

        # sort_unseen = self.get_min_list_wachers(unseen)


        while sort_unseen.__len__():
            cell = sort_unseen.pop()

            if not self.world.dict_wachers[cell[1]].intersection(remove_from_unseen_set):
                pivot[cell[1]] = self.world.dict_wachers[cell[1]]
                remove_from_unseen_set = remove_from_unseen_set | self.world.dict_wachers[cell[1]] | {cell[1]}

            if pivot.__len__() == self.max_pivot:
                return pivot

        return pivot

    def get_all_pivot_start(self, unseen, agents_location):
        pivot = dict()
        while unseen.__len__() > 0:
            remove_from_unseen_set = set()
            sort_unseen = self.get_centrality_list_wachers(unseen)
            # sort_unseen = self.get_min_list_wachers(unseen)

            while sort_unseen.__len__():
                cell = sort_unseen.pop()

                if not self.world.dict_wachers[cell[1]].intersection(remove_from_unseen_set):
                    pivot[cell[1]] = self.world.dict_wachers[cell[1]]
                    remove_from_unseen_set = remove_from_unseen_set | self.world.dict_wachers[cell[1]] | {cell[1]}

                # if pivot.__len__() == self.max_pivot:
                #     return pivot
            for key in pivot.keys():
                unseen = unseen - (pivot[key] | set(tuple([key])))

        # Utils.print_all_whacers(self.world, [list(pivot.keys()) + list(agents_location)])
        return pivot

    # find the index in open list for the new_node
    def find_index_to_open(self, new_node):
        # index lb
        index_a = 0

        # index hb
        index_b = len(self.open_list)

        if self.minimize == 0:
            # makspan cost
            cost_new = max(new_node.cost)
        else:
            # soc cost
            cost_new = sum(new_node.cost)

        # binry search
        while index_a < index_b:
            mid = (index_a + index_b) // 2
            data = self.open_list[mid]
            if new_node.f == data.f:
                # first tiebreaker smallest unseen is wining
                if data.unseen.__len__() == new_node.unseen.__len__():
                    if self.minimize == 0:
                        cost_data = max(np.abs(data.cost))
                    else:
                        cost_data = sum(np.abs(data.cost))

                    # second tiebreaker bisest cost is wining
                    if cost_data == cost_new:

                        # Third tiebreaker according to location in memory so as not to be random
                        if id(data) > id(new_node):
                            index_b = mid
                        else:
                            index_a = mid + 1
                    elif cost_data > cost_new:
                        index_b = mid
                    else:
                        index_a = mid + 1
                elif data.unseen.__len__() < new_node.unseen.__len__():
                    index_b = mid
                else:
                    index_a = mid + 1
            elif new_node.f > data.f:
                index_b = mid
            else:
                index_a = mid + 1
        return index_a

    def find_index_for_open(self, new_node):
        self.genrate_node += 1
        self.H_genrate += new_node.f
        index = self.find_index_to_open(new_node)
        self.open_list.insert(index, new_node)

    # insert new node to the open list for lazy_max huristics
    def insert_to_open_list_lazy_max(self, new_node):
        state = tuple((tuple(sorted(new_node.location)), tuple(new_node.dead_agent)))

        # find if the new node is already in open list or close lest (visit_list_dic)
        if new_node.first_genarate or not self.in_open_or_close(state):

            # if new_node is insert to open list second time no need to update visit_list_dic and calclate heuristic
            if not new_node.first_genarate:
                new_node.f = self.singelton_heuristic(new_node)
                self.visit_list_dic[state] = [new_node]
            self.find_index_for_open(new_node)

        # find if is an already node in open list or close lest (visit_list_dic) and if the new node is beter
        elif self.need_to_fix_parent(new_node, state):

            # if new_node is insert to open list second time no need to update visit_list_dic and calclate heuristic
            if not new_node.first_genarate:
                new_node.f = self.singelton_heuristic(new_node)
                self.visit_list_dic[state].append(new_node)
            self.find_index_for_open(new_node)
        return new_node

    def insert_to_open_list(self, new_node):

        state = tuple((tuple(sorted(new_node.location)), tuple(new_node.dead_agent)))

        if not self.in_open_or_close(state):
            new_node.f = self.get_heuristic(new_node)
            self.find_index_for_open(new_node)
            self.visit_list_dic[state] = [new_node]

        elif self.need_to_fix_parent(new_node, state):
            new_node.f = self.get_heuristic(new_node)
            self.find_index_for_open(new_node)
            self.visit_list_dic[state].append(new_node)

        return new_node

    def pop_open_list(self):

        if len(self.open_list):
            pop_open_list = self.open_list.pop()
            # pop_open_list = self.open_list.pop(0)

            while max(pop_open_list.cost) < 0:
                pop_open_list = self.open_list.pop()
        else:
            pop_open_list = 0

        return pop_open_list

    def get_real_dis(self, cell_a, cell_b):
        key = tuple(sorted((cell_a, cell_b)))
        return self.real_dis_dic[key]

    def get_closest_wachers(self, cell_a, cell_b):
        min_dis = 100000
        for k in self.world.dict_wachers[cell_a]:
            for t in self.world.dict_wachers[cell_b]:
                if self.real_dis_dic[tuple(sorted((k, t)))] < min_dis:
                    min_dis = self.real_dis_dic[tuple(sorted((k, t)))]
        return min_dis

    # Calculates the heuristics for Singleton
    def singelton_heuristic(self, new_node):

        # Holds the best heuristic value at any given time
        max_pivot_dist = 0
        #     if new_node.unseen.__len__()==0:
        #         return sum(new_node.cost)
        max_h_dis = 0
        all_cost = sum(new_node.cost)
        for cell in new_node.unseen:
            # Initialize a big number so that the heuristic is sure to be smaller than it
            min_dis = 1000000

            # Go through each of the cells wachers and look for the one that closest to one of the agents
            for whach in self.world.dict_wachers[cell]:

                if min_dis < max_pivot_dist:
                    break

                for index, agent in enumerate(new_node.location):
                    if index in new_node.dead_agent:
                        continue

                    if self.minimize == 0:
                        cost = new_node.cost[index]
                    else:
                        cost = all_cost

                    h = self.get_real_dis(agent, whach)

                    real_dis = cost + h

                    if min_dis >= real_dis:
                        # min_dis=real_dis
                        tmp_max_h_dis = h
                        min_dis = real_dis
                    # min_dis= min(min_dis, real_dis)

            if max_pivot_dist <= min_dis:
                max_h_dis = tmp_max_h_dis
                max_pivot_dist = min_dis
                # max_pivot_dist=min_dis
            # max_pivot_dist = max(max_pivot_dist, min_dis)

        if new_node.unseen.__len__() == 0:
            max_h_dis = 0
        if self.minimize == 0:
            max_pivot_dist = max(max_pivot_dist, max(new_node.cost))
        else:
            max_pivot_dist = max_h_dis + sum(new_node.cost)
        return max_pivot_dist

    def mtsp_makespan_heuristic_start(self, new_node, pivot):

        if pivot.__len__() == 0:
            return max(new_node.cost)

        citys = range(self.number_of_agent + pivot.__len__() + 1)
        all_pos = list(new_node.location) + list(pivot.keys())
        distance_agent_pivot = {}
        distance_in_pivot = {(i, 0): 0 for i in list(pivot.keys()) + list(range(1, self.number_of_agent + 1))}
        distance_out_start = {(0, i): 0 for i in list(range(1, self.number_of_agent + 1))}

        for i in citys[1: self.number_of_agent + 1]:
            for j in citys[self.number_of_agent + 1:]:
                if i - 1 not in new_node.dead_agent:
                    distance_agent_pivot[(i, all_pos[j - 1])] = min(
                        [self.real_dis_dic[tuple(sorted((all_pos[i - 1], k)))]
                         for k in self.world.dict_wachers[all_pos[j - 1]]])

        distance_pivot_pivot = dict()
        for i in citys[self.number_of_agent + 1:]:
            for j in citys[self.number_of_agent + 1:]:
                if i != j:
                    sort_pivot = tuple(sorted((all_pos[i - 1], all_pos[j - 1])))
                    if sort_pivot in self.old_pivot:
                        distance_pivot_pivot[(all_pos[i - 1], all_pos[j - 1])] = self.old_pivot[sort_pivot]
                    else:
                        distance_pivot_pivot[(all_pos[i - 1], all_pos[j - 1])] = self.get_closest_wachers(
                            all_pos[i - 1], all_pos[j - 1])
                        self.old_pivot[sort_pivot] = distance_pivot_pivot[(all_pos[i - 1], all_pos[j - 1])]

        # for_plot = [(0, 0)] + all_pos

        all_distance_dict = {**distance_out_start, **distance_in_pivot, **distance_agent_pivot, **distance_pivot_pivot}

        for_plot = [(0, 0)] + all_pos

        tmp = self.lp_model.get_makespan(for_plot, self.world, pivot, self.genrate_node, citys, all_distance_dict,
                                         new_node.cost)
        return tmp

    def mtsp_heuristic(self, new_node):

        tmp_pivot = self.get_pivot(new_node.unseen)
        pivot = {pivot: tmp_pivot[pivot] for pivot in tmp_pivot if pivot not in self.pivot_black_list}

        if pivot.__len__() == 0:
            if self.minimize == 0:
                return max(new_node.cost)
            else:
                return sum(new_node.cost)

        elif pivot.__len__() == 1:
            return -1  # self.singelton_heuristic(new_node)

        citys = range(self.number_of_agent + pivot.__len__() + 1)
        all_pos = list(new_node.location) + list(pivot.keys())
        distance_agent_pivot = {}

        distance_in_pivot = {(i, 0): 0 for i in list(pivot.keys()) + list(range(1, self.number_of_agent + 1))}
        distance_out_start = {(0, i): new_node.cost[i - 1] for i in (range(1, self.number_of_agent + 1))}

        # distance_out_start = {(0, i): 0 for i in (range(1, self.number_of_agent + 1))}

        for i in citys[1: self.number_of_agent + 1]:
            for j in citys[self.number_of_agent + 1:]:
                if i - 1 not in new_node.dead_agent:
                    distance_agent_pivot[(i, all_pos[j - 1])] = min(
                        [self.real_dis_dic[tuple(sorted((all_pos[i - 1], k)))]
                         for k in self.world.dict_wachers[all_pos[j - 1]]])

        distance_pivot_pivot = dict()
        for i in citys[self.number_of_agent + 1:]:
            for j in citys[self.number_of_agent + 1:]:
                if i != j:
                    sort_pivot = tuple(sorted((all_pos[i - 1], all_pos[j - 1])))
                    if sort_pivot in self.old_pivot:
                        distance_pivot_pivot[(all_pos[i - 1], all_pos[j - 1])] = self.old_pivot[sort_pivot]
                    else:
                        distance_pivot_pivot[(all_pos[i - 1], all_pos[j - 1])] = self.get_closest_wachers(
                            all_pos[i - 1], all_pos[j - 1])
                        self.old_pivot[sort_pivot] = distance_pivot_pivot[(all_pos[i - 1], all_pos[j - 1])]

        all_distance_dict = {**distance_out_start, **distance_in_pivot, **distance_agent_pivot, **distance_pivot_pivot}

        for_plot = [(0, 0)] + all_pos

        if self.minimize == 0:
            mtsp_cost = self.lp_model.get_makespan(for_plot, self.world, pivot, self.genrate_node, citys,
                                                   all_distance_dict, new_node.cost)
        elif self.minimize == 1:
            mtsp_cost = self.lp_model.get_soc(for_plot, self.world, pivot, self.genrate_node, citys, all_distance_dict,
                                              new_node.cost)
        else:
            print('no minimais')

        return mtsp_cost

    def get_heuristic(self, new_node):

        if self.huristic_index == 0:  # singelton
            closest_pivot_dist = self.singelton_heuristic(new_node)

        elif self.huristic_index == 1:  # max
            mtsp = self.mtsp_heuristic(new_node)
            singelton = self.singelton_heuristic(new_node)
            closest_pivot_dist = max(singelton, mtsp)

        elif self.huristic_index == 2:  # mtsp
            closest_pivot_dist = self.mtsp_heuristic(new_node)
            if closest_pivot_dist == -1:
                closest_pivot_dist = self.singelton_heuristic(new_node)

        return closest_pivot_dist

    def get_new_state(self, old_state, state_index):
        move_index = [0] * self.number_of_agent

        for j in range(self.number_of_agent):
            state_index, index = divmod(state_index, LOS)
            move_index[j] = index

        new_state, moving_status = self.world.get_action(old_state.location, move_index)
        return new_state, moving_status

    def get_cost(self, new_state, old_state, sort_indexing):

        old_cost = [old_state.cost[i] for i in sort_indexing]

        cost_from_acthon = [self.get_real_dis(data, old_state.location[i]) for i, data in enumerate(new_state)]
        cost_from_acthon = [cost_from_acthon[i] for i in sort_indexing]

        cost = list(map(add, cost_from_acthon, old_cost))

        return cost

    def in_open_or_close(self, state):
        # old_state, new_state, seen_state):

        # tmp_state = tuple(sorted(new_node.location))
        if not state in self.visit_list_dic:
            return False
        return True

    def get_all_frontire(self, old_state):
        all_frontire = []
        for index, agent_location in enumerate(old_state.location):
            if index not in old_state.dead_agent:
                all_frontire.append(self.world.BFS.get_frontire(agent_location, old_state.unseen))
            else:
                all_frontire.append([agent_location])

        return all_frontire

    def get_dead_list(self, old_state, new_state, sort_indexing):
        dead_list = old_state.dead_agent[:]
        for i in range(new_state.__len__()):
            if new_state[i] == old_state.location[i] and i not in dead_list:
                dead_list.append(i)

        dead_list = [sort_indexing[i] for i in dead_list]

        return dead_list

    def expend(self):
        # Returns the best valued node currently in the open list
        old_state = self.pop_open_list()

        if self.huristic_index == 3:
            if old_state.first_genarate == False:
                mtsp = self.mtsp_heuristic(old_state)
                old_state.first_genarate = True
                if mtsp >= old_state.f:
                    old_state.f = mtsp
                    self.insert_to_open_list_lazy_max(old_state)
                    return False

        # Checks if there are no more cell left to see (len(unseen)==0)
        if self.goal_test(old_state.unseen):
            return old_state
        self.expend_node += 1
        self.H_expend += old_state.f

        # Going through all the options to jump  for each one of the agents to produce all the valid situations
        for new_state in itertools.product(*self.get_all_frontire(old_state)):

            # There is no need to generate a situation similar to the father
            if new_state != old_state.location:

                # Rearranges agents and price (from largest to smallest) to maintain anonymity
                sorted_new_state, sorted_indexing = Utils.sort_list(new_state)

                # Gets the list that holds the dead agents
                dead_list = self.get_dead_list(old_state, new_state, sorted_indexing)

                # Calculates the unseen list for the new node
                seen_state = old_state.unseen - self.world.get_all_seen(sorted_new_state)

                new_node = Node(old_state, sorted_new_state, seen_state, dead_list,
                                self.get_cost(new_state, old_state, sorted_indexing))

                if self.huristic_index == 3:
                    self.insert_to_open_list_lazy_max(new_node)

                else:
                    # Inserts the new node to the open list
                    self.insert_to_open_list(new_node)

        return False

    # Checks whether there is a state similar to the new state and whether there is one better than the other
    def need_to_fix_parent(self, new_node, state):
        all_index = set()

        # Runs on all existing similar posters
        for index, old_node in enumerate(self.visit_list_dic[state]):
            cost_win = 0

            if self.minimize == 0:
                # Checks the cost of each cell individually and only if the cost of each cell in the same trend
                # (all high or all low) then it determines who is better in terms of cost
                for index, new_cell in enumerate(new_node.cost_map.keys()):
                    if cost_win == -5:
                        break
                    # A loop that passes through all the agents stnding in a particular cell (can be more than 1 such)
                    for i in range(new_node.cost_map[new_cell].__len__()):
                        if new_node.cost_map[new_cell][i] >= old_node.cost_map[new_cell][i] and cost_win >= 0:
                            # new_node is beter
                            cost_win = 1
                        elif new_node.cost_map[new_cell][i] <= old_node.cost_map[new_cell][i] and cost_win <= 0:
                            # old_node is beter
                            cost_win = -1
                        else:
                            cost_win = -5

            elif self.minimize == 1:
                # In soc the comparison can be made by the sum of the cost
                if sum(new_node.cost) >= sum(old_node.cost):
                    cost_win = 1
                elif sum(new_node.cost) <= sum(old_node.cost):
                    cost_win = -1

            if cost_win == 1 and old_node.unseen.issubset(new_node.unseen):
                self.open_is_beter += 1

                return False

            elif cost_win == -1 and new_node.unseen.issubset(old_node.unseen):
                self.new_is_beter += 1
                old_node.cost = [-max(old_node.cost)] * self.number_of_agent
                all_index.add(index)

        if all_index.__len__() > 0:
            self.visit_list_dic[state] = [data for i, data in enumerate(self.visit_list_dic[state])
                                          if i not in all_index]

        return True

    def run(self, writer, map_config, start_pos, obs_remove):

        # Writes to the file the type of heuristic that is activated
        h_type = {0: 'singlton', 1: 'max', 2: 'mtsp', 3: 'laze max', 4: 'BFS'}
        self.start_time = time()

        goal_node = False

        while not goal_node:
            # expend new node if goal_node is not folse the algoritem find solution
            goal_node = self.expend()

            # Checks if we have exceeded the time limit
            if time() - self.start_time > 300:
                # Writes to the file all the parameters of the experiment when the cost is 0 and the time is -1
                writer.writerow([map_config, start_pos, -1, h_type[self.huristic_index], self.H_start,
                                 self.H_genrate / self.genrate_node,
                                 self.H_expend / self.expend_node, self.max_pivot, 0, self.genrate_node,
                                 self.expend_node, self.open_is_beter, self.new_is_beter, obs_remove,
                                 [0] * self.number_of_agent])
                return

        all_path = self.get_path(goal_node, print_path=False)

        if self.genrate_node > 0:
            h_gen = self.H_genrate / self.genrate_node
            h_exp = self.H_expend / self.expend_node
        else:
            h_gen = self.H_genrate
            h_exp = self.H_expend

        # Writes to the file all the parameters of the experiment
        writer.writerow([map_config, start_pos, time() - self.start_time, h_type[self.huristic_index], self.H_start,
                         h_gen, h_exp, self.max_pivot, 0, self.genrate_node, self.expend_node,
                         self.open_is_beter, self.new_is_beter, obs_remove, goal_node.cost])

    # get all node on the optimal path . Still need to find the waypoints because of the jumps
    def get_path(self, gole_node, print_path):
        all_path = [gole_node]
        node = gole_node

        # parent.parent to prevent crashes when reaching the root of the tree
        while node.parent.parent is not None:
            if print_path:
                print(node.location, '\t', node.cost, '\t', node.f, '\t', self.expend_node)
            node = node.parent
            all_path.append(node)
        # Returns the inverted trajectory so that the starting point is the list head
        return all_path[::-1]


import csv
from alive_progress import alive_bar
import sys

if __name__ == '__main__':
    map_type = 'maze_11_11'
    name = 'soc_all'
    experement_name = f'{map_type}_{name}'
    map_config = f'./config/{map_type}_config.csv'

    row_map = Utils.convert_map(map_config)

    LOS = 4 + 1
    all_free = np.transpose(np.where(np.array(row_map) == 0))

    pivot = [5]
    exp_number = 2
    #  huristics_exp=[2]
    loop_number_of_agent = [2]
    minimize = {'mksp': 0, 'soc': 1}
    huristics_exp = [0, 1, 2, 3]
    #
    # if sys.argv:
    #     huristics_exp = [int(sys.argv[1])]
    #     loop_number_of_agent = [int(sys.argv[2])]

    #
    # else:
    # huristics_exp = [0, 1, 2]
    # loop_number_of_agent = [2, 4, 6]

    start_in = 0
    exp_index = 0
    data_file = open(f'{experement_name}_{loop_number_of_agent[0]}_agent_{huristics_exp[0]}_huristic.csv', 'w',
                     newline='\n')
    # for ii in range(100):
    #     start_pos = tuple(tuple(all_free[randint(0,all_free.__len__()-1)]) for f in range(loop_number_of_agent[0]))
    #     print(start_pos)

    # data_file = open(f'{loop_number_of_agent}_agent_{datetime.now()}.csv', 'w', newline='\n')

    writer = csv.writer(data_file, delimiter=',')
    writer.writerow(
        ['map_name', 'start_state', 'time', 'h type', 'h_start', 'h_genarate', 'h_expend', 'number of max pivot',
         'use black list', 'genarate', 'expend', 'open is beter', 'new is beter', 'obs remove', 'cost'])

    # start_config_as_string = np.loadtxt(f'./config/{map_type}{5}_agent_domain.csv', dtype=tuple,delimiter='\n')
    # all_start_config_as_tupel = [ast.literal_eval(i) for i in start_config_as_string]
    # b=list(all_start_config_as_tupel)
    #
    # all_start_config_as_tupel = [tuple((i[0],i[1],i[2],i[3],i[4],tuple(all_free[randint(0,all_free.__len__()-1)]))) for i in all_start_config_as_tupel]
    # for i in all_start_config_as_tupel:
    #     print(i)
    # all_start_config_as_tupel=[((1,1),())]
    remove_obs_number = 1

    # maps = pickle.load(open("all_maps_for_remove.p", "rb"))[:-1]
    # remove_obs_number=maps.__len__()

    with alive_bar(
            loop_number_of_agent.__len__() * exp_number * len(huristics_exp) * len(pivot) * remove_obs_number) as bar:
        for max_pivot in pivot:
            for number_of_agent in loop_number_of_agent:
                row_map = Utils.convert_map(map_config)
                start_config_as_string = np.loadtxt(f'./config/{map_type}_{number_of_agent}_agent_domain.csv',
                                                    dtype=tuple, delimiter='\n')
                all_start_config_as_tupel = [ast.literal_eval(i) for i in start_config_as_string]
                all_start_config_as_tupel = all_start_config_as_tupel[:exp_number]

                for remove_obs in range(remove_obs_number):
                    start_config_as_string = np.loadtxt(f'./config/{map_type}_{number_of_agent}_agent_domain.csv',
                                                        dtype=tuple, delimiter='\n')
                    all_start_config_as_tupel = [ast.literal_eval(i) for i in start_config_as_string]
                    all_start_config_as_tupel = all_start_config_as_tupel[:exp_number]

                    # all_start_config_as_tupel=list(map(tuple,all_free))
                    for start_pos in all_start_config_as_tupel:
                        # start_pos=tuple([start_pos]*number_of_agent)

                        for huristic in huristics_exp:
                            if exp_index >= start_in:
                                world = WorldMap(np.array(row_map), LOS)
                                mwrp = Mwrp(world, start_pos, huristic, max_pivot, map_type, minimize['soc'])
                                mwrp.run(writer, map_config, start_pos, remove_obs)
                            bar()
