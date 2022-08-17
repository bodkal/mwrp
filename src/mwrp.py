import matplotlib.pyplot as plt
import numpy as np
from script.utils import Node, Utils, LpMtsp, Loger
from script.world import WorldMap
from operator import add
import itertools
from time import time
import heapq
import csv
import ast
import pickle

class Mwrp(WorldMap):

    def __init__(self, start_pos: tuple, minimize: int, map_type,use_black_list: bool = True, huristic_index: int = 3, max_pivot: int = 5) -> None:
        """
        :param world:           An WorldMap  object from script.world that contains the parameters of the world
        :param start_pos:       The starting position of the agents as tupel . For example for two agents : ((1,1),(4,6))
        :param huristic_index:  heuristic activated in this session ( {0: 'singlton', 1: 'max', 2: 'mtsp',3 : 'laze max',4:'BFS'})
        :param max_pivot:       The maximum number of pivots for calculating heuristics . From experience 5 and 6 were the best
        :param map_name:        The name of the map. For example : maze_11_11
        :param minimize:        What we want to minimize (soc, mksp)
        """
        start_pos=tuple(start_pos)
        super().__init__(map_type)
        self.huristic_index = huristic_index
        self.minimize = minimize
        self.start_time = 0
        self.number_of_agent = start_pos.__len__()
        self.use_black_list=use_black_list

        # we do not want to limit the number of pivot now we well limit it later in the code
        self.max_pivot = max_pivot * 10

        # Need it only for the experiments
        self.node_expend_index = 1
        self.genrate_node = 0
        self.expend_node = 0

        # sor=sorted([(self.centrality_dict[key], key) for key in self.centrality_dict.keys()])
        # for key in self.centrality_dict.keys():
        #     self.grid_map[key]=self.centrality_dict[key]
        # plt.imshow(self.grid_map)
        # plt.show()

        self.centrality_dict = self.centrality_list_watchers(self.free_cell[1])

        # Filters all the cells that can be seen from the starting position
        unseen_start = self.free_cell[1] - self.get_all_seen(start_pos)

        # Produces the initial NODE
        start_node = Node(None, start_pos, unseen_start, [0] * start_pos.__len__(), self.minimize)

        # open sort from top to bottom (best down)
        self.open_list = [start_node]
        heapq.heapify(self.open_list)

        # open and close list
        self.visit_list_dic = {tuple(sorted(start_pos)): [start_node]}

        # Pre-calculate suspicious points that will be PIVOT to save time during the run
        self.pivot = self.get_pivot(unseen_start)

        self.old_pivot = {tuple(sorted((i, j))): self.get_closest_watchers(i, j) for i in self.pivot.keys()
                          for j in self.pivot.keys() if i != j}

        # limit the number of pivot
        self.max_pivot = max_pivot

        self.init_lp_model()

        if use_black_list:
            # List of pivot that lower the heuristic
            self.pivot_black_list = self.get_pivot_black_list(start_node)

        self.mast_stand_cell=[]
        self.mtsp_heuristic(start_node)


    # TODO  not the best method requires improvement
    def init_lp_model(self) -> None:
        """
        """
        # Calculate all the distances between the PIVOT to calculate heuristics The goal is to try to understand if
        # there are PIVOT points that lower the value of the heuristics and filter them
        distance_pivot_pivot = {(i, j): self.get_closest_watchers(i, j) for j in self.pivot.keys()
                                for i in self.pivot.keys() if i != j}
        distance_in_pivot = {(i, 0): 0 for i in list(self.pivot.keys()) + list(range(1, self.number_of_agent + 1))}
        distance_agent_pivot = {(i, j): 0 for j in self.pivot for i in range(1, self.number_of_agent + 1)}
        all_dist = {**distance_in_pivot, **distance_agent_pivot,
                    **{(0, i): 0 for i in range(1, self.number_of_agent + 1)},
                    **distance_pivot_pivot}


        # Initializes the CPLEX object
        self.lp_model = LpMtsp(self.number_of_agent, self.pivot, all_dist)

    def clean_serch(self, start_node: Node, mast_stand_cell,dict_fov) -> None:
        """
        refrash the MWRP serch
        :param start_node: the parameter for starting the serch
        """
        self.open_list = [start_node]
        heapq.heapify(self.open_list)
        self.visit_list_dic = {tuple(sorted(start_node.location)): [start_node]}
        self.number_of_agent = start_node.location.__len__()
        self.mast_stand_cell=mast_stand_cell
        self.dict_fov=dict_fov
        self.init_lp_model()
        if self.use_black_list:
            self.pivot_black_list = self.get_pivot_black_list(start_node)

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
            tmp_u = [self.mtsp_heuristic(start_node,
                                         {k: self.pivot[k] for k in tmp_pivot_key if k != cell}) for cell in
                     tmp_pivot_key]

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

    def centrality_list_watchers(self, unseen: set) -> dict:
        """
        sort the unseen based on the each cell centrality watchers , number of watchers and cell centrality higher it is the better
        :param unseen: the unseen set of a node
        :return: Dictionary of each cell centrality
        """
        centrality_dict = dict()
        # list_sort_by_centrality = []
        # Goes through every cell in unseen and And calculates its centrality
        for index, cell in enumerate(unseen):
            # sum all the cell watchers centrality
            centrality_value = sum(
                self.centrality_dict[cell_whacers] for cell_whacers in self.dict_watchers[cell])

            # calculate all cells centrality
            centrality_dict[cell] = centrality_value / (self.dict_watchers[cell].__len__() ** 2) * \
                                    self.centrality_dict[cell]

        return centrality_dict

    def get_centrality_list_watchers(self, unseen: set) -> list:
        """
        returt sorted list by centrality sort from the bottom up
        :param unseen: the unseen set of a node
        :return: sorted list of each cell centrality
        """
        # create sorted list of each cell centrality based on unseen
        list_sort_by_centrality = sorted([(self.centrality_dict[cell], cell) for cell in unseen])

        return list_sort_by_centrality

    def get_min_list_watchers(self, unseen: set) -> list:
        """
             returt sorted list by centrality sort from the bottom up
             :param unseen: the unseen set of a node
             :return: sorted list of each cell number watchers
             """

        # create sorted list of each cell number watchers based on unseen
        list_sort_by_centrality = sorted([(len(self.dict_watchers[cell]), cell) for cell in unseen], reverse=True)
        return list_sort_by_centrality

    def get_pivot(self, unseen: set) -> dict:
        """
        calculate the pivot location for the mtsp huristic
        :param unseen: the unseen set of a node
        :return: Dictionary of selected pivot
        """

        pivot = dict()
        remove_from_unseen_set = set()

        # get unseen list arranged according to the centrality of the watchers  (from experiments works better)
        sort_unseen = self.get_centrality_list_watchers(unseen)

        # TODO can get diffrint set of pivot bast on deiffrent metod
        # get unseen list arranged according to the smallest number of watchers
        # sort_unseen = self.get_min_list_watchers(unseen)

        # Finds the pivot points and keeps their watchers disjoint for the gdls graph
        while sort_unseen.__len__():

            cell = sort_unseen.pop()

            # Checks if the cell is ok and is not disjoint with pivot watchers that already selected
            if not self.dict_watchers[cell[1]].intersection(remove_from_unseen_set):
                pivot[cell[1]] = self.dict_watchers[cell[1]]

                # set() which contains all the pivot watchers that already selected (used to test the disjoint)
                remove_from_unseen_set = remove_from_unseen_set | self.dict_watchers[cell[1]] | {cell[1]}

            if pivot.__len__() == self.max_pivot:
                return pivot

        return pivot

    def insert_to_open_and_cunt(self, new_node: Node) -> None:
        """
        insert the new node to open lest
        :param new_node: node that need to insert to open lest
        """
        # Need it only for the experiments
        self.genrate_node += 1
        heapq.heappush(self.open_list, new_node)

    def insert_to_open_list_lazy_max(self, new_node: Node) -> None:
        """
        insert new node to the open list for lazy_max huristics
        :param new_node: node that need to insert to open lest
        """

        #  Constructs the state so that the agents maintain anonymity
        state = tuple((tuple(sorted(new_node.location)), tuple(new_node.dead_agent)))

        # find if the new node is already in open list or close lest (visit_list_dic)
        if new_node.first_genarate or not self.in_open_or_close(state):

            # if new_node is insert to open list second time no need to update visit_list_dic and calclate heuristic
            if not new_node.first_genarate:
                new_node.f = self.singelton_heuristic(new_node)
                self.visit_list_dic[state] = [new_node]
            self.insert_to_open_and_cunt(new_node)

        # find if is an already node in open list or close lest (visit_list_dic) and if the new node is beter
        elif self.need_to_fix_parent(new_node, state):

            # if new_node is insert to open list second time no need to update visit_list_dic and calclate heuristic
            if not new_node.first_genarate:
                new_node.f = self.singelton_heuristic(new_node)
                self.visit_list_dic[state].append(new_node)
            self.insert_to_open_and_cunt(new_node)

    def insert_to_open_list(self, new_node: Node) -> None:
        """
        insert new node to the open list
        :param new_node: node that need to insert to open lest
        """

        #  Constructs the state so that the agents maintain anonymity
        state = tuple((tuple(sorted(new_node.location)), tuple(new_node.dead_agent)))

        # find if the new node is already in open list or close lest (visit_list_dic)
        if not self.in_open_or_close(state):
            new_node.f = self.get_heuristic(new_node)
            self.insert_to_open_and_cunt(new_node)
            self.visit_list_dic[state] = [new_node]

        # find if is an already node in open list or close lest (visit_list_dic) and if the new node is beter
        elif self.need_to_fix_parent(new_node, state):
            new_node.f = self.get_heuristic(new_node)
            self.insert_to_open_and_cunt(new_node)
            self.visit_list_dic[state].append(new_node)

    def pop_open_list(self) -> None:
        """
        get the best valid node in the open list
        :return:
        """
        if len(self.open_list):
            # pop_open_list = self.open_list.pop()

            pop_open_list = heapq.heappop(self.open_list)
            # Throws zombie node to the bin (zombie node -> dead_agent = True)
            while pop_open_list.f < 0:
                # pop_open_list = self.open_list.pop()
                pop_open_list = heapq.heappop(self.open_list)
        else:
            pop_open_list = 0

        return pop_open_list

    def get_real_dis(self, cell_a: tuple, cell_b: tuple) -> int:
        """
        Pulls out the real distance between two cell on the map The calculation is offleen and based on the FloydWarshall algorithm
        :param cell_a: cell a = (x1,y1)
        :param cell_b:  cell b = (x2,y2)
        :return: real distance between two cell
        """
        key = tuple(sorted((cell_a, cell_b)))
        return self.real_dis_dic[key]

    def get_closest_watchers(self, cell_a: tuple, cell_b: tuple) -> int:
        """
        get the closest watchers between 2 cells
        :param cell_a: cell a = (x1,y1)
        :param cell_b:  cell b = (x2,y2)
        :return: real distance between the two cell  closest watchers
        """
        min_dis = 100000
        # iterat on all pairs of watchers that both cells have
        for t, k in itertools.product(*(self.watchers_frontier[cell_b], self.watchers_frontier[cell_a])):
            sort_k_t = tuple(sorted((k, t)))
            if self.real_dis_dic[sort_k_t] < min_dis:
                min_dis = self.real_dis_dic[sort_k_t]
        return min_dis

    def singelton_heuristic(self, new_node: Node) -> int:
        """
        Calculates the heuristics for Singleton
        :param new_node: node that need to calclate the hes heuristic
        :return: the h value for no eb and f value for eb
        """

        # Holds the best pivot and heuristic value at any given time
        max_pivot_dist = 0
        max_h_dis = 0

        all_cost = sum(new_node.cost)
        for cell in new_node.unseen:
            # Initialize a big number so that the heuristic is sure to be smaller than it
            min_dis = 1000000

            # Go through each of the cells watchers and look for the one that closest to one of the agents
            for whach in self.watchers_frontier[cell]:

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

                    # Holds the value of the nearest watchers to one of the agent
                    if min_dis >= real_dis:
                        tmp_max_h_dis = h
                        min_dis = real_dis

            # Holds the value of the farthest cell With the nearest watchers to one of the agent
            if max_pivot_dist <= min_dis:
                max_h_dis = tmp_max_h_dis
                max_pivot_dist = min_dis

        if new_node.unseen.__len__() == 0:
            max_h_dis = 0

        if self.minimize == 0:
            max_pivot_dist = max(max_pivot_dist, max(new_node.cost))
        else:
            max_pivot_dist = max_h_dis + sum(new_node.cost)
        return max_pivot_dist

    def get_all_distance_for_huristic(self, new_node, pivot):
        citys = range(self.number_of_agent + pivot.__len__() + 1)
        all_pos = list(new_node.location) + list(pivot.keys())

        # bild the dict of the agent pivot distance
        distance_agent_pivot = {}
        for i, j in itertools.product(*(citys[1: self.number_of_agent + 1], citys[self.number_of_agent + 1:])):
            if i - 1 not in new_node.dead_agent:
                distance_agent_pivot[(i, all_pos[j - 1])] = min(
                    [self.real_dis_dic[tuple(sorted((all_pos[i - 1], k)))]
                     for k in self.dict_watchers[all_pos[j - 1]]])

        # bild the dict of the pivot pivot distance
        distance_pivot_pivot = dict()
        for i, j in itertools.product(*(citys[self.number_of_agent + 1:], citys[self.number_of_agent + 1:])):
            if i != j:
                sort_pivot = tuple(sorted((all_pos[i - 1], all_pos[j - 1])))

                if sort_pivot in self.old_pivot:
                    distance_pivot_pivot[(all_pos[i - 1], all_pos[j - 1])] = self.old_pivot[sort_pivot]
                else:
                    distance_pivot_pivot[(all_pos[i - 1], all_pos[j - 1])] = self.get_closest_watchers(
                        all_pos[i - 1], all_pos[j - 1])
                    self.old_pivot[sort_pivot] = distance_pivot_pivot[(all_pos[i - 1], all_pos[j - 1])]

        # bild the dict of the pivot virtual distance
        distance_in_pivot = {(i, 0): 0 for i in list(pivot.keys()) + list(range(1, self.number_of_agent + 1))}

        # bild the dict of the virtual pivot distance
        distance_out_start = {(0, i): new_node.cost[i - 1] for i in range(1, self.number_of_agent + 1)}

        # join all dictionaries
        all_distance_dict = {**distance_out_start, **distance_in_pivot, **distance_agent_pivot, **distance_pivot_pivot}
        return all_distance_dict, all_pos

    def mtsp_heuristic(self, new_node: Node, pivot: dict = False) -> int:
        """
        Calculates the heuristics for mtsp
        :param new_node: node that need to calclate the hes heuristic
        :return: the h value for no eb and f value for eb
        """

        if not pivot:
            tmp_pivot = self.get_pivot(new_node.unseen)
            if self.use_black_list:
                # remove the pivot that lower the heuristic ( in pivot_black_list)
                pivot = {pivot: tmp_pivot[pivot] for pivot in tmp_pivot if pivot not in self.pivot_black_list}
            else:
                pivot=tmp_pivot

        # if there is no pivot on the map
        if pivot.__len__() == 0:
            if self.minimize == 0:
                return max(new_node.cost)
            else:
                return sum(new_node.cost)

        # for one pivot singelton >= mtsp
        elif pivot.__len__() == 1 and new_node.parent != None:
            # return -1 mean that we going to calculate the singelton ensted
            return -1

        all_distance_dict, all_pos = self.get_all_distance_for_huristic(new_node, pivot)

        if self.minimize == 0:
            mtsp_cost = self.lp_model.get_mksp(all_distance_dict, pivot, new_node, all_pos, self)
        elif self.minimize == 1:
            mtsp_cost = self.lp_model.get_soc(all_distance_dict, pivot, new_node, all_pos, self)
        else:
            print('no minimais')
        # Utils.print_serch_status(self,new_node,self.start_time,0,0,False)
        return mtsp_cost

    def get_heuristic(self, new_node: Node) -> int:
        """
        get the heuristic base on the heuristic_index value
        :param new_node: node that need to calclate the hes heuristic
        :return:  the h value for no eb and f value for eb
        """
        # singelton
        if self.huristic_index == 0:
            closest_pivot_dist = self.singelton_heuristic(new_node)

        # max
        elif self.huristic_index == 1:
            mtsp = self.mtsp_heuristic(new_node)
            singelton = self.singelton_heuristic(new_node)
            closest_pivot_dist = max(singelton, mtsp)

        # mtsp
        elif self.huristic_index == 2:
            closest_pivot_dist = self.mtsp_heuristic(new_node)
            if closest_pivot_dist == -1:
                closest_pivot_dist = self.singelton_heuristic(new_node)

        return closest_pivot_dist

    def get_cost(self, new_state: Node, old_state: Node) -> list:
        """
        get the cost for the new node sorted by hes new indexing
        :param new_state:  node that need to get hes new cost
        :param old_state: new_state ferent
        :return: cost for the new node
        """

        cost_from_acthon = [self.get_real_dis(data, old_state.location[i]) for i, data in enumerate(new_state)]
        cost = list(map(add, cost_from_acthon, old_state.cost))
        return cost

    def in_open_or_close(self, state: tuple) -> bool:
        """
        find if the state are opend already (open or close)
        :param state: the new state
        :return: True if we already open same state state
        """
        if not state in self.visit_list_dic:
            return False
        return True

    def get_all_frontire(self, old_state: Node) -> list:
        """
        return all frontire (see new cells) for the EF metod
        :param old_state: need to expend the Node
        :return: list of all frontire
        """

        all_frontire = []
        for index, agent_location in enumerate(old_state.location):

            # get frontire for spsific agent (find whit BFS)
            if index not in old_state.dead_agent:
                all_frontire.append(self.BFS.get_frontier(agent_location, old_state.unseen,self.mast_stand_cell))
            else:
                all_frontire.append([agent_location])

        return all_frontire

    def expend(self):
        # Returns the best valued node currently in the open list
        old_state = self.pop_open_list()
        if self.huristic_index == 3:
            if not old_state.first_genarate:
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

        # Going through all the options to jump  for each one of the agents to produce all the valid situations
        for new_state in itertools.product(*self.get_all_frontire(old_state)):

            # There is no need to generate a situation similar to the father
            if new_state != old_state.location:

                # # Rearranges agents and price (from largest to smallest) to maintain anonymity
                # sorted_new_state, sorted_indexing = Utils.sort_list(new_state)

                # Gets the list that holds the dead agents
                # dead_list = self.get_dead_list(old_state, new_state)

                # Calculates the unseen list for the new node
                seen_state = old_state.unseen - self.get_all_seen(new_state)

                new_node = Node(old_state, new_state, seen_state, self.get_cost(new_state, old_state), self.minimize)

                if self.huristic_index == 3:
                    self.insert_to_open_list_lazy_max(new_node)

                else:
                    # Inserts the new node to the open list
                    self.insert_to_open_list(new_node)

        return False

    def need_to_fix_parent(self, new_node: Node, state: tuple) -> bool:
        """
        Checks whether there is a state similar to the new state and whether there is one better than the other

        :param new_node: the new node
        :param state: the new state (pos , dead)
        :return: bool if need to insert to open
        """

        all_index = set()

        # Runs on all existing similar posters
        for index, old_node in enumerate(self.visit_list_dic[state]):
            cost_win = 0

            if self.minimize == 0:
                # Checks the cost of each cell individually and only if the cost of each cell in the same trend
                # (all high or all low) then it determines who is better in terms of cost
                for index, new_cell in enumerate(new_node.cost_map.keys()):

                    # A loop that passes through all the agents stnding in a particular cell (can be more than 1 such)
                    for i in range(new_node.cost_map[new_cell].__len__()):
                        if cost_win == -5:
                            break
                        if new_node.cost_map[new_cell][i] >= old_node.cost_map[new_cell][i] and cost_win >= 0:
                            # new_node is beter
                            cost_win = 1
                        elif new_node.cost_map[new_cell][i] <= old_node.cost_map[new_cell][i] and cost_win <= 0:
                            # old_node is beter
                            cost_win = -1
                        else:
                            cost_win = -5

            # In soc the comparison can be made by only the sum of the cost
            elif self.minimize == 1:
                cost_win = 1 if sum(new_node.cost) >= sum(old_node.cost) else -1

            if cost_win == 1 and old_node.unseen.issubset(new_node.unseen):
                return False

            elif cost_win == -1 and new_node.unseen.issubset(old_node.unseen):

                old_node.f = -old_node.f
                all_index.add(index)

        if all_index.__len__() > 0:
            self.visit_list_dic[state] = [data for i, data in enumerate(self.visit_list_dic[state])
                                          if i not in all_index]

        return True

    def run(self, loger: csv = None, map_config: str = "", save_to_file: bool = True, need_path=True) -> None:
        """
        run the algorithm and return if finds solution or 5 minutes limit
        :param loger: the csv file holder (panda pkg)
        :param map_config: name of the experiment
        :param obs_remove: number of obstacle remove (for the experiment)
        :return:
        """

        start_pos = self.open_list[0].location
        # Writes to the file the type of heuristic that is activated
        h_type = {0: 'singlton', 1: 'max', 2: 'mtsp', 3: 'laze max', 4: 'BFS'}
        self.start_time = time()
        goal_node = False

        while not goal_node:
            # expend new node if goal_node is not folse the algoritem find solution
            goal_node = self.expend()
            end_time=time() - self.start_time
            # Checks if we have exceeded the time limit
            if end_time > 600 :
                print('timute !!!   open_list size = ', self.open_list.__len__())
                if save_to_file:
                    # Writes to the file all the parameters of the experiment when the cost is 0 and the time is -1
                    loger.write([map_config, start_pos, -1, self.minimize, self.max_pivot, self.use_black_list, self.genrate_node, self.expend_node,0])

                return  False, [] ,self.expend_node ,end_time ,[0]

        all_path = self.get_path(goal_node, need_path)


        if save_to_file:
            # Writes to the file all the parameters of the experiment
            loger.write([map_config, start_pos,  time() - self.start_time, self.minimize, self.max_pivot, self.use_black_list, self.genrate_node,
                 self.expend_node, goal_node.cost])


        return True, all_path ,self.expend_node ,end_time , goal_node.cost

    def get_path(self, gole_node: Node, need_path: bool = True, print_path: bool = False) -> dict:
        """
        #fix sorted unsycronic location and get all node on the optimal path between jump points

        :param gole_node: the goal node
        :param need_path: flag if need the path if true calculate pate
        :param print_path: flag if nead to print the path
        :return: dict that hold all path one for each agent (not the same length)
        """

        if need_path:
            all_jump_points = []
            node = gole_node
            # geting all jump points
            while node.parent is not None:
                if print_path:
                    print(node)

                # fix usicronic sort (the agent jumps between paths)
                all_jump_points.append(node.location)
                node = node.parent
            all_jump_points.append(node.location)

            if print_path:
                print(node)

            # reverse point because need path from start to goal
            all_jump_points = all_jump_points[::-1]

            # get all point on path by using BFS method
            dict_all_path = {i: [all_jump_points[0][i]] for i in range(self.number_of_agent)}
            for index in range(1, all_jump_points.__len__()):
                for i in range(self.number_of_agent):
                    dict_all_path[i].extend(self.BFS.get_path(all_jump_points[index - 1][i], all_jump_points[index][i]))

            return dict_all_path
        return {}

if __name__ == '__main__':
    map_type = 'maze_11_11'
    name = 'same_pos_agent'
    x=set()
    experement_name = f'{map_type}_{name}'
    map_config = f'../config/{map_type}_config.csv'

    row_map = Utils.convert_map(map_config)
    #print(row_map)

    all_free = np.transpose(np.where(np.array(row_map) == 0))

    #exempel for serch parmeter
   # a=pickle.load( open("config/all_maps_for_remove.p",'rb'))
    pivot = 5
    all_minimize_opthion = {'mksp': 0, 'soc': 1}
    minimize=all_minimize_opthion['mksp']
    number_of_agent = 3
    huristics=3
    use_black_list=True
    Utils.print_map(row_map)

    data_file = f'{experement_name}_{number_of_agent}_agent_{huristics}_minimize_{Utils.get_key_from_value(all_minimize_opthion,minimize)}.csv'
    titel_list =  ['map_name', 'start_state', 'time','minimize', 'number of max pivot','use black list', 'genarate', 'expend', 'cost']
    loger = Loger(data_file, titel_list)

    start_pos = ((1,10),(1,1),(8,8))
    mwrp = Mwrp(start_pos, minimize, map_type, use_black_list=use_black_list)
    all_path = mwrp.run(loger, map_type)
    Utils.print_mwrp_path(mwrp,all_path[1])
