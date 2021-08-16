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

    def __init__(self, world, start_pos, index, max_pivot,map_name,minimize):
        self.index = index
        self.minimize = minimize

        self.max_pivot = max_pivot*10
        self.start_time = 0
        self.number_of_agent = start_pos.__len__()

        self.world = world

        self.node_expend_index = 1

        unseen_all = set(map(tuple, np.transpose(np.where(self.world.grid_map == 0))))

        unseen_start = unseen_all - self.world.get_all_seen(start_pos)

        start_node = Node(Node(None, start_pos, [0] * self.number_of_agent, 0, [0] * start_pos.__len__()), start_pos,
                          unseen_start, [], [0] * start_pos.__len__())

        self.open_list = [start_node]

        self.visit_list_dic = {tuple(sorted(start_pos)): [start_node]}

        self.genrate_node = 0
        self.expend_node = 0

        self.H_genrate = 0
        self.H_expend = 0

        self.open_is_beter=0
        self.new_is_beter=0

        try:
            self.real_dis_dic = pickle.load(open(f"config/real_dis_dic_{map_name}.p", "rb"))
            self.centrality_dict = pickle.load(open(f"config/centrality_dict_{map_name}.p", "rb"))
        except:
            print('start FloydWarshall')
            self.real_dis_dic, self.centrality_dict = FloydWarshall(self.world.grid_map).floyd_warshall()
            print('end FloydWarshall')
            pickle.dump(self.real_dis_dic, open(f"config/real_dis_dic_{map_name}.p", "wb"))
            pickle.dump(self.centrality_dict, open(f"config/centrality_dict_{map_name}.p", "wb"))
        #self.real_dis_dic, self.centrality_dict = FloydWarshall(self.world.grid_map).floyd_warshall()


        self.centrality_dict = self.centrality_list_wachers(unseen_start)
        #self.max_pivot = max_pivot

        self.pivot = self.get_pivot(unseen_start)
        #self.pivot = self.get_all_pivot_start(unseen_start, start_pos)

        self.max_pivot = max_pivot

        #Utils.print_all_whacers(self.world, [list(self.pivot.keys()) + list(start_pos)])


        self.old_pivot = {tuple(sorted((i, j))): self.get_closest_wachers(i, j) for i in self.pivot.keys()
                          for j in self.pivot.keys() if i != j}

        distance_pivot_pivot = {(i, j): self.get_closest_wachers(i, j) for j in self.pivot.keys()
                                for i in self.pivot.keys() if i != j}

        distance_in_pivot = {(i, 0): 0 for i in list(self.pivot.keys()) + list(range(1, self.number_of_agent + 1))}

        distance_agent_pivot = {(i, j): 0 for j in self.pivot for i in range(1, self.number_of_agent + 1)}

        all_dist = {**distance_in_pivot, **distance_agent_pivot,
                    **{(0, i): 0 for i in range(1, self.number_of_agent + 1)},
                    **distance_pivot_pivot}

        self.lp_model = lp_mtsp(self.number_of_agent, self.pivot, all_dist)

        self.pivot_black_list = self.get_pivot_black_list(start_node)

        #pivot = {pivot: self.pivot[pivot] for pivot in self.pivot if pivot not in self.pivot_black_list}

        #Utils.print_serch_status(self.world, start_node, self.start_time, self.expend_node, self.genrate_node,False,pivot)

        self.H_start = self.get_heuristic(start_node)


    def goal_test(self, map_state):
        if not map_state.__len__():
            return True
        return False

    def get_pivot_black_list(self, start_node):
        old_u = 0
        #u_max = 0
        pivot_black_list = []
        tmp_pivot_key1 =sorted([(self.centrality_dict[key],key) for key in self.pivot.keys()])
        tmp_pivot_key = [val[1] for val in tmp_pivot_key1]
        for i in range(self.pivot.__len__()):
            #print(i)
            tmp_u = [self.mtsp_makespan_heuristic_start(start_node, {k: self.pivot[k] for k in tmp_pivot_key if k != cell}) for cell in tmp_pivot_key]

            index_min = np.argmax(tmp_u)
            u_max = max(tmp_u)
            if (old_u >= u_max):
                break
            else:
                #print(u_max)
                old_u = u_max
                bad_index = tmp_pivot_key[index_min]
                pivot_black_list.append(bad_index)
                del tmp_pivot_key[index_min]
        return set(pivot_black_list)

    def centrality_list_wachers(self, unseen):

        list_sort_by_centrality = []

        for index, cell in enumerate(unseen):
            centrality_value = 0
            for cell_whacers in self.world.dict_wachers[cell]:
                centrality_value += self.centrality_dict[cell_whacers]
            list_sort_by_centrality.append(
                (cell, centrality_value / (self.world.dict_wachers[cell].__len__()**2) * self.centrality_dict[cell]))

        self.centrality_dict = dict(list_sort_by_centrality)

        return self.centrality_dict

    def get_centrality_list_wachers(self, unseen):
        list_sort_by_centrality = [(self.centrality_dict[cell], cell) for cell in unseen]
        return sorted(list_sort_by_centrality)

    def get_min_list_wachers(self, unseen):
        list_sort_by_centrality = [(self.world.dict_wachers[cell].__len__(), cell ) for cell in unseen]
        return sorted(list_sort_by_centrality,reverse=True)

    def get_pivot(self, unseen):
        pivot = dict()
        remove_from_unseen_set = set()
        #sort_unseen = self.get_min_list_wachers(unseen)
        sort_unseen = self.get_centrality_list_wachers(unseen)

        while sort_unseen.__len__():
            cell = sort_unseen.pop()

            if not self.world.dict_wachers[cell[1]].intersection(remove_from_unseen_set):
                pivot[cell[1]] = self.world.dict_wachers[cell[1]]
                remove_from_unseen_set = remove_from_unseen_set | self.world.dict_wachers[cell[1]] | {cell[1]}

            if pivot.__len__() == self.max_pivot:
                return pivot

        #Utils.print_all_whacers(self.world, [list(pivot.keys()) + list(agents_location)])

        return pivot

    def get_pivot1(self, unseen):

        pivot = set()
        remove_from_unseen_set = set()
        #sort_unseen = self.get_min_list_wachers(unseen)
        sort_unseen = self.get_centrality_list_wachers(unseen)

        while sort_unseen.__len__():
            cell = sort_unseen.pop()

            if not self.world.dict_wachers[cell[1]].intersection(remove_from_unseen_set):
                pivot.add(cell[1])
                remove_from_unseen_set = remove_from_unseen_set | self.world.dict_wachers[cell[1]] | {cell[1]}

            if pivot.__len__() == self.max_pivot:
                return pivot

        #Utils.print_all_whacers(self.world, [list(pivot.keys()) + list(agents_location)])

        return pivot

    def get_all_pivot_start(self, unseen, agents_location):
        pivot = dict()
        while unseen.__len__()>0:
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
                unseen=unseen-(pivot[key] | set(tuple([key])))

        #Utils.print_all_whacers(self.world, [list(pivot.keys()) + list(agents_location)])
        return pivot

    # def get_new_node(self, new_state, seen_state, new_cost, old_state):
    #     heuristic = self.get_heuristic(new_state, seen_state)
    #     new_node = Node(old_state, new_state, seen_state, new_cost, heuristic)
    #     return new_node

    # def find_index_to_open_makspan(self, new_node):
    #     index_a = 0
    #     index_b = len(self.open_list)
    #     new_node_unseen_size=new_node.unseen.__len__()
    #     while index_a < index_b:
    #         mid = (index_a + index_b) // 2
    #
    #         data = self.open_list[mid]
    #
    #         if new_node.f == data.f:
    #
    #             if data.unseen.__len__() == new_node_unseen_size:
    #                 if  (max(np.abs(data.cost))) < max(new_node.cost):
    #                     index_b = mid
    #                 else:
    #                     index_a = mid + 1
    #             elif data.unseen.__len__() > new_node_unseen_size:
    #                 index_b = mid
    #             else:
    #                 index_a = mid + 1
    #         elif new_node.f < data.f:
    #             index_b = mid
    #         else:
    #             index_a = mid + 1
    #     return index_a

    def find_index_to_open_makspan(self, new_node):
        index_a = 0
        index_b = len(self.open_list)
        #all_cost_estimate = max([new_node.cost[i] + new_node.heuristics[i] for i in range(self.number_of_agent)])

        while index_a < index_b:
            mid = (index_a + index_b) // 2

            data = self.open_list[mid]

            #tmp_abs_data_cost = max([abs(data.cost[i]) + data.heuristics[i] for i in range(self.number_of_agent)])

            if new_node.f == data.f:

                if data.unseen.__len__() == new_node.unseen.__len__():
                    if  (max(np.abs(data.cost))) > max(new_node.cost):
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




    def insert_to_open_list(self, new_node):
        state=tuple((tuple(sorted(new_node.location)), tuple(new_node.dead_agent)))
        if not self.in_open_or_close(state):
            self.genrate_node += 1
            new_node.f = self.get_heuristic(new_node)
            self.H_genrate += new_node.f
            index = self.find_index_to_open_makspan(new_node)
            self.open_list.insert(index, new_node)
            self.visit_list_dic[state] = [new_node]

        elif self.need_to_fix_parent(new_node,state):
            self.genrate_node += 1
            new_node.f = self.get_heuristic(new_node)
            self.H_genrate += new_node.f
            index = self.find_index_to_open_makspan(new_node)
            self.open_list.insert(index, new_node)
            self.visit_list_dic[state].append(new_node)
        else:
            x=1

        return new_node

    def pop_open_list(self):

        if len(self.open_list):
            pop_open_list = self.open_list.pop()
            #pop_open_list = self.open_list.pop(0)

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

    def singelton_heuristic_makespan(self, new_node):
        max_pivot_dist = 0

        for cell in new_node.unseen:
            min_dis = 1000000

            for whach in self.world.dict_wachers[cell]:

                if min_dis < max_pivot_dist:
                    break

                for index,agent in enumerate(new_node.location):
                    if index in new_node.dead_agent:
                        continue

                    real_dis = new_node.cost[index] + self.get_real_dis(agent, whach)

                    if min_dis > real_dis:
                        tmp_max_a = agent
                        tmp_max_cell=whach

                    min_dis = min(min_dis, real_dis)

            # if max_pivot_dist <= min_dis:
            #     max_cell=tmp_max_cell
            #     max_a=tmp_max_a

            max_pivot_dist = max(max_pivot_dist, min_dis)

        max_pivot_dist = max(max_pivot_dist, max(new_node.cost))
        return max_pivot_dist

    def singelton_heuristic_soc(self, new_node):
        max_pivot_dist = 0
        if new_node.unseen.__len__()==0:
            return sum(new_node.cost)

        for cell in new_node.unseen:
            min_dis = 1000000
            if cell==(7,6):
                ppp=1
            for whach in self.world.dict_wachers[cell]:

                if min_dis < max_pivot_dist:
                    break

                for index, agent in enumerate(new_node.location):
                    if index in new_node.dead_agent:
                        continue

                    cost=new_node.cost[index]
                    h=self.get_real_dis(agent, whach)

                    real_dis = cost + h

                    if min_dis > real_dis:
                        tmp_max_h_dis = h

                    min_dis = min(min_dis, real_dis)

            if max_pivot_dist <= min_dis:
                max_h_dis = tmp_max_h_dis


            max_pivot_dist = max(max_pivot_dist, min_dis)
            #print(max_a, max_cell, max_pivot_dist, max_h_dis)

        #Utils.print_serch_status(self.world,new_node,self.start_time,self.expend_node,self.genrate_node,False,{max_cell : self.world.dict_wachers[max_cell]})

       # print(new_node.location, new_node.cost, max_h_dis)
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
                if i-1 not in new_node.dead_agent:
                    distance_agent_pivot[(i, all_pos[j - 1])] = min([self.real_dis_dic[tuple(sorted((all_pos[i - 1], k)))]
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

        tmp = self.lp_model.get_makespan(for_plot, self.world, pivot, self.genrate_node, citys, all_distance_dict,new_node.cost)
        return tmp

    def mtsp_makespan_heuristic1(self, new_node):

        tmp_pivot = self.get_pivot1(new_node.unseen)
        if not tmp_pivot.issubset(self.pivot_black_list):
            pivot = tmp_pivot -self.pivot_black_list
        else:
            pivot=tmp_pivot
        if pivot.__len__() == 0:
            return max(new_node.cost), 'zero'
        elif pivot.__len__() == 1:
            return self.singelton_heuristic(new_node), 'singleton'

        citys = range(self.number_of_agent + pivot.__len__() + 1)
        all_pos = list(new_node.location) + list(pivot)
        distance_agent_pivot = {}

        distance_in_pivot = {(i, 0): 0 for i in  pivot.union(range(1, self.number_of_agent + 1))}

        distance_out_start = {(0, i): 0 for i in range(1, self.number_of_agent + 1)}

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

        mtsp_cost = self.lp_model.get_makespan1(for_plot, self.world, pivot, self.genrate_node, citys, all_distance_dict,
                                               new_node.cost)

        return mtsp_cost, 'mtsp'

    def mtsp_heuristic(self, new_node):

        tmp_pivot = self.get_pivot(new_node.unseen)
        pivot = {pivot: tmp_pivot[pivot] for pivot in tmp_pivot if pivot not in self.pivot_black_list}

        if pivot.__len__() == 0:
            return max(new_node.cost)
        elif pivot.__len__() == 1:
            return  -1 #self.singelton_heuristic(new_node)

        citys = range(self.number_of_agent + pivot.__len__() + 1)
        all_pos = list(new_node.location) + list(pivot.keys())
        distance_agent_pivot = {}

        distance_in_pivot = {(i, 0): 0 for i in list(pivot.keys()) + list(range(1, self.number_of_agent + 1))}
        distance_out_start = {(0, i): new_node.cost[i-1] for i in (range(1, self.number_of_agent + 1))}

        #distance_out_start = {(0, i): 0 for i in (range(1, self.number_of_agent + 1))}

        for i in citys[1: self.number_of_agent + 1]:
            for j in citys[self.number_of_agent + 1:]:
                if i-1 not in new_node.dead_agent:
                    distance_agent_pivot[(i, all_pos[j - 1])] = min([self.real_dis_dic[tuple(sorted((all_pos[i - 1], k)))]
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

        if self.minimize==0:
            mtsp_cost = self.lp_model.get_makespan(for_plot, self.world, pivot, self.genrate_node, citys, all_distance_dict,new_node.cost)
        elif self.minimize==1:
            mtsp_cost = self.lp_model.get_soc(for_plot, self.world, pivot, self.genrate_node, citys, all_distance_dict,new_node.cost)
        else:
            print('no minimais')


        return mtsp_cost

    # def mtsp_sum_of_cost_heuristic(self, new_node):
    #
    #     tmp_pivot = self.get_pivot(new_node.unseen)
    #     pivot = {pivot: tmp_pivot[pivot] for pivot in tmp_pivot if pivot not in self.pivot_black_list}
    #
    #     if pivot.__len__() == 0:
    #         return max(new_node.cost)
    #     elif pivot.__len__() == 1:
    #         return  -1 #self.singelton_heuristic(new_node)
    #
    #     citys = range(self.number_of_agent + pivot.__len__() + 1)
    #     all_pos = list(new_node.location) + list(pivot.keys())
    #     distance_agent_pivot = {}
    #     distance_in_pivot = {(i, 0): 0 for i in list(pivot.keys()) + list(range(1, self.number_of_agent + 1))}
    #     distance_out_start = {(0, i): 0 for i in list(range(1, self.number_of_agent + 1))}
    #
    #     for i in citys[1: self.number_of_agent + 1]:
    #         for j in citys[self.number_of_agent + 1:]:
    #             if i-1 not in new_node.dead_agent:
    #                 distance_agent_pivot[(i, all_pos[j - 1])] = min([self.real_dis_dic[tuple(sorted((all_pos[i - 1], k)))]
    #                                                                  for k in self.world.dict_wachers[all_pos[j - 1]]])
    #
    #     distance_pivot_pivot = dict()
    #     for i in citys[self.number_of_agent + 1:]:
    #         for j in citys[self.number_of_agent + 1:]:
    #             if i != j:
    #                 sort_pivot = tuple(sorted((all_pos[i - 1], all_pos[j - 1])))
    #                 if sort_pivot in self.old_pivot:
    #                     distance_pivot_pivot[(all_pos[i - 1], all_pos[j - 1])] = self.old_pivot[sort_pivot]
    #                 else:
    #                     distance_pivot_pivot[(all_pos[i - 1], all_pos[j - 1])] = self.get_closest_wachers(
    #                         all_pos[i - 1], all_pos[j - 1])
    #                     self.old_pivot[sort_pivot] = distance_pivot_pivot[(all_pos[i - 1], all_pos[j - 1])]
    #
    #
    #     all_distance_dict = {**distance_out_start, **distance_in_pivot, **distance_agent_pivot, **distance_pivot_pivot}
    #
    #     for_plot = [(0, 0)] + all_pos
    #
    #     mtsp_cost = self.lp_model.get_makespan(for_plot, self.world, pivot, self.genrate_node, citys, all_distance_dict,new_node.cost)
    #
    #
    #     return mtsp_cost


    def get_heuristic(self, new_node):
        if self.minimize == 0:
            if self.index == 0: # singelton
                closest_pivot_dist = self.singelton_heuristic_makespan(new_node)

            elif self.index == 1: # max
                mtsp = self.mtsp_heuristic(new_node)
                singelton = self.singelton_heuristic_makespan(new_node)
                closest_pivot_dist = max(singelton, mtsp)

                #print(f'singelton {singelton} \t mtsp {mtsp}')

            elif self.index == 2: # mtsp
                #t=time()

                closest_pivot_dist = self.mtsp_heuristic(new_node)
                #print(time()-t)

                if closest_pivot_dist == -1:
                    closest_pivot_dist = self.singelton_heuristic_makespan(new_node)

            elif self.index == 3: # BFS
                closest_pivot_dist = max(new_node.cost)

        if self.minimize == 1:
            if self.index == 0:  # singelton
                closest_pivot_dist = self.singelton_heuristic_soc(new_node)

            elif self.index == 1:  # max
                mtsp = self.mtsp_makespan_heuristic(new_node)
                singelton = self.singelton_heuristic_soc(new_node)
                closest_pivot_dist = max(singelton, mtsp)

                # print(f'singelton {singelton} \t mtsp {mtsp}')

            elif self.index == 2:  # mtsp
                #t=time()
                closest_pivot_dist = self.mtsp_heuristic(new_node)
                #print(time()-t)
                if closest_pivot_dist == -1:
                    closest_pivot_dist = self.singelton_heuristic_soc(new_node)

            elif self.index == 3:  # BFS
                closest_pivot_dist = max(new_node.cost)

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

        cost_from_acthon=[self.get_real_dis(data, old_state.location[i]) for i, data in enumerate(new_state)]
        cost_from_acthon = [cost_from_acthon[i] for i in sort_indexing]

        cost=list(map(add, cost_from_acthon, old_cost))


        return cost

    def in_open_or_close(self,state):
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
        old_state = self.pop_open_list()
        self.expend_node += 1
        self.H_expend += old_state.f
        #print(old_state.f)
        if self.goal_test(old_state.unseen):
            return old_state

        for new_state in itertools.product(*self.get_all_frontire(old_state)):
            if new_state != old_state.location:

                sorted_new_state, sorted_indexing = Utils.sort_list(new_state)

                dead_list = self.get_dead_list(old_state, new_state, sorted_indexing)

                seen_state = old_state.unseen - self.world.get_all_seen(sorted_new_state)

                new_node = Node(old_state, sorted_new_state, seen_state, dead_list,
                                self.get_cost(new_state, old_state, sorted_indexing))

                self.insert_to_open_list(new_node)

        return False

    def need_to_fix_parent(self, new_node,state):
        cost_sort_new=sorted(new_node.cost)
        all_index=set()
        for index, old_node in enumerate(self.visit_list_dic[state]):
            cost_sort_old = sorted(old_node.cost)

            # max_new_node_cost = max(new_node.cost)
            # max_old_node_cost = max(old_node.cost)
            if False not in [cost_sort_new[i] >= cost_sort_old[i] for i in range(self.number_of_agent)] and old_node.unseen.issubset(new_node.unseen):
                self.open_is_beter+=1
                return False

            elif False not in [cost_sort_new[i] <= cost_sort_old[i] for i in range(self.number_of_agent)] and new_node.unseen.issubset(old_node.unseen):
                self.new_is_beter+=1
                old_node.cost = [-max(old_node.cost)] * self.number_of_agent
                all_index.add(index)
                #return True

        if all_index.__len__()>0:
            self.visit_list_dic[state]=[data for i , data in enumerate(self.visit_list_dic[state])
                                                    if i not in all_index]
            #del self.visit_list_dic[new_node.location][tuple(all_index)]

        return True

    def run(self, writer, map_config, start_pos,obs_remove):
        htype = {0: 'singlton', 1: 'max', 2: 'mtsp',3:'BFS'}
        self.start_time = time()
        # print("\nstart algoritem ... ", end='')

        goal_node = False
        while not goal_node:
            goal_node = self.expend()

            if time() - self.start_time > 300:
                writer.writerow([map_config, start_pos, -1, htype[self.index], self.H_start,
                                 self.H_genrate / self.genrate_node,
                                 self.H_expend / self.expend_node, self.max_pivot, 0, self.genrate_node,
                                 self.expend_node, self.open_is_beter, self.new_is_beter,obs_remove*5, [0]*self.number_of_agent])
                return
        #self.get_path(goal_node)

        #Utils.print_serch_status(world,goal_node,0,0,0,False)

        if self.genrate_node > 0:
            h_gen = self.H_genrate / self.genrate_node
        else:
            h_gen = self.H_genrate

        writer.writerow([map_config, start_pos, time() - self.start_time, htype[self.index], self.H_start,
                         h_gen,
                         self.H_expend / self.expend_node, self.max_pivot, 0, self.genrate_node, self.expend_node,
                         self.open_is_beter,self.new_is_beter,obs_remove*5, goal_node.cost])

        # print(f"find solution in -> {time() - self.start_time} sec at cost of {goal_node.cost}"
        #       f" and open {self.number_of_node} node")

    def get_path(self, gole_node):
        all_path = [gole_node]
        node = gole_node
        while node.parent.parent is not None:
            print(node.location ,'\t', node.cost,'\t', node.f,'\t',self.genrate_node)
            node = node.parent
            all_path.append(node)
        return all_path[::-1]

    def print_path(self, gole_node, see_agent_walk):
        all_path = self.get_path(gole_node)
        tmp_location = []
        for cell in all_path:

            # print(f'L = {cell.location} \t h = {cell.heuristics} ')
            tmp_location.append(cell.location)
            tmp_word = np.copy(self.world.grid_map)
            for k in cell.unseen:
                tmp_word[k] = 2
            for j in cell.location:
                tmp_word[j] = 3
            if see_agent_walk:
                plt.figure(1)
                plt.pcolormesh(tmp_word, edgecolors='black', linewidth=0.01)
                plt.gca().set_aspect('equal')
                plt.gca().set_ylim(plt.gca().get_ylim()[::-1])
                plt.draw()
                plt.pause(0.001)
                plt.clf()
                sleep(0.5)
        plt.close('all')
        Utils.print_all_whacers(self.world, tmp_location)


import csv
from alive_progress import alive_bar
import sys

if __name__ == '__main__':
    map_type = 'maze_11_11'
    experement_name=f'{map_type}_soc'
    map_config = f'./config/{map_type}_config.csv'

    #map_config='./config/for_jornal.csv'
    #row_map = np.genfromtxt(map_config, delimiter=',', case_sensitive=True)

    row_map=Utils.convert_map(map_config)

    LOS = 4 + 1
    all_free = np.transpose(np.where(np.array(row_map) == 0))

    pivot = [5]
    exp_number = 10
  #  huristics_exp=[2]
  #  loop_number_of_agent=[2]
    minimize = { 'makespan' : 0, 'soc' : 1}

    #loop_number_of_agent = [2,4,6]

    if sys.argv:
        huristics_exp = [int(sys.argv[1])]
        loop_number_of_agent = [int(sys.argv[2])]

    else:
        #huristics_exp = [0, 1, 2]
        loop_number_of_agent = [2, 4, 6]

    # for ii in range(100):
    #     start_pos = tuple(tuple(all_free[randint(0,all_free.__len__()-1)]) for f in range(loop_number_of_agent[0]))
    #     print(start_pos)

    start_in=0
    exp_index=0
    data_file = open(f'{experement_name}_{loop_number_of_agent[0]}_agent_{huristics_exp[0]}_huristic.csv', 'w', newline='\n')

    #data_file = open(f'{loop_number_of_agent}_agent_{datetime.now()}.csv', 'w', newline='\n')

    writer = csv.writer(data_file, delimiter=',')
    writer.writerow(
        ['map_name', 'start_state', 'time', 'h type', 'h_start', 'h_genarate', 'h_expend', 'number of max pivot',
         'use black list', 'genarate', 'expend','open is beter','new is beter','obs remove', 'cost'])

    # start_config_as_string = np.loadtxt(f'./config/{map_type}{5}_agent_domain.csv', dtype=tuple,delimiter='\n')
    # all_start_config_as_tupel = [ast.literal_eval(i) for i in start_config_as_string]
    # b=list(all_start_config_as_tupel)
    #
    # all_start_config_as_tupel = [tuple((i[0],i[1],i[2],i[3],i[4],tuple(all_free[randint(0,all_free.__len__()-1)]))) for i in all_start_config_as_tupel]
    # for i in all_start_config_as_tupel:
    #     print(i)
    # all_start_config_as_tupel=[((1,1),())]

    #remove_obs_number=71//5+1
    remove_obs_number=1

    with alive_bar(loop_number_of_agent.__len__() * exp_number * len(huristics_exp) * len(pivot) * remove_obs_number) as bar:
        for max_pivot in pivot:
            for number_of_agent in loop_number_of_agent:
                row_map = Utils.convert_map(map_config)
                for remove_obs in range(remove_obs_number):
                    start_config_as_string = np.loadtxt(f'./config/{map_type}_{number_of_agent}_agent_domain.csv',dtype=tuple,delimiter='\n')
                    all_start_config_as_tupel= [ast.literal_eval(i) for i in start_config_as_string]
                    all_start_config_as_tupel=all_start_config_as_tupel[:exp_number]
                    #all_start_config_as_tupel=[((1,1),(1,2))]
                   # all_start_config_as_tupel=list(map(tuple,all_free))
                    for start_pos in all_start_config_as_tupel:
                        #start_pos=tuple([start_pos]*number_of_agent)
                        for huristic in huristics_exp:
                            if exp_index >= start_in:
                                world = WorldMap(np.array(row_map), LOS)

                                mwrp = Mwrp(world, start_pos, huristic, max_pivot,map_type,minimize['makespan'])
                                mwrp.run(writer, map_config, start_pos,remove_obs)
                            exp_index += 1
                            bar()
                    #row_map = world.remove_obstical(5)

# TODO:
# fix pop(0) to pop() -> V
# add kill agent -> V

# jump to frontir -> V

# number of pivot -> V

# start random state
# start same state

# soc  + makspan
# conected line of cite
# stat agant on pivot


# multy singelton
#


