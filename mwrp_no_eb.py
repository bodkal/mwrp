import numpy as np
from script.utils_no_eb import Node, Utils, FloydWarshall ,lp_mtsp
from script.world import WorldMap
import pickle
import matplotlib.pyplot as plt

from time import time, sleep
from random import randint
import ast

class Mwrp:

    def __init__(self, world, start_pos,huristic_index,max_pivot,map_name,minimize):
        self.huristic_index=huristic_index
        self.max_pivot=max_pivot
        self.start_time = 0
        self.number_of_agent = start_pos.__len__()

        self.world = world

        self.node_expend_index = 1

        unseen_all = set(map(tuple, np.transpose(np.where(self.world.grid_map == 0))))

        unseen_start = unseen_all - self.world.get_all_seen(start_pos)

        start_node = Node(Node(None, start_pos,[0]*self.number_of_agent, 0), start_pos, unseen_start,[])

        self.open_list = [start_node]

        self.visit_list_dic = {tuple(sorted(start_pos)): [start_node]}

        self.genrate_node = 0
        self.expend_node = 0

        self.H_genrate=0
        self.H_expend=0

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

        self.centrality_dict=self.centrality_list_wachers(unseen_start)

        self.pivot = self.get_pivot(unseen_start,start_pos)


        #Utils.print_all_whacers(self.world, [list(self.pivot.keys()) + list(start_pos)])

       # Utils.print_all_whacers(self.world, [list(start_pos)])

        self.old_pivot={tuple(sorted((i, j))): self.get_closest_wachers(i,j) for i in self.pivot.keys()
                                                                            for j in self.pivot.keys() if i != j }


        distance_pivot_pivot={(i, j):self.get_closest_wachers(i,j) for j in self.pivot.keys()
                                                                        for i in self.pivot.keys() if i != j}

        distance_in_pivot = {(i, 0): 0 for i in list(self.pivot.keys())+list(range(1,self.number_of_agent+1))}

        distance_agent_pivot={(i, j) : 0  for j in self.pivot for i in range(1,self.number_of_agent+1)}

        all_dist={**distance_in_pivot,**distance_agent_pivot,**{(0,i):0 for i in range(1,self.number_of_agent+1)},
                                                                                                **distance_pivot_pivot}

        self.lp_model = lp_mtsp(self.number_of_agent,self.pivot,all_dist)

        self.pivot_black_list=self.get_pivot_black_list(start_node)

        #Utils.print_all_whacers(self.world, [[k for k in self.pivot.keys() if k not in self.pivot_black_list]+ list(start_pos)])

        self.min_unseen=unseen_start.__len__()

        self.H_start=self.get_heuristic(start_node)
        self.open_is_beter=0
        self.new_is_beter=0
        #Utils.print_all_whacers(self.world, [list(self.pivot.keys())-+list(start_pos)])


    def goal_test(self, map_state):
        if not map_state.__len__():
            return True
        return False

    def get_pivot_black_list(self,start_node):
        old_u=self.mtsp_makespan_heuristic_start(start_node,self.pivot)
        u_max=0
        pivot_black_list=[]
        tmp_pivot_key=list(self.pivot.keys())
        #Utils.print_all_whacers(self.world, [list(self.pivot.keys())+list(start_pos)])


        for i in range(self.pivot.__len__()):
            tmp_u=[self.mtsp_makespan_heuristic_start(start_node,
                            {k: self.pivot[k] for k in self.pivot.keys() if k != cell}) for cell in tmp_pivot_key]

            index_min=np.argmax(tmp_u)
            u_max=max(tmp_u)
            if(old_u>=u_max):
                break
            else:
                old_u = u_max
                bad_index=tmp_pivot_key[index_min]
                pivot_black_list.append(bad_index)
                del tmp_pivot_key[index_min]
        return pivot_black_list


    def centrality_list_wachers(self, unseen):

        list_sort_by_centrality = []

        for index, cell in enumerate(unseen):
            centrality_value = 0
            for cell_whacers in self.world.dict_wachers[cell]:
                centrality_value += self.centrality_dict[cell_whacers]
            list_sort_by_centrality.append(
                ( cell,centrality_value / (self.world.dict_wachers[cell].__len__() ** 2) * self.centrality_dict[cell]))

        self.centrality_dict=dict(list_sort_by_centrality)

        return self.centrality_dict


    def get_centrality_list_wachers(self, unseen):
        list_sort_by_centrality=[(self.centrality_dict[cell],cell ) for cell in unseen]
        #list_sort_by_centrality=sorted(list_sort_by_centrality)
        return list_sort_by_centrality

    def get_pivot_test(self, unseen):
        centrality_dict = self.get_centrality_list_wachers(unseen)
        pivot = dict()
        while unseen.__len__():
            cell = centrality_dict.pop(0)
            if cell[1] in unseen:
                pivot[cell[1]] = self.world.dict_wachers[cell[1]]
                for pivot_wacher in pivot[cell[1]]:
                    unseen = unseen - self.world.dict_wachers[pivot_wacher]
        return pivot

    def get_pivot(self, unseen, agents_location):
        pivot = dict()
        remove_from_unseen_set = set()
        centrality_dict =  sorted(self.get_centrality_list_wachers(unseen))

        while centrality_dict.__len__():
            cell = centrality_dict.pop()

            if not self.world.dict_wachers[cell[1]].intersection(remove_from_unseen_set):
                pivot[cell[1]] = self.world.dict_wachers[cell[1]]
                remove_from_unseen_set = remove_from_unseen_set | self.world.dict_wachers[cell[1]] | {cell[1]}

            if pivot.__len__() == self.max_pivot:
                return pivot
        return pivot





    def get_new_node(self, new_state, seen_state, new_cost, old_state):
        heuristic = self.get_heuristic(new_state, seen_state)
        new_node = Node(old_state, new_state, seen_state, new_cost, heuristic)
        return new_node

    def find_index_to_open(self, new_node):
        index_a = 0
        index_b = len(self.open_list)
        all_cost_estimate = new_node.cost + new_node.heuristics
        while index_a < index_b:
            mid = (index_a + index_b) // 2
            data = self.open_list[mid]
            tmp_abs_data_cost=abs(data.cost)
            if all_cost_estimate == tmp_abs_data_cost + data.heuristics:

                if data.unseen.__len__() == new_node.unseen.__len__():
                    if tmp_abs_data_cost < all_cost_estimate:
                        index_b = mid
                    else:
                        index_a = mid + 1
                elif data.unseen.__len__() > new_node.unseen.__len__():
                    index_b = mid
                else:
                    index_a = mid + 1
            elif all_cost_estimate < tmp_abs_data_cost + data.heuristics:
                index_b = mid
            else:
                index_a = mid + 1
        return index_a

    def insert_to_open_list(self, new_node):

        if not self.in_open_or_close(new_node):
            self.genrate_node+=1
            new_node.heuristics = self.get_heuristic(new_node)
            self.H_genrate+=new_node.heuristics
            index = self.find_index_to_open(new_node)
            self.open_list.insert(index, new_node)
            self.visit_list_dic[tuple(sorted(new_node.location))] = [new_node]

        elif self.need_to_fix_parent(new_node):
            self.genrate_node+=1
            new_node.heuristics = self.get_heuristic(new_node)
            self.H_genrate+=new_node.heuristics
            index = self.find_index_to_open(new_node)
            self.open_list.insert(index, new_node)
            self.visit_list_dic[tuple(sorted(new_node.location))].append(new_node)

        return new_node

    def pop_open_list(self):

        if len(self.open_list):
            pop_open_list = self.open_list.pop(0)

            while pop_open_list.cost < 0:
                pop_open_list = self.open_list.pop(0)
        else:
            pop_open_list = 0

        return pop_open_list

    def get_real_dis(self, cell_a, cell_b):
        key = tuple(sorted((cell_a, cell_b)))
        #if key in self.real_dis_dic:
        return self.real_dis_dic[key]


    def get_closest_wachers(self,cell_a,cell_b):
        min_dis = 100000
        for k in self.world.dict_wachers[cell_a]:
            for t in self.world.dict_wachers[cell_b]:
                if self.real_dis_dic[tuple(sorted((k, t)))] < min_dis:
                    min_dis = self.real_dis_dic[tuple(sorted((k, t)))]
        return min_dis

    def singelton_heuristic(self, new_node):
        max_pivot_dist = 0
        # Utils.print_whacers(self.world,new_node.unseen)
        for cell in new_node.unseen:
            min_dis = 1000000


            for whach in self.world.dict_wachers[cell]:
                if min_dis < max_pivot_dist:
                    break
                for agent in new_node.location:
                    real_dis = self.get_real_dis(agent, whach)
                    min_dis = min(min_dis, real_dis)
            #print(f'min_dis = {min_dis} \t max_pivot_dist = {max_pivot_dist} ')
            max_pivot_dist = max(max_pivot_dist, min_dis)

        return max_pivot_dist

    def mtsp_makespan_heuristic_start(self, new_node, pivot):

        citys = range(self.number_of_agent + pivot.__len__() + 1)
        all_pos = list(new_node.location) + list(pivot.keys())
        distance_agent_pivot = {}
        distance_in_pivot = {(i, 0): 0 for i in list(pivot.keys()) + list(range(1, self.number_of_agent + 1))}
        distance_out_start = {(0, i): 0 for i in list(range(1, self.number_of_agent + 1))}

        for i in citys[1: self.number_of_agent + 1]:
            for j in citys[self.number_of_agent + 1:]:
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
                            all_pos[i - 1],
                            all_pos[j - 1])
                        self.old_pivot[sort_pivot] = distance_pivot_pivot[(all_pos[i - 1], all_pos[j - 1])]

        for_plot = [(0, 0)] + all_pos

        all_distance_dict = {**distance_out_start, **distance_in_pivot, **distance_agent_pivot, **distance_pivot_pivot}

        for_plot = [(0, 0)] + all_pos

        # print(f' pivot - {pivot.__len__()}  genarate : {time()-t} ',end='\t')

        t2 = time()
        tmp = self.lp_model.get_makespan(for_plot, self.world, pivot, self.genrate_node, citys, all_distance_dict)
        # print(f'solve all - {time() - t2}  minmax - {tmp} pivot - {pivot.__len__()}')

        return tmp

    def mtsp_makespan_heuristic(self, new_node):
        #print('--------------------------')
        #t1=time()

        tmp_pivot=self.get_pivot(new_node.unseen, new_node.location)
        pivot={pivot:tmp_pivot[pivot] for pivot in tmp_pivot if pivot not in self.pivot_black_list}
        #print(f'pivot - {time() - t1}')
        #t=time()

        if self.huristic_index==2:
            if pivot.__len__()<2:
                return -1
        if pivot.__len__()==0:
                return 0
        citys = range(self.number_of_agent + pivot.__len__() + 1)
        all_pos = list(new_node.location) + list(pivot.keys())
        distance_agent_pivot = {}
        distance_in_pivot = {(i, 0): 0 for i in list(pivot.keys())+list(range(1,self.number_of_agent+1))}
        distance_out_start = {(0, i): 0 for i in list(range(1,self.number_of_agent+1))}

        for i in citys[1: self.number_of_agent + 1]:
            for j in citys[self.number_of_agent + 1:]:
                distance_agent_pivot[(i, all_pos[j-1])] = min([self.real_dis_dic[tuple(sorted((all_pos[i - 1], k)))]
                                                    for k in self.world.dict_wachers[all_pos[j - 1]]])

        distance_pivot_pivot = dict()
        for i in citys[self.number_of_agent + 1:]:
            for j in citys[self.number_of_agent + 1:]:
                if i != j:
                    sort_pivot = tuple(sorted((all_pos[i - 1], all_pos[j - 1])))
                    if sort_pivot in self.old_pivot:
                        distance_pivot_pivot[(all_pos[i - 1], all_pos[j - 1])] = self.old_pivot[sort_pivot]
                    else:
                        distance_pivot_pivot[(all_pos[i - 1], all_pos[j - 1])] = self.get_closest_wachers(all_pos[i - 1], all_pos[j - 1])
                        self.old_pivot[sort_pivot] = distance_pivot_pivot[(all_pos[i - 1], all_pos[j - 1])]


        for_plot = [(0, 0)] + all_pos

        all_distance_dict = {**distance_out_start, **distance_in_pivot, **distance_agent_pivot, **distance_pivot_pivot}

        for_plot=[(0,0)]+all_pos


        #print(f' pivot - {pivot.__len__()}  genarate : {time()-t} ',end='\t')


        #print(f'genarae makespan- {time() - t}')
        t2=time()

        tmp=self.lp_model.get_makespan(for_plot, self.world, pivot, self.genrate_node, citys, all_distance_dict)
        #print(f'pivot - {pivot.__len__()} get_makespan- {time() - t2}  minmax - {tmp }')

        #print(f'all - {time() - t1}')

        return tmp

    def mtsp_sum_of_cost_heuristic(self, new_node):
        x=1

    # def multy_singelton(self,new_node):
    #     max_pivot_dist = 0
    #     # Utils.print_whacers(self.world,new_node.unseen)
    #     tmp_unseen=new_node.unseen
    #     tmp_agent=new_node.location
    #     poslist=[]
    #     for cell in tmp_unseen:
    #         min_dis = 1000000
    #         best_agent =0
    #         best_agentpivot=0
    #
    #         for whach in self.world.dict_wachers[cell]:
    #             if min_dis < max_pivot_dist:
    #                 break
    #             for agent in tmp_agent:
    #                 real_dis = self.get_real_dis(agent, whach)
    #
    #                 if min_dis > real_dis:
    #                     best_agent = agent
    #                     best_agentpivot = whach
    #
    #                 min_dis = min(min_dis, real_dis)
    #
    #         # print(f'min_dis = {min_dis} \t max_pivot_dist = {max_pivot_dist} ')
    #
    #         max_pivot_dist = max(self.get_real_dis(agent, whach), min_dis)


    def get_heuristic(self, new_node):
        if self.huristic_index==0:
           # closest_pivot_dist = self.multy_singelton(new_node)
           closest_pivot_dist = self.singelton_heuristic(new_node)

        elif self.huristic_index==1:
            closest_pivot_dist = max(self.singelton_heuristic(new_node),self.mtsp_makespan_heuristic(new_node))
        elif self.huristic_index==2:
            closest_pivot_dist = self.mtsp_makespan_heuristic(new_node)
            if closest_pivot_dist==-1:
                closest_pivot_dist=self.singelton_heuristic(new_node)
        elif self.huristic_index == 3:
            closest_pivot_dist = self.mtsp_makespan_heuristic(new_node)
        elif self.huristic_index == 4:
            closest_pivot_dist = 0

        return closest_pivot_dist

    def get_new_state(self, old_state, state_index):
        move_index = [0] * self.number_of_agent

        for j in range(self.number_of_agent):
            state_index, index = divmod(state_index, LOS)
            move_index[j] = index

        new_state , moving_status = self.world.get_action(old_state.location, move_index)
        return new_state ,moving_status

    def get_cost(self, old_state):
        return old_state.cost + 1

    def in_open_or_close(self, new_node):  # old_state, new_state, seen_state):
        tmp_state = tuple(sorted(new_node.location))
        if not tmp_state in self.visit_list_dic:
            return False
        return True

    def expend(self):
        old_state = self.pop_open_list()
        self.expend_node+=1
        self.H_expend+=old_state.heuristics
        #print(self.number_of_node,'\t',old_state.heuristics)

        if self.goal_test(old_state.unseen):
            return old_state

        for state_index in range(LOS ** (self.number_of_agent)):

            new_state , moving_status = self.get_new_state(old_state, state_index)

            if self.world.is_valid_node(new_state, old_state,moving_status):

                seen_state = old_state.unseen - self.world.get_all_seen(new_state)
                dead_list=old_state.dead_agent[:]
                for i in range(self.number_of_agent):
                    if moving_status[i]==0 and i not in dead_list:
                        dead_list.append(i)
                new_node = Node(old_state, new_state, seen_state,dead_list, self.get_cost(old_state), 0)

                self.insert_to_open_list(new_node)
                #print(time()-t)
        return False

    def need_to_fix_parent(self, new_node):
        tmp_state = tuple(sorted(new_node.location))

        for index, old_node in enumerate(self.visit_list_dic[tmp_state]):

            if new_node.cost >= old_node.cost and old_node.unseen.issubset(new_node.unseen):
                return False

            elif new_node.cost <= old_node.cost and new_node.unseen.issubset(old_node.unseen):
                old_node.cost = -old_node.cost
                del self.visit_list_dic[tmp_state][index]

                return True

        return True

    def run(self,writer,map_config,start_pos,remove_obs):
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
                                 self.expend_node, self.open_is_beter, self.new_is_beter, remove_obs,
                                 [0] * self.number_of_agent])
                return

        all_path = self.get_path(goal_node)
        if self.genrate_node > 0:
            h_gen = self.H_genrate / self.genrate_node
            h_exp = self.H_expend / self.expend_node
        else:
            h_gen = self.H_genrate
            h_exp = self.H_expend

        # Writes to the file all the parameters of the experiment
        writer.writerow([map_config, start_pos, time() - self.start_time, h_type[self.huristic_index], self.H_start,
                         h_gen, h_exp, self.max_pivot, 0, self.genrate_node, self.expend_node,
                         self.open_is_beter, self.new_is_beter, remove_obs, goal_node.cost])

    def get_path(self, gole_node):
        all_path = [gole_node]
        node = gole_node
        while node.parent.parent is not None:
            node = node.parent
            all_path.append(node)
        return all_path[::-1]

    def print_path(self, gole_node, see_agent_walk):
        all_path = self.get_path(gole_node)
        tmp_location = []
        for cell in all_path:

            #print(f'L = {cell.location} \t h = {cell.heuristics} ')
            tmp_location.append(cell.location)
            tmp_word = np.copy(self.world.grid_map)
            for k in cell.unseen:
                tmp_word[k] = 2
            for j in cell.location:
                tmp_word[j] = 3
            if see_agent_walk:
                plt.figure(1)
                plt.pcolormesh(tmp_word, edgecolors='grey', linewidth=0.01)
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

if __name__ == '__main__':
    #map_type='small_for_exp'
    map_type = 'maze_11_11'
    name = 'soc_all'
    experement_name = f'{map_type}_{name}'
    map_config = f'./config/{map_type}_config.csv'

    row_map = Utils.convert_map(map_config)

    LOS = 4 + 1
    all_free = np.transpose(np.where(np.array(row_map) == 0))

    pivot = [5]
    exp_number = 72
    #  huristics_exp=[2]
    loop_number_of_agent = [2]
    minimize = {'mksp': 0, 'soc': 1}
    huristics_exp = [0, 1,2]

    # if sys.argv:
    #     huristics_exp = [int(sys.argv[1])]
    #     loop_number_of_agent = [int(sys.argv[2])]


    # else:
    # huristics_exp = [0, 1, 2]
    # loop_number_of_agent = [2, 4, 6]

    start_in = 0
    exp_index = 0
    data_file = open(f'{experement_name}_{loop_number_of_agent[0]}_agent_{huristics_exp[0]}_huristic.csv', 'w',newline='\n')
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
                # start_config_as_string = np.loadtxt(f'./config/{map_type}_{number_of_agent}_agent_domain.csv',
                #                                     dtype=tuple, delimiter='\n')
                # all_start_config_as_tupel = [ast.literal_eval(i) for i in start_config_as_string]
                # all_start_config_as_tupel = all_start_config_as_tupel[:exp_number]
                #all_start_config_as_tupel=[((3,3),(3,5))]
                for remove_obs in range(remove_obs_number):
                    start_config_as_string = np.loadtxt(f'./config/{map_type}_{number_of_agent}_agent_domain.csv',
                                                        dtype=tuple, delimiter='\n')
                    all_start_config_as_tupel = [ast.literal_eval(i) for i in start_config_as_string]
                    all_start_config_as_tupel = all_start_config_as_tupel[:exp_number]

                    all_start_config_as_tupel=list(map(tuple,all_free))
                    for start_pos in all_start_config_as_tupel:
                        start_pos=tuple([start_pos]*number_of_agent)

                        for huristic in huristics_exp:
                            if exp_index >= start_in:
                                world = WorldMap(np.array(row_map), LOS)
                                mwrp = Mwrp(world, start_pos, huristic, max_pivot, map_type, minimize['soc'])
                                mwrp.run(writer, map_config, start_pos, remove_obs)
                            bar()