import numpy as np
from script.utils import Node, Utils
from time import time
from random import sample


class Astar:
    def __init__(self, grath, cost, start):
        self.number_of_agent=start.__len__()
        self.grath=grath
        self.cost=cost
        need_to_be=self.grath[start[0]]-set(start)

        start_node = Node(Node(None, start, 0), start, need_to_be)

        self.visit_list_dic = {tuple(sorted(start)): [start_node]}
        self.open_list = [start_node]


    def find_index_to_open(self, new_node):
        all_cost_estimate = new_node.cost + new_node.heuristics
        for index, data in enumerate(self.open_list):
            if data.cost + data.heuristics == all_cost_estimate:
                if data.need_to_be.__len__() > new_node.need_to_be.__len__():
                    return index
                elif data.need_to_be.__len__() == new_node.need_to_be.__len__():
                    if data.cost < new_node.cost:
                        return index
            elif data.cost + data.heuristics > all_cost_estimate:
                return index
        return len(self.open_list)

    def insert_to_open_list(self, new_node):
        new_node.heuristics = self.get_heuristic(new_node)
        index = self.find_index_to_open(new_node)
        self.open_list.insert(index, new_node)
        return new_node

    def pop_open_list(self):
        if len(self.open_list):
            pop_open_list = self.open_list.pop(0)
        else:
            pop_open_list = 0
        return pop_open_list

    def move_from_open_to_close(self, index=0):
        self.open_list = np.delete(self.open_list, index)

    def get_heuristic(self, state):
        return sum(abs(self.goal - state))  # np.linalg.norm(state - self.goal)

    def get_action(self, state, move_index):
        action = []
        for i in move_index:
            action = np.hstack((action, self.action[i]))

        return (action).astype(int) + state

    def get_path(self, gole_node):
        all_path = np.array([self.goal])
        node = gole_node
        while node.cost:
            print(node.Location, end='')
            all_path = np.vstack((all_path, node.Location))
            node = self.close_list_dic[node.Location.__str__()]
        all_path = np.vstack((all_path, self.start))

        return all_path

    def get_path(self, gole_node):
        all_path = [gole_node]
        node = gole_node
        cost = 0
        while node.parent.parent is not None:
            cost += node.cost
            node = node.parent
            all_path.append(node)
        return all_path[::-1], cost

    def expend(self):
        old_state = self.pop_open_list()
        #for agent_action in range(self.number_of_agent):

        new_state = self.get_new_state(old_state)

        #self.number_of_node += 1
        for state in new_state:
            for i in state:
                need_to_be = old_state.need_to_be - {i}

            if self.goal_test(need_to_be):
                new_node = Node(old_state, new_state, seen_state, old_state.cost + 1, 0)
                print(f"find solution in -> {time() - self.start_time} sec at cost of {new_node.cost}"
                      f" and open {self.number_of_node} node")
                return new_node

            new_cost = self.get_cost(old_state)

            new_node = Node(old_state, new_state, seen_state, new_cost, 0)

            if not self.in_open_or_close(new_node):
                self.insert_to_open_list(new_node)
                self.visit_list_dic[tuple(sorted(new_node.location))] = [new_node]
            elif self.need_to_fix_parent(new_node):
                self.insert_to_open_list(new_node)
                self.visit_list_dic[tuple(sorted(new_node.location))].append(new_node)

            return False

    def goal_test(self, map_state):
        if not map_state.__len__():
            return True
        return False


    def get_new_state(self, old_state):
        move_index=[]
        for index in range(self.number_of_agent):
            for new_cell in old_state.need_to_be:
                tmp_stat = list(old_state.location)
                tmp_stat=tuple(sorted(tmp_stat[:index]+[new_cell]+tmp_stat[1+index:]))
                if tmp_stat not in move_index:
                    move_index.append(tmp_stat)
        return move_index

    def get_cost(self, old_state,):
        return old_state.cost + 1

    def run(self):
        self.start_time = time()
        print("\nstart algoritem ... ", end='')

        gole_node = False
        while not gole_node:
            gole_node = self.expend()
            continue
        _, cost = self.get_path()

        return cost


class Utils:

    def __init__(self, Location, cost=0, heuristics=0):
        self.Location = Location
        self.cost = cost
        self.heuristics = heuristics


class Node:
    def __init__(self, parent, location, need_to_be, cost=0, heuristics=0):
        self.parent = parent
        self.location = location
        self.need_to_be = need_to_be
        self.cost = cost
        self.heuristics = heuristics


class WorldMap:

    def __init__(self, graf_map):
        self.grath_map = graf_map

if __name__=='__main__':
    i=0
    cost=dict()
    start=((1,1),(7,2))
    grath={(1,1) : {(2,2),(3,3),(4,4)},(2,2) : {(1,1),(3,3),(4,4)},(3,3) : {(2,2),(1,1),(4,4)},(4,4) : {(2,2),(1,1),(3,3)}}
    for key ,data in grath.items():
        for cell in data:
            i+=1
            new_cell=tuple(sorted([key,cell]))
            if new_cell not in cost.keys():
                cost[new_cell]= i
    a=Astar(grath,cost,start)
    a.run()