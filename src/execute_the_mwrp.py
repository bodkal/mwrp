import pickle
import time

import numpy as np
from script.utils import Node, Utils, Loger
from mwrp import Mwrp
import ast

import random
from alive_progress import alive_bar


class agent(Mwrp):
    def __init__(self, agent_id, start_location, path, map_type, fixing_metod):
        super().__init__([start_location], 1, map_type)

        self.id = agent_id
        self.path = path[self.id]

        self.old_path = path[self.id]
        self.tmp_path = path[self.id]

        self.location = start_location
        self.seen = set(tuple([start_location]))
        self.didet_see = {}
        self.step_index = -1

        self.is_finis = False
        self.fixing = False
        self.fixing_dict = dict()

        self.fixing_path = []
        self.path_for_plot = []
        self.old_dident_see = set()
        del path[self.id]

        self.all_otear_path = path
        self.all_dident_see = set()
        self.number_of_didet_see = 0

        self.fixing_metod = fixing_metod
        self.all_seen = set()
        self.old_fov = self.dict_fov.copy()

        if fixing_metod in [6,7]:
            self.fix_far_from_path=True
        else:
            self.fix_far_from_path=False

    def move(self):
        try:
            #if self.location == self.path[-1] and self.didet_see.__len__() == 0:
            if self.step_index+1 == self.path.__len__() and self.didet_see.__len__() == 0:
                self.is_finis = True
            else:
                self.step_index += 1
                self.location = self.path[self.step_index]
                if self.location in self.old_path and self.location not in self.didet_see.values():
                    self.fixing = False
            self.see()

        except:
            xxx = 1
            print(1)
            exit()

        return self.seen

    def see(self):
        corrent_seen = self.fov_validation()
        self.seen = self.seen | {self.location} | corrent_seen

    def fov_validation(self):
        fov = self.dict_fov[self.location]

        for fov_cell in fov:
            if fov_cell not in self.seen:
                if self.didet_see.__len__() < 1 and fov_cell not in self.path:
                    # dident see a cell that need to see
                    if random.randint(0, 10) == 0 and not self.fixing:
                        if self.do_we_need_too_see_cell(fov_cell):
                            self.didet_see[fov_cell] = 0
                            self.all_dident_see.add(fov_cell)
                            self.is_finis = False

                            print(f"agent {self.id} new need to see : {fov_cell}")
                            self.number_of_didet_see += 1
                        else:
                            print(f"agent {self.id} dosent need to see : {fov_cell}")

                else:
                    # see a cell that dident see in the past
                    if fov_cell in self.all_dident_see:
                        # TODO retorn to 50
                        if random.randint(0, 50) == 0:
                            print(f"remove from need to see by miracle: {fov_cell}")
                            if fov_cell in self.didet_see:
                                del self.didet_see[fov_cell]
                            self.all_dident_see = self.all_dident_see - {fov_cell}

        seen_fov = fov - {cell for cell in self.all_dident_see}
        return seen_fov

    def see_unseen_by_stand(self):
        if self.location in self.didet_see:
            self.remove_from_dident_see(self.location)
            print(f"agent {self.id} remove from need to see by stand: {self.location}")
            return True
        return False

    def cell_to_fix_from_close(self, didet_see):
        all_dis = {didet_see: []}
        for path_cell in self.path[self.step_index:]:
            all_dis[didet_see].append(self.real_dis_dic[tuple(sorted((path_cell, didet_see)))])

        if self.path[self.step_index:].__len__() != 0:
            insert_index = self.step_index + int(all_dis[didet_see].index(min(all_dis[didet_see])))
            min_dis = self.path[insert_index]
        else:
            min_dis = self.path[-1]
        return min_dis

    def do_we_need_too_see_cell(self, cell):
        # if cell in self.get_multy_fov(self.old_path) and cell not in self.all_dident_see:
        if cell not in self.all_dident_see:
            if self.fix_far_from_path:
                return True
            elif cell in self.get_multy_fov(self.old_path):
                return True

        return False

    def return_to_path(self, start):
        if self.step_index < self.old_path.__len__() and set(self.didet_see.values()).issubset(self.old_path) and set(
                self.fixing_dict).issubset(self.old_path):
            tmp_dic = {goal: self.real_dis_dic[tuple(sorted((start, goal)))] for goal in self.old_path}
            tmp_dic = {goal: tmp_dic[goal] for goal in tmp_dic if self.path[self.step_index:].__len__() > tmp_dic[goal]}

            sorted_tmp_dic = sorted(tmp_dic, key=tmp_dic.get)

            for cell in sorted_tmp_dic:
                fixing_path = self.BFS.get_path(start, cell)
                if fixing_path[-1] in self.path[self.step_index:]:
                    connect_index = self.path[self.step_index:].index(fixing_path[-1])
                else:
                    connect_index = 0
                tmp_path = self.path[:self.step_index] + [self.location] + fixing_path[:-1] + self.path[
                                                                                              connect_index + self.step_index:]
                if tmp_path.__len__() > self.path.__len__():
                    return
                if self.get_multy_fov(self.old_path).issubset(self.get_multy_fov(self.tmp_path)):
                    self.path = tmp_path
                    return

    def go_to_cell_by_cell(self, start, goal):
        if goal == self.old_path[-1]:
            self.path = self.path[:self.step_index] + [self.location]
        else:
            fixing_path = self.BFS.get_path(start, goal)
            self.path = self.path[:self.step_index] + [self.location] + fixing_path + self.path[
                                                                                      self.step_index + self.path[
                                                                                                        self.step_index:].index(
                                                                                          goal) + 1:]

        print(f"agent {self.id} retarn to path from {start} to {goal}")

    def get_fixing_path(self, start, goal):
        fixing_path = self.BFS.get_path(start, goal)

        if self.step_index == 0:
            self.path = self.path[:self.step_index] + fixing_path + fixing_path[::-1][1:] + self.path[self.step_index:]
        else:
            self.path = self.path[:self.step_index] + [self.location] + fixing_path + fixing_path[::-1][1:] + \
                        self.path[self.step_index:]

        self.fixing = True

    def remove_from_dident_see(self, unseen):
        del self.didet_see[unseen]
        self.seen = self.seen | {unseen}

    def get_close_cell(self, cell):

        min_dis = (self.location, self.real_dis_dic[tuple(sorted((cell, self.location)))])
        for path_cell in self.path[self.step_index + 1:]:
            dis = self.real_dis_dic[tuple(sorted((cell, path_cell)))]
            if dis < min_dis[1]:
                min_dis = (path_cell, dis)
        return min_dis

    def update_dident_see(self):
        for cell in self.didet_see:
            self.didet_see[cell] = self.get_close_cell(cell)[0]

    def get_key(self, tmp_dir):
        return list(tmp_dir.keys())[0]

    def get_altrntive_path_through_cell(self, didet_see):

        tmp_didet_see = {x: self.real_dis_dic[tuple(sorted((x, self.location)))] for x in didet_see}

        dident_see = min(tmp_didet_see, key=tmp_didet_see.get)
        path_len = self.path.__len__() - 1
        all_path_to_cell = {}
        can_walk_on_path = 1

        if self.step_index < self.path.__len__():
            tmp_path_to_cell = {
                i + self.step_index: [self.path[i + self.step_index]] + self.BFS.get_path(data, dident_see) for i, data
                in enumerate(self.path[self.step_index:])}
        else:
            tmp_path_to_cell = {self.step_index: [self.path[-1]] + self.BFS.get_path(self.path[-1], dident_see)}

        while all_path_to_cell.__len__() == 0:
            all_path_to_cell = {i: tmp_path_to_cell[i] for i in tmp_path_to_cell if
                                (set(tmp_path_to_cell[i]) & set(self.path)).__len__() == can_walk_on_path}
            can_walk_on_path += 1

        tmp_all_path = {(i, j): all_path_to_cell[i] + all_path_to_cell[j][::-1][1:] for i in all_path_to_cell for j in
                        all_path_to_cell if i <= j}

        if (path_len, path_len) in tmp_all_path:
            tmp_all_path[(path_len, path_len)] = tmp_path_to_cell[path_len]

        tmp_need_to_see = {cell: self.get_multy_fov(self.path[cell[0]:cell[1]]) for cell in tmp_all_path}

        Optional_path = {i: tmp_all_path[i] for i in tmp_all_path if
                         tmp_need_to_see[i].issubset(self.get_multy_fov(tmp_all_path[i]))}

        return min(Optional_path.items(), key=lambda item: item[1].__len__() - (item[0][1] - item[0][0]))

    def is_cells_in_path(self, cells):
        if cells in self.path[self.step_index:]:
            return True
        return False

    def fix_dident_see(self):
        if self.fixing_metod == 0:
            self.fix_solo_go_to_close()

        elif self.fixing_metod == 2:
            self.fix_solo_minimal_path()

        elif self.fixing_metod == 4:
            self.fix_solo_singet_wrp()

    def fix_solo_minimal_path(self):

        _ = self.fix_by_see_or_stend()

        if self.didet_see.__len__() > 0:
            if not set(self.didet_see).issubset(self.path):
                key, selected_path = self.get_altrntive_path_through_cell(self.didet_see)
                self.tmp_path = self.path[:key[0]] + selected_path + self.path[key[1] + 1:]
                self.didet_see[self.get_key(self.didet_see)] = selected_path[0]

            if self.location in self.didet_see.values():
                self.fixing_dict = {self.location: self.get_key(self.didet_see)}
                self.path = self.tmp_path.copy()

    def fix_solo_go_to_close(self):

        _ = self.fix_by_see_or_stend()
        if self.didet_see.__len__() > 0:
            fixing_cell = list(self.fixing_dict.values())
            if self.fixing and fixing_cell[0] not in self.path[:self.step_index + 1] and fixing_cell[
                0] in self.path[self.step_index:] and fixing_cell[0] not in self.didet_see:
                self.go_to_cell_by_cell(self.location, list(self.fixing_dict.keys())[0])

            self.update_dident_see()
            if self.location in self.didet_see.values():
                self.fixing_dict = {self.location: self.get_key(self.didet_see)}
                key = Utils.get_key_from_value(self.didet_see, self.location)
                self.get_fixing_path(self.location, key)

    def fix_solo_singet_wrp(self):
        _ = self.fix_by_see_or_stend()
        if self.didet_see.__len__() > 0 and False in [i in self.path[self.step_index:] for i in self.didet_see]:
            unseen = (self.get_multy_fov(self.old_path) - self.all_seen) | set(self.didet_see)
            new_path = self.run_new_serch(self.didet_see, unseen, tuple([self.location]), [self.step_index])
            self.path[self.step_index:] = new_path[0]

    def fix_by_see_or_stend(self):
        self.see_unseen_by_stand()
        if self.didet_see.__len__() > 0:
            return self.any_one_see_cell()
        return False

    def any_one_see_cell(self):
        see_somting = False
        for unseen_cell in self.didet_see.copy():
            # for unseen_cell in self.all_seen & set(agent.didet_see):
            if unseen_cell in self.all_seen:
                self.remove_from_dident_see(unseen_cell)
                print(f"agent {self.id} remove by see: {unseen_cell}  locate at {self.location}")
                self.number_of_didet_see -= 1
                see_somting = True
        return see_somting

    def run_new_serch(self, didet_see, unseen, location, cost):

        start_node = Node(None, location, unseen, cost, 1)
        self.clean_serch(start_node, set(didet_see), self.old_fov)
        self.remove_from_fov(set(didet_see))

        print("start calculate path")
        new_path = self.run(save_to_file=False, need_path=True)
        print("finis calculate path")

        return new_path


class ExecuteTheMwrp(Mwrp):

    def __init__(self, sp: tuple, fixing_metod: int, minimize: int, map_type: list) -> None:
        """
        :param map_type:        name of the map that we want to solve
        :param sp:              all robot start point ((x1,y1),(x2,y2))
        :param minimize:        What we want to minimize (soc, mksp)
        :param fixing_metod:    the metod for fixing if we dident see some cell
        """

        file_name = f"config/path/{sp.__len__()}_agent/{map_type}_{sp}_{minimize}"
        with open(file_name, "rb") as output_file:
            path = pickle.load(output_file)

        super().__init__(sp, minimize, map_type)

        self.minimize = minimize

        self.step_index = -1
        self.all_seen = set()
        self.all_agent = [agent(agent_index, path[agent_index][0], path, map_type, fixing_metod) for agent_index in
                          range(path.__len__())]
        self.fix_path_asiment = {}
        self.fixing_metod = fixing_metod
        self.old_fov = self.dict_fov.copy()

    def get_close_agent(self, agent):

        tmp_dir = {}
        if agent.didet_see.__len__() > 0 and 0 in agent.didet_see.values():
            for i in self.all_agent:
                key, selected_path = i.get_altrntive_path_through_cell(agent.didet_see)

                if self.minimize == 1 or self.fixing_metod == 7:
                    tmp_dir[i.id] = (key, selected_path.__len__() - (key[1] - key[0]), selected_path)
                elif self.minimize == 0:
                    tmp_dir[i.id] = (key, i.path.__len__() - (key[1] - key[0]) + selected_path.__len__(), selected_path)

            _id, data = min(tmp_dir.items(), key=lambda item: item[1][1])
            return _id, data
        return agent.id, None

    def get_close_agent_v2(self, agent,didet_see):

        tmp_dir = {}
        if agent.didet_see.__len__() > 0 and 0 in agent.didet_see.values():
            for i in self.all_agent:
                key, selected_path = i.get_altrntive_path_through_cell([didet_see])

                if self.minimize == 1 or self.fixing_metod == 7:
                    tmp_dir[i.id] = (key, selected_path.__len__() - (key[1] - key[0]), selected_path)
                elif self.minimize == 0:
                    tmp_dir[i.id] = (key, i.path.__len__() - (key[1] - key[0]) + selected_path.__len__(), selected_path)

            _id, data = min(tmp_dir.items(), key=lambda item: item[1][1])
            return _id, data
        return agent.id, None


    def move_cell_bitwine_agent(self, agent, _id):
        if _id != agent.id:
            print(f"agent {agent.id} move {agent.didet_see.keys()} to agent {_id} ")
            self.all_agent[_id].didet_see[agent.get_key(agent.didet_see)] = 0
            del agent.didet_see[agent.get_key(agent.didet_see)]
            self.all_agent[_id].is_finis = False
        else:
            agent.didet_see[agent.get_key(agent.didet_see)] = 0

    def move_cell_bitwine_agent_v2(self,cell, agent, _id):
        if agent.didet_see.__len__() > 0:
            if _id != agent.id:
                print(f"agent {agent.id} move {cell} to agent {_id} ")
                self.all_agent[_id].didet_see[cell] = 0
                del agent.didet_see[cell]
                self.all_agent[_id].is_finis = False
            else:
                agent.didet_see[cell] = 0

    def fix_mangment_multy(self,agent):
        if agent.didet_see.__len__() > 0:

            _id, data = self.get_close_agent(agent)
            self.move_cell_bitwine_agent(agent, _id)

            if self.fixing_metod==1:
                self.all_agent[_id].fix_solo_minimal_path()
                if _id != agent.id:
                    agent.fix_solo_minimal_path()
            elif self.fixing_metod==3:
                self.all_agent[_id].fix_solo_minimal_path()
                if _id != agent.id:
                    agent.fix_solo_minimal_path()
            elif self.fixing_metod==5:
                self.all_agent[_id].fix_solo_singet_wrp()
                if _id != agent.id:
                    agent.fix_solo_singet_wrp()



    def run_new_serch(self, didet_see, unseen, location, cost):

        start_node = Node(None, location, unseen, cost, 1)
        self.clean_serch(start_node, set(didet_see), self.old_fov)
        self.remove_from_fov(set(didet_see))




        print("start calculate path")
        status,new_path ,old_exp,old_time,old_cost  = self.run(save_to_file=False, need_path=True)
        print("finis calculate path")

        return status, new_path ,old_exp,old_time,old_cost

    def fix_mangment_prsral_mwrp(self):

        all_id=self.mangmet_cell_moving()
        location = []
        cost = []
        all_path = set()
        for agent in self.all_agent:
            if agent.didet_see.__len__() > 0:
                _ = agent.fix_by_see_or_stend()
            all_path = all_path | set(agent.path[agent.step_index:])

        all_didet_see = set()

        if self.fixing_metod==6:
            unseen = set()
            for i in all_id.copy():
                if not set(self.all_agent[i].didet_see).issubset(self.all_agent[i].path[self.all_agent[i].step_index:]):
                    unseen = unseen | self.get_multy_fov(self.all_agent[i].old_path) | set(self.all_agent[i].didet_see)
                    all_didet_see = all_didet_see | set(self.all_agent[i].didet_see.keys())
                    location.append(self.all_agent[i].location)
                    cost.append(self.all_agent[i].step_index)
                else:
                    all_id = all_id - {i}

        elif self.fixing_metod==7:
            for agent in self.all_agent:
                all_didet_see = all_didet_see | set(agent.didet_see)
                location.append(agent.location)
                cost.append(agent.step_index)
            unseen = self.free_cell[1] - self.all_seen
            all_id=range(self.number_of_agent)

        if all_id.__len__() > 0 and not all_didet_see.issubset(all_path):
            unseen = unseen - self.all_seen
            new_path = self.run_new_serch(all_didet_see, unseen, tuple(location), cost)
            if new_path==False:
                return False
            for index, data in enumerate(all_id):
                self.all_agent[data].path[self.all_agent[data].step_index:] = new_path[index]
                self.all_agent[data].is_finis=False
                for end_index, cell in enumerate(self.all_agent[data].path[::-1]):
                    if cell not in self.all_agent[data].path[:-end_index - 1] or cell == self.all_agent[data].location:
                        break
                if end_index != 0:
                    self.all_agent[data].path = self.all_agent[data].path[:-end_index]

        self.move_cell_to_agent_path(self.all_agent)



    def mangmet_cell_moving(self):
        all_id = set()
        for agent in self.all_agent:
            for didet_see in agent.didet_see.copy():
                if not set(agent.didet_see).issubset(agent.path):
                    _id, data = self.get_close_agent_v2(agent,didet_see)
                    self.move_cell_bitwine_agent_v2(didet_see,agent, _id)
                    if _id != agent.id and agent.didet_see.__len__() > 0:
                        all_id = all_id | {agent.id}
                    all_id = all_id | {_id}
        return all_id

    def move_cell_to_agent_path(self,all_agent):
        for agent in all_agent:
            for didet_see in agent.didet_see.copy():
                if not agent.is_cells_in_path(didet_see):
                    for _id in range(all_agent.__len__()):
                        if all_agent[_id].is_cells_in_path(didet_see):
                            self.move_cell_bitwine_agent_v2(didet_see,agent, _id)
                            break



    def update_agent_new_dident_see_and_new_see(self):
        tmp_all_dident_see = {list(agent.didet_see.keys())[0] for agent in self.all_agent if
                              agent.didet_see.__len__() > 0}
        for agent in self.all_agent:
            agent.all_dident_see = tmp_all_dident_see
            agent.all_seen = self.all_seen

    def one_step(self):
        self.step_index += 1
        for one_agent in self.all_agent:
            if not one_agent.is_finis:
                one_agent_see = one_agent.move()
                self.all_seen = self.all_seen | one_agent_see

    def fix_dident_see_if_need(self):
        for one_agent in self.all_agent:

            if self.fixing_metod in [0, 2, 4]:
                one_agent.fix_dident_see()

            elif self.fixing_metod in [1,3,5]:
                self.fix_mangment_multy(one_agent)

        if self.fixing_metod in  [6,7]:
            status=self.fix_mangment_prsral_mwrp()
            return status


    def print_all_path_and_unseen(self):
        print("\n")
        for i in self.all_agent:
            print(i.path[i.step_index:])
        print(self.free_cell[1] - self.all_seen)
        print("\n")

    def exaxute_path(self, need_to_see=False):
        while not min([agent.is_finis for agent in self.all_agent]) :
            self.one_step()
            self.update_agent_new_dident_see_and_new_see()
            status=self.fix_dident_see_if_need()
            if status==False:
                return False
            if need_to_see:
                Utils.print_exexute(self.grid_map, self.all_agent)
                self.print_all_path_and_unseen()

            # sleep(0.2)
        if self.free_cell[0] != self.all_seen.__len__():
            print(f"teory :{self.free_cell[0]} \t realty :{self.free_cell[1] - self.all_seen}")
            exit()
        return True


if __name__ == '__main__':
    map_type = 'rooms_15_15'
    #map_type = 'maze_11_11'

    name = 'mwrp_vs_execusen'

    # all_free = np.transpose(np.where(np.array(row_map) == 0))

    minimize = {'mksp': 0, 'soc': 1}

    minimize = minimize['mksp']
    experement_name = f'{map_type}_{name}_{minimize}'
    huristics = 3

    random_seed = [340.09, 765.55, 812.1, 475.66, 287.55, 80.27, 24.76, 403.98, 959.14, 183.61, 728.34, 525.18, 942.96,
                   916.23, 879.31, 529.71, 403.13, 127.63, 791.69, 984.85, 373.05, 298.64, 928.13, 842.73, 509.7,
                   984.41, 885.83, 800.5, 125.54, 984.07, 139.77, 536.54, 778.7, 955.12, 664.76, 373.07, 98.73, 46.71,
                   522.38, 472.86, 376.03, 613.87, 130.7, 37.21, 11.33, 380.66, 963.73, 217.79, 107.09, 369.3]

    with alive_bar(50 * 10) as bar:

        for number_of_agent in [5]:

            exp_index = 3
            start_pos = [ast.literal_eval(i) for i in
                         np.loadtxt(f'./config/{map_type}_{number_of_agent}_agent_domain.csv', dtype=tuple,
                                    delimiter='\n')]
            fixing_metod = 7

            data_file = f'{experement_name}_{number_of_agent}_agent_{huristics}_huristic_{fixing_metod}_fixing.csv'

            titel_list = ['map_name', 'start_state', 'time', 'minimize', 'cost', 'expend', 'number_didet_see',
                          'repet_benxmark', 'didet_see']
            loger = Loger(data_file, titel_list)


            for sp in start_pos[32:50]:


                for repet in range(10):
                    random.seed(random_seed[repet+2])
                    print(f"repet={repet}")

                    execute = ExecuteTheMwrp(sp, fixing_metod, minimize, map_type)

                    start_pos = tuple(agent.location for agent in execute.all_agent)
                    unseen_start = execute.free_cell[1] - execute.get_all_seen(start_pos)

                    if repet==0:
                        status, _, old_exp, old_time, old_cost = execute.run_new_serch({}, unseen_start, start_pos, [0] * execute.all_agent.__len__())
                        loger.write([map_type, sp, old_time * status - (1 - status), minimize, old_cost,old_exp, 0, 0, []])

                    time1 = time.time()


                    old_exp_all=[]
                    need_to_see=set()



                    need_to_see.add(random.sample(unseen_start, 1)[0])
                    status,_ ,old_exp,old_time,old_cost = execute.run_new_serch(need_to_see,unseen_start, start_pos, [0]*execute.all_agent.__len__())
                    loger.write([map_type, sp, old_time * status - (1 - status), minimize, old_cost, old_exp, 1, repet,
                                 list(need_to_see)])

                    need_to_see.add(random.sample(unseen_start, 1)[0])
                    status,_ ,old_exp,old_time,old_cost = execute.run_new_serch(need_to_see,unseen_start, start_pos, [0]*execute.all_agent.__len__())
                    loger.write([map_type, sp, old_time * status - (1 - status), minimize, old_cost, old_exp, 2, repet,
                                 list(need_to_see)])

                    need_to_see.add(random.sample(unseen_start, 1)[0])
                    status,_ ,old_exp,old_time,old_cost = execute.run_new_serch(need_to_see,unseen_start, start_pos, [0]*execute.all_agent.__len__())
                    loger.write([map_type, sp, old_time * status - (1 - status), minimize, old_cost, old_exp, 3, repet,
                                 list(need_to_see)])

                    bar()
