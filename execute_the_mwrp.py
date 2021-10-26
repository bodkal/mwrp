import numpy as np
from script.utils import Utils
from script.world import WorldMap
from mwrp import Mwrp
import ast
import pprint
from time import sleep
import csv
import random
from alive_progress import alive_bar


class agent:
    def __init__(self, agent_id, start_location, path, world):
        self.id = agent_id
        self.path = path[self.id]
        self.old_path = path[self.id]
        self.tmp_path = path[self.id]

        self.location = start_location
        self.seen = set()
        self.didet_see = {}
        self.step_index = -1
        self.fixing_step_index = -1

        self.is_finis = False
        self.fixing = False

        self.world = world
        self.fixing_path = []
        self.path_for_plot = []
        self.old_dident_see = set()
        del path[self.id]
        self.all_otear_path = path
        self.all_dident_see = set()
        # self.unseen={self.world.get_multy_fov(self.path)}

    def move(self):
        self.step_index += 1
        if self.location == self.path[-1] and self.didet_see.__len__() == 0:
            self.is_finis = True
        else:
            self.location = self.path[self.step_index]
        self.see()

        return self.seen

    def move_return_to_path(self):
        self.step_index += 1
        if self.location == self.path[-1]:
            self.is_finis = True
        else:
            self.location = self.path[self.step_index]

    def see(self):
        corrent_seen = self.fov_validation()
        self.seen = self.seen | {self.location} | corrent_seen
        # self.unseen = self.unseen -  self.seen

    def fov_validation(self):
        fov = self.world.dict_fov[self.location]

        for fov_cell in fov:
            if fov_cell not in self.seen:
                if self.didet_see.__len__() < 1 and fov_cell not in self.path:
                    # dident see a cell that need to see
                    if random.randint(0, 10) == 0 and not self.fixing:
                        self.didet_see[fov_cell] = 0
                        self.all_dident_see.add(fov_cell)
                        print(f"agent {self.id} new need to see : {fov_cell}")
                else:
                    # see a cell that dident see in the past
                    if fov_cell in self.all_dident_see:
                        if random.randint(0, 50) == 0:
                            print(f"remove from need to see by miracle: {fov_cell}")
                            del self.didet_see[fov_cell]
                            self.all_dident_see = self.all_dident_see - {fov_cell}

        seen_fov = fov - {cell for cell in self.all_dident_see}
        return seen_fov

    def see_unseen_by_stand(self, location):
        if location in self.didet_see:
            self.remove_from_dident_see(location)
            print(f"agent {self.id} remove from need to see by stand: {location} locate at {self.location}")
            return True
        return False

    def cell_to_fix_from_close(self, didet_see):
        all_dis = {didet_see: []}
        for path_cell in self.path[self.step_index:]:
            all_dis[didet_see].append(self.world.real_dis_dic[tuple(sorted((path_cell, didet_see)))])

        if self.path[self.step_index:].__len__() != 0:
            insert_index = self.step_index + int(all_dis[didet_see].index(min(all_dis[didet_see])))
            min_dis = self.path[insert_index]
        else:
            min_dis = self.path[-1]
        return min_dis

    def return_to_path(self, start, goal):
        fixing_path = self.world.BFS.get_path(start, goal)

        self.path = self.path[:self.step_index] + [self.location] + fixing_path + self.path[self.fixing_step_index:]
        print(f"agent {self.id} retarn to path from {start} to {goal}")

    def get_fixing_path(self, start, goal):
        fixing_path = self.world.BFS.get_path(start, goal)
        if self.step_index == 0:
            self.fixing_step_index = (fixing_path.__len__() * 2 - 1)
            self.path = self.path[:self.step_index] + fixing_path + fixing_path[::-1][1:] + self.path[self.step_index:]
        else:
            self.fixing_step_index = (fixing_path.__len__() * 2) + self.step_index
            self.path = self.path[:self.step_index] + [self.location] + fixing_path + fixing_path[::-1][1:] + \
                        self.path[self.step_index:]
        self.fixing = True

    def remove_from_dident_see(self, unseen):
        del self.didet_see[unseen]
        # self.all_dident_see=self.all_dident_see-{unseen}
        self.seen = self.seen | {unseen}

    def get_close_cell(self, cell):

        min_dis = (self.location, self.world.real_dis_dic[tuple(sorted((cell, self.location)))])
        for path_cell in self.path[self.step_index + 1:]:
            dis = self.world.real_dis_dic[tuple(sorted((cell, path_cell)))]
            if dis < min_dis[1]:
                min_dis = (path_cell, dis)
        return min_dis

    def update_dident_see(self):
        self.didet_see = {cell: self.get_close_cell(cell)[0] for cell in self.didet_see if self.didet_see[cell] == 0}

    def get_key(self,tmp_dir):
        return  list(tmp_dir.keys())[0]

    def get_altrntive_path_through_cell(self):
        dident_see = self.get_key(self.didet_see)
        path_len = self.path.__len__() - 1
        all_path_to_cell = {}
        can_walk_on_path = 1

        tmp_path_to_cell = {
            i + self.step_index: [self.path[i + self.step_index]] + self.world.BFS.get_path(data, dident_see)
            for i, data in enumerate(self.path[self.step_index:])}

        while all_path_to_cell.__len__() == 0:
            all_path_to_cell = {i: tmp_path_to_cell[i] for i in tmp_path_to_cell if
                                (set(tmp_path_to_cell[i]) & set(self.path)).__len__() == can_walk_on_path}
            can_walk_on_path += 1

        tmp_all_path = {(i, j): all_path_to_cell[i] + all_path_to_cell[j][::-1][1:] for i in all_path_to_cell
                        for j in all_path_to_cell if i <= j}
        if (path_len, path_len) in tmp_all_path:
            tmp_all_path[(path_len, path_len)] = tmp_path_to_cell[path_len]
        tmp_need_to_see = {cell: self.world.get_multy_fov(self.path[cell[0]:cell[1]]) for cell in tmp_all_path}

        Optional_path = {i: tmp_all_path[i] for i in tmp_all_path if
                         tmp_need_to_see[i].issubset(self.world.get_multy_fov(tmp_all_path[i]))}

        return min(Optional_path.items(), key=lambda item: item[1].__len__() - (item[0][1] - item[0][0]))

class ExecuteTheMwrp:

    def __init__(self, world: WorldMap, path: dict, fixing_metod: int, minimize: int) -> None:
        """
        :param world:           An WorldMap  object from script.world that contains the parameters of the world
        :param path:            all robot path sort as dict {0:(x1,y1),(x2,y2)... 1:(x1,y1),(x2,y2)... }
        :param minimize:        What we want to minimize (soc, mksp)
        :param fixing_metod:    the metod for fixing if we dident see some cell
        """

        self.minimize = minimize
        self.world = world
        self.step_index = -1
        self.all_seen = set()
        self.all_agent = [agent(agent_index, path[agent_index][0], path, world) for agent_index in
                          range(path.__len__())]
        self.fix_path_asiment = {}
        self.fixing_metod = fixing_metod

    def fix_by_see_or_stend(self,agent):
        agent.see_unseen_by_stand(agent.location)
        if agent.didet_see.__len__() > 0:
            self.any_one_see_cell(agent)

    def fix_mangment_solo_go_to_close(self, agent):
        self.fix_by_see_or_stend(agent)

        if agent.didet_see.__len__() > 0:
            if 0 in agent.didet_see.values():
                agent.update_dident_see()
            for unseen_cell in self.all_seen & {cell for cell in agent.didet_see}:
                if unseen_cell in agent.didet_see:
                    if agent.fixing and agent.didet_see not in agent.path[:agent.step_index]:
                        agent.return_to_path(agent.location, agent.path[agent.fixing_step_index])
                    agent.remove_from_dident_see(unseen_cell)
                    print(f"agent {agent.id} remove by see: {unseen_cell}  locate at {agent.location}")

            if agent.location in agent.didet_see.values():
                key = Utils.get_key_from_value(agent.didet_see, agent.location)
                agent.get_fixing_path(agent.location, key)

    def fix_mangment_solo_minimal_path(self, agent):
        self.fix_by_see_or_stend(agent)

        if agent.didet_see.__len__() > 0:
            if not agent.fixing and agent.didet_see not in agent.path:
                key, selected_path = agent.get_altrntive_path_through_cell()

                agent.tmp_path = agent.path[:key[0]] + selected_path + agent.path[key[1] + 1:]
                agent.didet_see[ agent.get_key(agent.didet_see)] = selected_path[0]

            if agent.location in agent.didet_see.values():
                agent.path = agent.tmp_path

    def fix_mangment_multy_minimal_path(self, agent):
        self.fix_by_see_or_stend(agent)

        if agent.didet_see.__len__() > 0:
            if not agent.fixing and agent.didet_see not in agent.path:
                key, selected_path = agent.get_altrntive_path_through_cell()

                agent.old_path = agent.path[:key[0]] + selected_path + agent.path[key[1] + 1:]
                agent.didet_see[agent.get_key(agent.didet_see)] = selected_path[0]

            if agent.location in agent.didet_see.values():
                tmp_path = agent.old_path
                agent.old_path = agent.path
                agent.path = tmp_path

    def fix_mangment_multy_go_to_close(self, agent):

        self.fix_by_see_or_stend(agent)

        if agent.didet_see.__len__() > 0:
            if 0 in agent.didet_see.values():
                agent.update_dident_see()

            dident_see = list(agent.didet_see.keys())[0]
            close_cell = [i.get_close_cell(dident_see) for i in self.all_agent]
            close_cell_index = close_cell.index(
                min([i.get_close_cell(dident_see) for i in self.all_agent], key=lambda x: x[1]))
            if close_cell_index != agent.id:
                agent.remove_from_dident_see(dident_see)
                if close_cell[close_cell_index][1] > 0:
                    self.all_agent[close_cell_index].didet_see[dident_see] = close_cell[close_cell_index][0]
                    self.all_agent[close_cell_index].old_dident_see.add(dident_see)
                    self.fix_mangment_multy_go_to_close(self.all_agent[close_cell_index])
                    print(f"move cell {dident_see} from id {agent.id} to id {self.all_agent[close_cell_index].id}")
                else:
                    print(f"move cell {dident_see} is in {self.all_agent[close_cell_index].id} path")

        if agent.didet_see.__len__() > 0:

            if agent.location in agent.didet_see.values():
                key = Utils.get_key_from_value(agent.didet_see, agent.location)
                agent.get_fixing_path(agent.location, key)

    def update_dident_see(self):

        tmp_all_dident_see = {list(agent.didet_see.keys())[0] for agent in self.all_agent if
                              agent.didet_see.__len__() > 0}
        for agent in self.all_agent:
            agent.all_dident_see = tmp_all_dident_see
        x = 1

    def one_step(self):
        self.step_index += 1
        for one_agent in self.all_agent:
            if not one_agent.is_finis:
                one_agent_see = one_agent.move()
                self.all_seen = self.all_seen | one_agent_see

    def any_one_see_cell(self,agent):
        for unseen_cell in self.all_seen & {cell for cell in agent.didet_see}:
            if unseen_cell in agent.didet_see:
                if agent.fixing and agent.didet_see not in agent.path[:agent.step_index]:
                    agent.return_to_path(agent.location, agent.path[agent.fixing_step_index])
                agent.remove_from_dident_see(unseen_cell)
                print(f"agent {agent.id} remove by see: {unseen_cell}  locate at {agent.location}")

    def exaxute_path(self, need_to_see=True):
        while not min([agent.is_finis for agent in self.all_agent]):
            self.one_step()

            if self.fixing_metod == 0:
                for one_agent in self.all_agent:
                    self.fix_mangment_solo_go_to_close(one_agent)
            elif self.fixing_metod == 1:
                for one_agent in self.all_agent:
                    self.fix_mangment_multy_go_to_close(one_agent)
            elif self.fixing_metod == 2:
                for one_agent in self.all_agent:
                    self.fix_mangment_solo_minimal_path(one_agent)

            self.update_dident_see()

            if need_to_see:
                Utils.print_exexute(self.world, self.all_agent)
                print([i.location for i in self.all_agent])

            sleep(0.1)
        if self.world.free_cell != self.all_seen.__len__():
            print(f"teory :{self.world.free_cell} \t realty :{self.all_seen.__len__()}")
        return True


if __name__ == '__main__':
    map_type = 'maze_11_11'
    name = 'test'

    experement_name = f'{map_type}_{name}'

    # all_free = np.transpose(np.where(np.array(row_map) == 0))

    minimize = {'mksp': 0, 'soc': 1}
    number_of_agent = 3
    huristics = 3

    exp_index = 3
    start_pos = [ast.literal_eval(i) for i in np.loadtxt(f'./config/{map_type}_{number_of_agent}_agent_domain.csv',
                                                         dtype=tuple, delimiter='\n')]

    data_file = open(f'{experement_name}_{number_of_agent}_agent_{huristics}_huristic.csv', 'w',
                     newline='\n')
    writer = csv.writer(data_file, delimiter=',')
    writer.writerow(
        ['map_name', 'start_state', 'time', 'h type', 'h_start', 'h_genarate', 'h_expend', 'number of max pivot',
         'use black list', 'genarate', 'expend', 'open is better', 'new is beter', 'obs remove', 'cost'])

    while True:
        world = WorldMap(map_type)
        sp = start_pos[random.randint(0, 99)]
        mwrp = Mwrp(world, sp, huristics, minimize['mksp'])
        all_path = mwrp.run(writer, map_type, start_pos, need_path=True)
        if all_path is not None:
            execute = ExecuteTheMwrp(world, all_path, 2, minimize['mksp'])
            execute.exaxute_path()
        else:
            print("no path fond")
        input()

    # TODO
    # GO see and go back for shorter track
    # use cplex for geting path
    # solve wrp from start
