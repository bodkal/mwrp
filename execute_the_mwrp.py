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
        self.path = path
        self.location = start_location
        self.seen = set()
        self.didet_see = {}
        self.step_index = -1
        self.fixing_step_index=0
        self.is_finis=False
        self.fixing=False
        self.world = world
        self.fixing_path=[]
        self.path_for_plot=[]

    def move(self,all_see):
        if not self.fixing:
            self.step_index += 1
            if self.location == self.path[-1]:
                self.is_finis=True
            else:
                self.location = self.path[self.step_index]
        else:
            print(f" {self.id} is fixing")
            self.fixing_step_index += 1
            self.location = self.fixing_path[self.fixing_step_index]
            if self.location == self.fixing_path[-1]:
                self.fixing = False
        self.path_for_plot.append(self.location)
        return self.see(all_see)

    def see(self, all_seen):
        self.seen = self.seen | {self.location} | self.fov_validation(all_seen)
        need_to_fix = False if self.didet_see.__len__() == 0 else True
        return self.seen, need_to_fix

    def validation_didet_see_dict(self,all_seen):
        tmp=list(self.didet_see.keys())
        for cell in tmp:
            if all_seen & {cell} != set():
                del self.didet_see[cell]
                #del self.add_to_path[cell]

                print(f"agent {self.id} not longer need to see : {cell}")

    def fov_validation(self, all_seen):
        fov = self.world.dict_fov[self.location]
        remove_from_fov = set()
        self.validation_didet_see_dict(all_seen)

        for fov_cell in fov:
            if fov_cell not in all_seen:
                if self.didet_see.__len__() < 1 and fov_cell not in self.path:
                    # dident see a cell that need to see
                    if random.randint(0, 10) == 0 and not self.fixing:
                        self.didet_see[fov_cell]=self.cell_to_fix_from_close(fov_cell)
                        remove_from_fov.add(fov_cell)

                        print(f"agent {self.id} new need to see : {fov_cell}")
                else:
                    # see a cell that dident see in the past
                    if fov_cell in self.didet_see:
                        if random.randint(0, 5000000) == 0:
                            print(f"remove from need to see by miracle: {fov_cell}")
                            del self.didet_see[fov_cell]
                           # del self.add_to_path[fov_cell]

                        else:
                            remove_from_fov.add(fov_cell)
        seen_fov = fov - remove_from_fov
        return seen_fov

    def see_unseen_by_stand(self):
        if self.location in self.didet_see:
            del self.didet_see[self.location]

            print(f"agent {self.id} remove from need to see by stend: {self.location}")
            return True
        return False


    def cell_to_fix_from_close(self,didet_see):
        all_dis={didet_see : [] }
        for path_cell in self.path[self.step_index:]:
            all_dis[didet_see].append(self.world.real_dis_dic[tuple(sorted((path_cell,didet_see)))])

        if self.path[self.step_index:].__len__()!=0:
            insert_index=self.step_index+int(all_dis[didet_see].index(min(all_dis[didet_see])))
            min_dis=self.path[insert_index]
        else:
            min_dis=self.path[-1]
        return min_dis


class ExecuteTheMwrp:

    def __init__(self, world: WorldMap, path: dict, minimize: int) -> None:
        """
        :param world:           An WorldMap  object from script.world that contains the parameters of the world
        :param path:            all robot path sort as dict {0:(x1,y1),(x2,y2)... 1:(x1,y1),(x2,y2)... }
        :param minimize:        What we want to minimize (soc, mksp)
        """
        self.minimize = minimize
        self.world = world
        self.step_index = -1
        self.all_seen = set()
        self.all_agent = [agent(agent_index, path[agent_index][0], path[agent_index], world) for agent_index in
                          range(path.__len__())]
        self.fix_path_asiment = {}

    def fix_mangment(self,didet_see,agent):
        if didet_see:

                if agent.see_unseen_by_stand():
                       return

                if agent.location in agent.didet_see.values():
                    agent.fixing=True
                    key=Utils.get_key_from_value(agent.didet_see, agent.location)
                    path_for_fix=self.world.BFS.get_path(agent.location,key)
                    agent.fixing_path = path_for_fix[:-1]+path_for_fix[::-1]+[agent.location]
                    agent.fixing_step_index=-1



    def one_step(self, need_to_see=True):
        self.step_index += 1
        for one_agent in self.all_agent:
            if not one_agent.is_finis:
                one_agent_see, didet_see = one_agent.move(self.all_seen)
                self.all_seen = self.all_seen | one_agent_see
                self.fix_mangment(didet_see, one_agent)

        if need_to_see:
            Utils.print_exexute(self.world, self.all_agent)
        # if self.didet_see.__len__() > 0:
        #     return False
        return True

    def exaxute_path(self):
        step_is_valid = True
        # while step_is_valid:
        while not min([agent.is_finis for agent in self.all_agent]):
            step_is_valid = self.one_step()
            sleep(0.1)
        return True



if __name__ == '__main__':
    map_type = 'maze_11_11'
    name = 'test'

    experement_name = f'{map_type}_{name}'


    #all_free = np.transpose(np.where(np.array(row_map) == 0))

    minimize = {'mksp': 0, 'soc': 1}
    number_of_agent = 2
    huristics = 3

    exp_index = 3
    start_pos = [ast.literal_eval(i) for i in np.loadtxt(f'./config/{map_type}_{number_of_agent}_agent_domain.csv',
                                                         dtype=tuple, delimiter='\n')]

    data_file = open(f'{experement_name}_{number_of_agent}_agent_{huristics}_huristic.csv', 'w',
                     newline='\n')
    writer = csv.writer(data_file, delimiter=',')
    writer.writerow(
        ['map_name', 'start_state', 'time', 'h type', 'h_start', 'h_genarate', 'h_expend', 'number of max pivot',
         'use black list', 'genarate', 'expend', 'open is beter', 'new is beter', 'obs remove', 'cost'])

    world = WorldMap(map_type)
    mwrp = Mwrp(world, start_pos[exp_index], huristics, minimize['mksp'])
    all_path = mwrp.run(writer, map_type, start_pos, need_path=True)
    execute = ExecuteTheMwrp(world, all_path, minimize['mksp'])
    execute.exaxute_path()
    input()

    # TODO
    # go to fix strat wane you see
    # go see from the shorter cell
    # GO see and go back for shorter track
    # use cplex for geting path
    # solve wrp from start

    # with alive_bar(exp_index+1) as bar:
    #     for i in range(exp_index):
    #
    #         world = WorldMap(row_map)
    #         mwrp = Mwrp(world, start_pos[i], huristics, map_type, minimize['mksp'])
    #         all_path=mwrp.run(writer, map_config, start_pos,need_path=True)
    #         bar()
    # pprint.pprint([all_path])
