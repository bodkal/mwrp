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
    def __init__(self,agent_id,start_location,path,world):
        self.id=agent_id
        self.path=path
        self.location=start_location
        self.seen=set()
        self.didet_see=[]
        self.step_index=-1
        self.world=world

    def move(self):
        self.step_index += 1
        if self.step_index < self.path.__len__():
            self.location = self.path[self.step_index]

    def see(self,all_seen):
        self.seen = self.seen | {self.location} | self.fov_validation(all_seen)
        return self.seen

    def fov_validation(self,all_seen):
        fov=self.world.dict_fov[self.location]
        remove_from_fov=set()
        for fov_cell in fov:
            if fov_cell not in all_seen:
                if self.didet_see.__len__() < 1 :
                    # dident see a cell that need to see
                    if random.randint(0, 10) == 0 :
                        self.didet_see.append(fov_cell)
                        print(f"agent {self.id} new need to see : {fov_cell}")
                        remove_from_fov.add(fov_cell)
                else:
                    # see a cell that dident see in the past
                    if fov_cell in self.didet_see:
                        if random.randint(0, 50) == 0:
                            print(f"remove from need to see by miracle: {fov_cell}")
                            self.didet_see.remove(fov_cell)
                        else:
                            remove_from_fov.add(fov_cell)

        return fov-remove_from_fov

    def see_unseen_by_stend(self):
        if self.location in self.didet_see:
            self.didet_see.remove(self.location)
            print(f"agent {self.id} remove from need to see by stend: {self.location}")

    def fix_unseen_strate(self):
        for i in self.didet_see:
            tmp_path = self.world.BFS.get_path(self.path[self.didet_see[i]][self.step_index], self.didet_see[i])

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
        self.all_agent=[agent(agent_index,path[agent_index][0],path[agent_index],world) for agent_index in range(path.__len__())]

    def one_step(self, need_to_see=True):
        self.step_index += 1
        for one_agent in self.all_agent:
            one_agent.move()
            self.all_seen= self.all_seen | one_agent.see(self.all_seen)
            one_agent.see_unseen_by_stend()

        if need_to_see:
            Utils.print_exexute(self.world,self.all_agent)
        # if self.didet_see.__len__() > 0:
        #     return False
        return True

    def exaxute_path(self):
        step_is_valid=True
        #while step_is_valid:
        for i in range(20):
            step_is_valid = self.one_step()
            sleep(0.1)
        self.one_step()
        return True

    def run(self):
        while not self.exaxute_path():
            self.fix_unseen_strate()


if __name__ == '__main__':
    map_type = 'maze_11_11'
    name = 'test'

    experement_name = f'{map_type}_{name}'
    map_config = f'./config/{map_type}_config.csv'

    row_map = np.array(Utils.convert_map(map_config))

    all_free = np.transpose(np.where(np.array(row_map) == 0))

    minimize = {'mksp': 0, 'soc': 1}
    number_of_agent = 3
    huristics = 3

    exp_index = 6
    start_pos = [ast.literal_eval(i) for i in np.loadtxt(f'./config/{map_type}_{number_of_agent}_agent_domain.csv',
                                                         dtype=tuple, delimiter='\n')]

    data_file = open(f'{experement_name}_{number_of_agent}_agent_{huristics}_huristic.csv', 'w',
                     newline='\n')
    writer = csv.writer(data_file, delimiter=',')
    writer.writerow(
        ['map_name', 'start_state', 'time', 'h type', 'h_start', 'h_genarate', 'h_expend', 'number of max pivot',
         'use black list', 'genarate', 'expend', 'open is beter', 'new is beter', 'obs remove', 'cost'])

    world = WorldMap(row_map)
    mwrp = Mwrp(world, start_pos[exp_index], huristics, map_type, minimize['mksp'])
    all_path = mwrp.run(writer, map_config, start_pos, need_path=True)
    execute = ExecuteTheMwrp(world, all_path, minimize['mksp'])
    execute.run()

    sleep(100)
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
