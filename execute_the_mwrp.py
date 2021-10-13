import numpy as np
from script.utils import Utils
from script.world import WorldMap
from mwrp import Mwrp
import ast
import pprint
from time import  sleep
import csv
import random
from alive_progress import alive_bar

class ExecuteTheMwrp:

    def __init__(self, world: WorldMap, path: dict ,minimize: int) -> None:
        """
        :param world:           An WorldMap  object from script.world that contains the parameters of the world
        :param path:            all robot path sort as dict {0:(x1,y1),(x2,y2)... 1:(x1,y1),(x2,y2)... }
        :param minimize:        What we want to minimize (soc, mksp)
        """
        self.minimize = minimize
        self.path = path
        self.world = world
        self.step_index=-1
        self.agents_location = {index : self.path[index][0] for index in self.path}
        self.agents_seen = {index : set() for index in self.path}
        self.all_seen = set()
        self.didet_see= set()

    def field_of_view_validation(self,cell):
        field_of_view=self.world.dict_fov[cell]
        for i in field_of_view:
            if self.didet_see.__len__() < 1:
                if random.randint(0,0) == 0 and i not in self.all_seen:
                    self.didet_see.add(i)
                    print(self.didet_see)
                    field_of_view=field_of_view - {i}
            else:
                if i in self.didet_see:
                    if random.randint(0,99) != 0:
                        field_of_view = field_of_view - {i}
        return field_of_view

    def fix_unseen(self):
        for i in self.agents_location:
            if self.agents_location[i] in self.didet_see:
                self.didet_see=self.didet_see-{self.agents_location[i]}

    def move(self):
        for i in range(self.agents_location.__len__()):
            if self.step_index < self.path[i].__len__():
                self.agents_location[i] = self.path[i][self.step_index]

    def see(self):
        for i in range(self.agents_location.__len__()):
            tmp_seen = self.field_of_view_validation(self.agents_location[i])
            self.agents_seen[i] = self.agents_seen[i] | tmp_seen | {self.agents_location[i]}
            self.all_seen = self.all_seen | self.agents_seen[i]

    def one_step(self,need_to_see=True):
        self.step_index+=1
        self.move()
        self.fix_unseen()
        self.see()

        if need_to_see:
            Utils.print_exexute(self.world,self.path,self.step_index,self.agents_seen)

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
    start_pos =  [ast.literal_eval(i) for i in np.loadtxt(f'./config/{map_type}_{number_of_agent}_agent_domain.csv',
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
    execute=ExecuteTheMwrp(world,all_path,minimize['mksp'])
    while execute.step_index+1<max([execute.path[i].__len__() for i in execute.path]):
        execute.one_step()
        sleep(0.1)
    execute.one_step()

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
