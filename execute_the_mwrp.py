import numpy as np
from script.utils import Utils
from script.world import WorldMap
from mwrp import Mwrp
import ast
import pprint

import csv
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
    x=1


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
