import numpy as np
from itertools import product
from script.utils import Vision, Utils
from script.BFS import BFS


class WorldMap:

    def __init__(self, row_map: list) -> None:
        """
        hold the world object
        :param row_map: grid map 1 obtical 0 free
        """

        # world parmeter
        self.grid_map = row_map.astype(np.int8)
        self.vision = Vision(self.grid_map)
        [self.col_max, self.row_max] = self.grid_map.shape
        self.free_cell = len((np.where(self.grid_map == 0)[0]))
        self.dict_watchers = dict()
        self.dict_fov = dict()
        self.watchers_frontier = dict()
        self.create_watchers()
        self.action = self.get_static_action_5(1)
        # BFS metod for expending border
        self.BFS = BFS(self)

    def is_obstical(self, state: tuple) -> bool:
        """
        find if cell is obstical
        :param state: list of cells
        :return: True if free (0) False if obstical (1)
        """

        for cell in state:
            if self.grid_map[cell] == 1:
                return False
        return True

    def get_static_action_9(self, agents_number: int) -> list:
        """
        get all action combination for all agent
        :param agents_number: number of robot
        :return: list of action for all agent
        """
        return np.reshape(list(product([1, 0, -1], repeat=agents_number * 2)), (-1, agents_number * 2)).astype(int)

    def get_static_action_8(self, agents_number: int) -> list:
        """
        get all action combination for all agent
        :param agents_number: number of robot
        :return: list of action for all agent
        """
        all_opsens = np.reshape(list(product([1, 0, -1], repeat=agents_number * 2)), (-1, agents_number * 2))
        all_index = np.arange(9 ** agents_number)
        for i in range(agents_number):
            tmp_index = np.where(np.any(all_opsens[:, i * 2: i * 2 + 2], axis=1))
            all_index = np.intersect1d(tmp_index, all_index)
        return np.copy(all_opsens[all_index]).astype(int)

    def get_static_action_5(self, agents_number: int) -> list:
        """
        get all action combination for all agent
        :param agents_number: number of robot
        :return: list of action for all agent
        """
        all_opsens = np.reshape(list(product([1, 0, -1], repeat=agents_number * 2)), (-1, agents_number * 2))
        all_index = np.arange(9 ** agents_number)
        for i in range(agents_number):
            tmp_index = np.where(np.any(all_opsens[:, i * 2: i * 2 + 2] == 0, axis=1))
            all_index = np.intersect1d(tmp_index, all_index)
        return np.copy(all_opsens[all_index]).astype(int)

    def get_static_action_4(self, agents_number: int) -> list:
        """
        get all action combination for all agent
        :param agents_number: number of robot
        :return: list of action for all agent
        """
        all_opsens = np.reshape(list(product([1, 0, -1], repeat=agents_number * 2)), (-1, agents_number * 2))
        all_index = np.arange(9 ** agents_number)
        for i in range(agents_number):
            tmp_index = np.where(np.abs(all_opsens[:, i * 2]) != np.abs(all_opsens[:, i * 2 + 1]))
            all_index = np.intersect1d(tmp_index, all_index)
        return np.copy(all_opsens[all_index]).astype(int)

    def remove_obstical(self, number_of_obstical_to_remove: int) -> list:
        """
        remove random obstical from a grid map
        :param number_of_obstical_to_remove: number of obstical to remove
        :return: new grid map withe less obstecal
        """
        import random
        all_obstical = [i for i in np.transpose(np.where(np.array(self.grid_map) == 1))
                        if not np.any(i == 0) and i[0] != self.col_max - 1 and i[1] != self.row_max - 1]

        while number_of_obstical_to_remove > 0 and all_obstical.__len__() > 0:
            random_obstical = all_obstical.pop(random.randrange(len(all_obstical)))
            actihon = self.get_static_action_4(1) + random_obstical
            obs_number_row = 0
            obs_number_col = 0
            # rolls for remove obstecal that cant remove cell that i can see true it but not walk
            for index, cell in enumerate(actihon):
                if np.any(cell == 0) or cell[0] == self.col_max - 1 or cell[1] == self.row_max - 1:
                    continue
                elif self.grid_map[tuple(cell)] == 1 and index in [0, 3]:
                    obs_number_row += 1
                elif self.grid_map[tuple(cell)] == 1 and index in [1, 2]:
                    obs_number_col += 1
            if obs_number_row + obs_number_col < 3 and (obs_number_row == 0 or obs_number_col == 0):
                self.grid_map[tuple(random_obstical)] = 0
                number_of_obstical_to_remove -= 1

        return self.grid_map

    def get_free_cell(self, cell: tuple) -> set:
        """
        get all free cells around specific cell
        :param cell: specific cell
        :return: list of free cells
        """

        free_cells = {tuple((tmp[0] + cell[0], tmp[1] + cell[1])) for tmp in self.get_static_action_4(1) if
                      self.grid_map[tmp[0] + cell[0], tmp[1] + cell[1]] == 0}

        return free_cells

    # TODO add frontire
    def create_watchers(self) -> None:
        """
        creat all watchers for all cells
        """
        all_free_cell = set(map(tuple, np.asarray(np.where(self.grid_map == 0)).T))
        for cell in all_free_cell:
            # get fov (i see =/= see me)
            self.dict_fov[cell] = self.vision.get_fov(cell)

            # run an all fov and create the wochers from it
            for wahers in self.dict_fov[cell]:
                if wahers != cell:
                    if wahers in self.dict_watchers:
                        self.dict_watchers[wahers] = self.dict_watchers[wahers].union(Utils.map_to_sets(cell))
                    else:
                        self.dict_watchers[wahers] = Utils.map_to_sets(cell) | {wahers}

        # find all watchers frontier
        self.watchers_frontier = {cell: set() for cell in self.dict_watchers}
        for cell in self.dict_watchers.keys():
            for watchers in self.dict_watchers[cell]:
                if self.get_free_cell(watchers) - self.dict_watchers[cell] - {cell} != set():
                    self.watchers_frontier[cell].add(watchers)

    def get_all_seen(self, state: tuple) -> set:
        """
        get all fov from all agent
        :param state: all agent location
        :return: set off all seen cell
        """
        tmp_set_seen = set()
        for cell in state:
            tmp_set_seen = tmp_set_seen.union(self.dict_fov[cell])
        return tmp_set_seen

    def is_valid_node(self, new_state: object, old_state: object, moving_status: list) -> bool:
        """
        find if cell is valid need only whit no eb
        :param new_state:
        :param old_state:
        :param moving_status:
        :return:
        """
        if not self.is_obstical(new_state):
            return False
        elif new_state == old_state.parent.location:
            return False
        # if acthon is all dead
        elif not np.any(moving_status):
            return False

        # if dead agent moves
        for i in old_state.dead_agent:
            if moving_status[i] != 0:
                return False
        return True
