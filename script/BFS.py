class Node:
    def __init__(self, location: tuple, cost: int, parent: object) -> None:
        self.parent = parent
        self.location = location
        self.cost = cost

    def __str__(self):
        return f'location = {self.location} \t cost = {self.cost}'

    def __lt__(self, other: object) -> bool:
        """
        compere between 2 nodes for the binary serach
        :param other: node
        :return: True if need to insert on higher in open False otherwise
        """
        if self.cost > other.cost:
            return True
        else:
            return False


class BFS:

    def __init__(self, world: object) -> None:
        self.start = 0
        self.unseen = 0
        self.open_list = []
        self.world = world
        self.action = [(1, 0), (0, 1), (-1, 0), (0, -1)]

        # open and close directory
        self.visit_node = {}
        # need to find all frontier
        self.frontier = {}

    def _get_index_binary_search(self, new_node: object) -> int:
        """
        find the index in the open list using binary search bast on node cost
        :param new_node: the node we want to find his index in the open list
        :return: the index for the new node in open list
        """
        index_a = 0
        index_b = len(self.open_list)
        while index_a < index_b:
            mid = (index_a + index_b) // 2
            data = self.open_list[mid]
            # __lt__ bast on cost
            if new_node.__lt__(data):
                index_b = mid
            else:
                index_a = mid + 1
        return index_a

    def _insert_to_open_list(self, new_node: object) -> None:
        """
        insert to open list and visibal dict, open sort bottom up (best in bottom)
        :param new_node: new node
        """
        self.visit_node[new_node.location] = new_node.parent
        self.open_list.insert(self._get_index_binary_search(new_node), new_node)

    def _get_all_action(self, old_node: object) -> list:
        """
        return all valid cell around parent for generate new nods
        :param old_node: parent node
        :return: list of valid cells
        """
        x=old_node.location[0]
        y=old_node.location[1]
        valid_state = [(s[0]+x, s[1]+y) for s in self.action if self.world.grid_map[s[0]+x,s[1]+y]==0]

        return valid_state

    def _pop_from_open_list(self) -> object:
        """
        if open lest is not empty return node from bottem else return 0 (finis search)
        :rtype: object
        """
        if self.open_list.__len__():
            return self.open_list.pop()
        else:
            return 0

    def _expend_frontier_search(self) -> None:
        """
        expend new node
        """
        old_node = self._pop_from_open_list()
        from time import time
        # run over all action and generate all children
        for action in self._get_all_action(old_node):
            if action not in self.visit_node:
                if self.world.dict_fov[action] & self.unseen == set():
                    # dident see any new cell
                    self._insert_to_open_list(Node(action, old_node.cost + 1, old_node))
                else:
                    # see new cell
                    self.frontier.append(action)

    def _expend_path_search(self,goal) -> object:
        """
        expend new node
        """
        old_node = self._pop_from_open_list()

        # goal test
        if old_node.location == goal:
            return old_node

        # run over all action and generate all children
        for action in self._get_all_action(old_node):
            if action not in self.visit_node.keys():
                self._insert_to_open_list(Node(action, old_node.cost + 1, old_node))
        return False

    def goal_test(self,old_node,goal):
        # goal test
        if old_node.location in goal:
            node = old_node
            path = []
            # get nodes from goal to root
            while node.parent != None:
                path.append(node.location)
                node = node.parent
            path.append(node.location)
            self.all_gols[(self.start, old_node.location)] = path
            goal.remove(old_node.location)

            if goal.__len__() == 0:
                return True
            else:
                return False


    def _expend_path_search_multipal_goal(self,goal) -> object:
        """
        expend new node
        """
        old_node = self._pop_from_open_list()
        if self.goal_test(old_node,goal):
            return True


        # run over all action and generate all children
        for action in self._get_all_action(old_node):
            if action not in self.visit_node.keys():
                self._insert_to_open_list(Node(action, old_node.cost + 1, old_node))


        return False

    def get_frontier(self, start: tuple, unseen: set) -> list:
        """
        get all frontier cell (see new cells)
        :param start: agent location
        :param unseen: cell that we dont see detrmenat is cell is frontier
        :return: list of all frontere [(x1,y1),(x2,y2),...]
        """
        self.unseen = unseen
        self.start = start
        self.open_list = [Node(self.start, 0, None)]

        self.frontier = [start]
        self.visit_node = {self.start: None}

        while self.open_list.__len__() > 0:
            self._expend_frontier_search()

        return self.frontier

    def get_path(self, start: tuple, goal: tuple) -> list:
        """
          get sorted path between 2 cell
          :param start: start cell
          :param goal: goal cell
          :return: list of all path [(x1,y1),(x2,y2),...]
          """
        node = False
        self.start = start
        self.open_list = [(Node(start, 0, None))]
        self.visit_node = {self.start: None}

        while node == False:
            node = self._expend_path_search(goal)

        all_path = []
        # get nodes from goal to root
        while node.parent != None:
            all_path.append(node.location)
            node = node.parent

        # return revers array because need path from root to goal
        return all_path[::-1]

    def get_all_paths(self, start: tuple, goal: set) -> list:
        """
          get sorted path between 2 cell
          :param start: start cell
          :param goal: all goal cell ((x1,y1),(x2,y2),....)
          :return: list of all path [(x1,y1),(x2,y2),...]
          """

        finis = False
        self.start = start
        self.open_list = [(Node(start, 0, None))]
        self.visit_node = {self.start: None}
        self.all_gols = {}

        while not finis:
            finis = self._expend_path_search_multipal_goal(goal)

        # return revers array because need path from root to goal
        return self.all_gols
