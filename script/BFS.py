class node:
    def __init__(self,location ,cost,parent):
        self.parent = parent
        self.location = location
        self.cost = cost

    def __print__(self):
        print(f'location = {self.location} \t cost = {self.cost}')

class BFS:

    def __init__(self, world):
        self.start=0
        self.unseen=0
        self.frontire={}
        self.open_list=[]
        self.world=world
        self.visit_node={}

    def insert_to_open_list(self,new_node):
        index_a = 0
        index_b = len(self.open_list)
        while index_a < index_b:
            mid = (index_a + index_b) // 2
            data = self.open_list[mid]
            if new_node.cost > data.cost:
                index_b = mid
            else:
                index_a = mid + 1

        self.visit_node[new_node.location]=new_node.parent
        self.open_list.insert(index_a, new_node)

    def get_all_action(self,old_node):
        action=[(1,0),(0,1),(-1,0),(0,-1)]
        valid_state=[]
        for state in action:
            tmp_state=(state[0]+old_node.location[0],state[1]+old_node.location[1])

            if self.world.grid_map[tmp_state[0]][tmp_state[1]] == 0:
                valid_state.append(tmp_state)
        return valid_state

    def pop_from_open_list(self):
        if len(self.open_list):
            return self.open_list.pop()
        else:
            return 0

    def expend(self):
        old_node=self.pop_from_open_list()
        all_action=self.get_all_action(old_node)

        for action in all_action:

            if action not in self.visit_node:

                if self.world.dict_fov[action].intersection(self.unseen) == set():
                    self.insert_to_open_list(node(action,old_node.cost+1,old_node))
                else:
                    self.frontire.append(action)


    def get_frontire(self,start,unseen):
        self.unseen=unseen
        self.start=start
        self.open_list=[node(self.start, 0, None)]
        self.frontire=[start]

        self.visit_node={self.start:None}

        while self.open_list.__len__()>0:
            self.expend()
        return self.frontire



