from helpers import Map, load_map_10, load_map_40, show_map
import math

map_10 = load_map_10()
map_40 = load_map_40()

class PathPlanner():

    """Construct a PathPlanner Object"""
    def __init__(self, M, start=None, goal=None):
        """ """
        self.map = M
        self.start= start
        self.goal = goal
        self.closedSet = self.create_closedSet() if goal != None and start != None else None
        self.openSet = self.create_openSet() if goal != None and start != None else None
        self.cameFrom = self.create_cameFrom() if goal != None and start != None else None
        self.gScore = self.create_gScore() if goal != None and start != None else None
        self.fScore = self.create_fScore() if goal != None and start != None else None
        self.path = self.run_search() if self.map and self.start != None and self.goal != None else None

    def reconstruct_path(self, current):
            """ Reconstructs path after search """
            total_path = [current]
            while current in self.cameFrom.keys():
                current = self.cameFrom[current]
                total_path.append(current)
            return total_path

    def _reset(self):
        """Private method used to reset the closedSet, openSet, cameFrom, gScore, fScore, and path attributes"""
        self.closedSet = None
        self.openSet = None
        self.cameFrom = None
        self.gScore = None
        self.fScore = None
        self.path = self.run_search() if self.map and self.start and self.goal else None

    def run_search(self):
        """ """
        if self.map == None:
            raise(ValueError, "Must create map before running search. Try running PathPlanner.set_map(start_node)")
        if self.goal == None:
            raise(ValueError, "Must create goal node before running search. Try running PathPlanner.set_goal(start_node)")
        if self.start == None:
            raise(ValueError, "Must create start node before running search. Try running PathPlanner.set_start(start_node)")

        self.closedSet = self.closedSet if self.closedSet != None else self.create_closedSet()
        self.openSet = self.openSet if self.openSet != None else  self.create_openSet()
        self.cameFrom = self.cameFrom if self.cameFrom != None else  self.create_cameFrom()
        self.gScore = self.gScore if self.gScore != None else  self.create_gScore()
        self.fScore = self.fScore if self.fScore != None else  self.create_fScore()

        while not self.is_open_empty():
            current = self.get_current_node()

            if current == self.goal:
                self.path = [x for x in reversed(self.reconstruct_path(current))]
                return self.path
            else:
                self.openSet.remove(current)
                self.closedSet.add(current)

            for neighbor in self.get_neighbors(current):
                if neighbor in self.closedSet:
                    continue    # Ignore the neighbor which is already evaluated.

                if not neighbor in self.openSet:    # Discover a new node
                    self.openSet.add(neighbor)
                
                # The distance from start to a neighbor
                #the "dist_between" function may vary as per the solution requirements.
                if self.get_tentative_gScore(current, neighbor) >= self.get_gScore(neighbor):
                    continue        # This is not a better path.

                # This path is the best until now. Record it!
                self.record_best_path_to(current, neighbor)
        print("No Path Found")
        self.path = None
        return False

    def create_closedSet(self):
        """ Creates and returns a data structure suitable to hold the set of nodes already evaluated"""
        # EXAMPLE: return a data structure suitable to hold the set of nodes already evaluated
        return set()

    def create_cameFrom(self):
        """Creates and returns a data structure that shows which node can most efficiently be reached from another,
        for each node."""
        # TODO: return a data structure that shows which node can most efficiently be reached from another,
        # for each node. 
        return {}

    def create_openSet(self):
        """ Creates and returns a data structure suitable to hold the set of currently discovered nodes 
        that are not evaluated yet. Initially, only the start node is known."""
        if self.start != None:
            # TODO: return a data structure suitable to hold the set of currently discovered nodes 
            # that are not evaluated yet. Make sure to include the start node.
            openSet = set([self.start])
            return openSet
    
        raise(ValueError, "Must create start node before creating an open set. Try running PathPlanner.set_start(start_node)")

    def create_gScore(self):
        """Creates and returns a data structure that holds the cost of getting from the start node to that node, 
        for each node. The cost of going from start to start is zero."""
        # TODO:  return a data structure that holds the cost of getting from the start node to that node, for each node.
        # for each node. The cost of going from start to start is zero. The rest of the node's values should 
        # be set to infinity.
        gScore = {}
        for intersection in self.map.intersections:
            if intersection == self.start:
                gScore[intersection] = 0
            else:
                gScore[intersection] = math.inf
    
        return gScore

    def create_fScore(self):
        """Creates and returns a data structure that holds the total cost of getting from the start node to the goal
        by passing by that node, for each node. That value is partly known, partly heuristic.
        For the first node, that value is completely heuristic."""
        # TODO: return a data structure that holds the total cost of getting from the start node to the goal
        # by passing by that node, for each node. That value is partly known, partly heuristic.
        # For the first node, that value is completely heuristic. The rest of the node's value should be 
        # set to infinity.
        fScore = {}
        for intersection in self.map.intersections:
            if intersection == self.start:
                fScore[intersection] = self.distance(self.start, self.goal)
            else:
                fScore[intersection] = math.inf
        return fScore

    def set_map(self, M):
        """Method used to set map attribute """
        self._reset(self)
        self.start = None
        self.goal = None
        # TODO: Set map to new value. 
        self.M = M

    def set_start(self, start):
        """Method used to set start attribute """
        self._reset(self)
        # TODO: Set start value. Remember to remove goal, closedSet, openSet, cameFrom, gScore, fScore, 
        # and path attributes' values.
        self.start = start
        self.goal = None
        self.closedSet = None
        self.openSet = None
        self.cameFrom = None
        self.gScore = None
        self.fScore = None
        self.path = None
        

    def set_goal(self, goal):
        """Method used to set goal attribute """
        self._reset(self)
        # TODO: Set goal value.
        self.goal = goal
        self.closedSet = self.create_closedSet()
        self.openSet = self.create_openSet()
        self.cameFrom = self.create_cameFrom()
        self.gScore = self.create_gScore()
        self.fScore = self.create_fScore()
        self.path = self.run_search()

    def is_open_empty(self):
        """returns True if the open set is empty. False otherwise. """
        # TODO: Return True if the open set is empty. False otherwise.
        return (len(self.openSet) == 0)

    def get_current_node(self):
        """ Returns the node in the open set with the lowest value of f(node)."""
        # TODO: Return the node in the open set with the lowest value of f(node).
        rsp = None
        for node in self.openSet:
            if rsp is None or self.fScore[node] < self.fScore[rsp]:
                rsp = node
        return rsp

    def get_neighbors(self, node):
        """Returns the neighbors of a node"""
        # TODO: Return the neighbors of a node
        return self.map.roads[node]

    def get_gScore(self, node):
        """Returns the g Score of a node"""
        # TODO: Return the g Score of a node
        return self.gScore[node]

    def distance(self, node_1, node_2):
        """ Computes the Euclidean L2 Distance"""
        # TODO: Compute and return the Euclidean L2 Distance
        x1 = self.map.intersections[node_1][0]
        y1 = self.map.intersections[node_1][1]
        x2 = self.map.intersections[node_2][0]
        y2 = self.map.intersections[node_2][1]
        return math.sqrt(((x2 - x1)**2) + ((y2 - y1)**2))

    def get_tentative_gScore(self, current, neighbor):
        """Returns the tentative g Score of a node"""
        # TODO: Return the g Score of the current node 
        # plus distance from the current node to it's neighbors
        return self.gScore[current] + self.distance(current, neighbor)

    def heuristic_cost_estimate(self, node):
        """ Returns the heuristic cost estimate of a node """
        # TODO: Return the heuristic cost estimate of a node
        return self.distance(node, self.goal)

    def calculate_fscore(self, node):
        """Calculate the f score of a node. """
        # TODO: Calculate and returns the f score of a node. 
        # REMEMBER F = G + H
        return self.gScore[node] + self.heuristic_cost_estimate(node)

    def record_best_path_to(self, current, neighbor):
        """Record the best path to a node """
        # TODO: Record the best path to a node, by updating cameFrom, gScore, and fScore
        self.cameFrom[neighbor] = current
        self.gScore[neighbor] = self.gScore[current] + self.distance(current, neighbor)
        self.fScore[neighbor] = self.gScore[neighbor] + self.heuristic_cost_estimate(neighbor)

planner = PathPlanner(map_40, 5, 34)
path = planner.path
if path == [5, 16, 37, 12, 34]:
    print("great! Your code works for these inputs!")
else:
    print("something is off, your code produced the following:")
    print(path)

# Visualize your the result of the above test! You can also change start and goal here to check other paths
start = 8
goal = 24

show_map(map_40, start=start, goal=goal, path=PathPlanner(map_40, start, goal).path)