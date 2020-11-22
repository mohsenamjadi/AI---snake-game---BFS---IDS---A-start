

import random
import heapq
import math
import sys
from collections import defaultdict, deque, Counter
from itertools import combinations


class State:
    def __init__(self, snake_body_coordinates , foods_with_value : dict):
        #self.snake_body_coordinates = snake_body_coordinates
        #self.foods_with_value = foods_with_value
        self.__dict__.update(foods_with_value=foods_with_value, snake_body_coordinates=snake_body_coordinates)          


    def new(self, changes: dict, **kwds) -> 'State':
        state = State(snake_body_coordinates=self.snake_body_coordinates, foods_with_value=self.foods_with_value)
        state.update(self)
        state.update(changes)
        return state

        
    def get_snake_body_coordinates(self):
        return self.snake_body_coordinates


    def get_foods_with_value(self):
        return self.foods_with_value

    #def get_snake_head_coordinate(self):
        #head_of_snake = self.snake_body_coordinates ## not implemented
        #return head_of_snake 


class Game:
    def __init__(self, grid_width , grid_height , snake_body_coordinates , foods_with_value : dict ):
        self.snake_body_coordinates = snake_body_coordinates
        self.foods_with_value = foods_with_value
        self.initial = State( self.snake_body_coordinates, self.foods_with_value)
        self.grid = {(x, y) for x in range(grid_width) for y in range(grid_height)}


    directions = [ (0, -1), 
                    (-1, 0),           
                    (1,  0),
                    (0, +1) ]



    def select_from_direction(self, head_of_snake , snake_body_parts):
        (x, y) = head_of_snake
        can_move_to = {(x + dx, y + dy) for (dx, dy) in self.directions } - snake_body_parts
        return can_move_to
                
    
    def actions(self , state):
        snake_body_parts = state.get_snake_body_coordinates()
        head_of_snake = state.get_snake_head_coordinate()    
        
        moves = self.select_from_direction( head_of_snake , snake_body_parts)
        return moves


    def result(self , state , move):
        
        snake_body_update = state.get_snake_body_coordinates()
        snake_body_update.append(move)
        foods_value_update = state.get_foods_with_value() 

        for key in foods_value_update:
            if ( key == move ):
                foods_value_update[key] = foods_value_update[key] - 1
            else:
                snake_body_update.popleft()

        state = state.new( snake_body_coordinates = snake_body_update , foods_with_value = foods_value_update )

        return state
        


    def is_goal(self , state):
        goal_dict = state.get_foods_with_value()
        for key in goal_dict:
            while goal_dict[key] != 0:
                return False
        return True


    def action_cost(self, s, a, s1): return 1

    def h(self, node):               return 0

  

class Node:
    "A Node in a search tree."
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.__dict__.update(state=state, parent=parent, action=action, path_cost=path_cost)

    def __repr__(self): return '<{}>'.format(self.state)
    def __len__(self): return 0 if self.parent is None else (1 + len(self.parent))
    def __lt__(self, other): return self.path_cost < other.path_cost



failure = Node('failure', path_cost=math.inf) # Indicates an algorithm couldn't find a solution.
cutoff  = Node('cutoff',  path_cost=math.inf) # Indicates iterative deepening search was cut off.
      

def expand(game, node):
    "Expand a node, generating the children nodes."
    s = node.state
    for action in game.actions(s):
        s1 = game.result(s, action)
        cost = node.path_cost + game.action_cost(s, action, s1)
        yield Node(s1, node, action, cost)



def path_actions(node):
    "The sequence of actions to get to this node."
    if node.parent is None:
        return []  
    return path_actions(node.parent) + [node.action]


def path_states(node):
    "The sequence of states to get to this node."
    if node in (cutoff, failure, None): 
        return []
    return path_states(node.parent) + [node.state]


FIFOQueue = deque

LIFOQueue = list

class PriorityQueue:
    """A queue in which the item with minimum f(item) is always popped first."""

    def __init__(self, items=(), key=lambda x: x): 
        self.key = key
        self.items = [] # a heap of (score, item) pairs
        for item in items:
            self.add(item)
         
    def add(self, item):
        """Add item to the queuez."""
        pair = (self.key(item), item)
        heapq.heappush(self.items, pair)

    def pop(self):
        """Pop and return the item with min f(item) value."""
        return heapq.heappop(self.items)[1]
    
    def top(self): return self.items[0][1]

    def __len__(self): return len(self.items)



def breadth_first_search(game):
    "Search shallowest nodes in the search tree first."
    node = Node(game.initial)
    if game.is_goal(game.initial):
        return node
    frontier = FIFOQueue([node])
    reached = {game.initial}
    while frontier:
        node = frontier.pop()
        for child in expand(game, node):
            s = child.state
            if game.is_goal(s):
                return child
            if s not in reached:
                reached.add(s)
                frontier.appendleft(child)
    return failure



def iterative_deepening_search(game):
    "Do depth-limited search with increasing depth limits."
    for limit in range(1, sys.maxsize):
        result = depth_limited_search(game, limit)
        if result != cutoff:
            return result
        
        
def depth_limited_search(game, limit=10):
    "Search deepest nodes in the search tree first."
    frontier = LIFOQueue([Node(game.initial)])
    result = failure
    while frontier:
        node = frontier.pop()
        if game.is_goal(node.state):
            return node
        elif len(node) >= limit:
            result = cutoff
        elif not is_cycle(node):
            for child in expand(game, node):
                frontier.append(child)
    return result


def best_first_search(game, f):
    "Search nodes with minimum f(node) value first."
    node = Node(game.initial)
    frontier = PriorityQueue([node], key=f)
    reached = {game.initial: node}
    while frontier:
        node = frontier.pop()
        if game.is_goal(node.state):
            return node
        for child in expand(game, node):
            s = child.state
            if s not in reached or child.path_cost < reached[s].path_cost:
                reached[s] = child
                frontier.add(child)
    return failure


def g(n): return n.path_cost


def astar_search(game, h=None):
    """Search nodes with minimum f(n) = g(n) + h(n)."""
    h = h or game.h
    return best_first_search(game, f=lambda n: g(n) + h(n))


def weighted_astar_search(game, h=None, weight=1.8):
    """Search nodes with minimum f(n) = g(n) + weight * h(n)."""
    h = h or game.h
    return best_first_search(game, f=lambda n: g(n) + weight * h(n))

 
def main():


    with open("./tests/test1.txt") as file: 
        #Lines = file.readlines()
        Lines = file.read().splitlines()
        
    characterized_map = list()
    for line in Lines:
        characterized_map.append(list(line))

    x = int(Lines[0][0])
    y = int(Lines[0][2])

    snake_X_coordinate = int(Lines[1][0])
    snake_Y_coordinate = int(Lines[1][2])

    number_of_foods = int(Lines[2])

    foods_with_value = {(3, 1): 1, (3, 2): 1, (1, 4): 2, (4, 3): 1}


    mar_co = list()
    # Declaring deque  
    #mar_co = deque([])   
    mar_co.append((snake_X_coordinate,snake_Y_coordinate))
    #print(mar_co)
    


    game = Game(x , y , mar_co , foods_with_value )

    breadth_first_search(game)


main()