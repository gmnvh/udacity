# ----------
# User Instructions:
# 
# Implement the function optimum_policy2D below.
#
# You are given a car in grid with initial state
# init. Your task is to compute and return the car's 
# optimal path to the position specified in goal; 
# the costs for each motion are as defined in cost.
#
# There are four motion directions: up, left, down, and right.
# Increasing the index in this array corresponds to making a
# a left turn, and decreasing the index corresponds to making a 
# right turn.

forward = [[-1,  0], # go up
           [ 0, -1], # go left
           [ 1,  0], # go down
           [ 0,  1]] # go right
forward_name = ['up', 'left', 'down', 'right']

# action has 3 values: right turn, no turn, left turn
action = [-1, 0, 1]
action_name = ['R', '#', 'L']

# EXAMPLE INPUTS:
# grid format:
#     0 = navigable space
#     1 = unnavigable space 
grid = [[1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 1, 1],
        [1, 1, 1, 0, 1, 1]]

init = [4, 3, 0] # given in the form [row,col,direction]
                 # direction = 0: up
                 #             1: left
                 #             2: down
                 #             3: right
                
goal = [2, 0] # given in the form [row,col]

cost = [2, 1, 20, 2] # cost has 3 values, corresponding to making 
                     # a right turn, no turn, and a left turn

# EXAMPLE OUTPUT:
# calling optimum_policy2D with the given parameters should return 
# [[' ', ' ', ' ', 'R', '#', 'R'],
#  [' ', ' ', ' ', '#', ' ', '#'],
#  ['*', '#', '#', '#', '#', 'R'],
#  [' ', ' ', ' ', '#', ' ', ' '],
#  [' ', ' ', ' ', '#', ' ', ' ']]
# ----------
import numpy as np
# ----------------------------------------
# modify code below
# ----------------------------------------
def compute_value(grid,goal,cost):
    # ----------------------------------------
    # insert code below
    # ----------------------------------------
    
    # make sure your function returns a grid of values as 
    # demonstrated in the previous video.
    value = [[[999 for col in range(len(grid[0]))] for row in range(len(grid))] for ori in range(len(forward))]
    policy = [[' ' for col in range(len(grid[0]))] for row in range(len(grid))]
     
    change = True
    while change:
        change = False
        
        for orientation in range(len(forward)):
            for x in range(len(grid)):
                for y in range(len(grid[0])):
                    if goal[0] == x and goal[1] == y:
                        if value[orientation][x][y] > 0:
                            value[orientation][x][y] = 0
                            policy[x][y] = '*'
                            change = True
                            
                    elif grid[x][y] == 0:
                        for a in range(len(forward)):
                            x2 = x + forward[a][0]
                            y2 = y + forward[a][1]
                            
                            if x2 >= 0 and x2 < len(grid) and y2 >=0 and y2 < len(grid[0]) \
                                and grid[x2][y2] == 0:
                                
                                step = (a - orientation)
                                if step in action:
                                    indx = action.index(step)
                                    v2 = value[orientation][x2][y2] + cost[indx]
                                    print step, indx, v2
                            
                                    if v2 < value[orientation][x][y]:
                                        change = True
                                        value[orientation][x][y] = v2
                                        policy[x][y] = forward_name[a]
                                else:
                                    print 'step error'
                                    print step
                
                    print x, y
                    print np.array(value)
                    a = raw_input()
    return policy                         
	
def optimum_policy2D(grid,init,goal,cost):

    return policy2D


pol = compute_value(grid, goal, cost)
print np.array(pol)