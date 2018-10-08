# -----------
# User Instructions:
#
# Modify the the search function so that it returns
# a shortest path as follows:
# 
# [['>', 'v', ' ', ' ', ' ', ' '],
#  [' ', '>', '>', '>', '>', 'v'],
#  [' ', ' ', ' ', ' ', ' ', 'v'],
#  [' ', ' ', ' ', ' ', ' ', 'v'],
#  [' ', ' ', ' ', ' ', ' ', '*']]
#
# Where '>', '<', '^', and 'v' refer to right, left, 
# up, and down motions. Note that the 'v' should be 
# lowercase. '*' should mark the goal cell.
#
# You may assume that all test cases for this function
# will have a path from init to goal.
# ----------

grid = [[0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0]]
init = [0, 0]
goal = [len(grid)-1, len(grid[0])-1]
cost = 1

delta = [[-1, 0 ], # go up
         [ 0, -1], # go left
         [ 1, 0 ], # go down
         [ 0, 1 ]] # go right

delta_name = ['^', '<', 'v', '>']

# Initialize expand table
expand = [[-1 for x in range(len(grid[0]))] for y in range(len(grid))]

def find_next(p):
    found = False
    new_paths = []
    for pin in range(len(p)):
        x = p[pin][-1][0]
        y = p[pin][-1][1]
        for move, symbol in zip(delta, delta_name):
            new_p = p[pin][:]
            x2 = x + move[1]
            y2 = y + move[0]
            if (x2 >=0  and x2 < len(grid[0])):
                if (y2 >= 0) and (y2 < len(grid)):
                    if expand[y2][x2] > expand[y][x]:
                        found = True
                        new_p.append([x2, y2, symbol])
                        new_paths.append(new_p)
                        #myprint(x2, y2, new_p)        
    if found is False:
        return p
    else:
        new_paths = find_next(new_paths)
        return new_paths
        
def search(grid,init,goal,cost):
    # ----------------------------------------
    # modify code below
    # ----------------------------------------
    closed = [[0 for row in range(len(grid[0]))] for col in range(len(grid))]
    closed[init[0]][init[1]] = 1

    x = init[0]
    y = init[1]
    g = 0
    exp_counter = 0
    expand[init[0]][init[1]] = exp_counter
    paths = [[[0,0]]]

    open = [[g, x, y]]

    found = False  # flag that is set when search is complete
    resign = False # flag set if we can't find expand

    while not found and not resign:
        if len(open) == 0:
            resign = True
            return 'fail'
        else:
            open.sort()
            open.reverse()
            next = open.pop()
            x = next[1]
            y = next[2]
            g = next[0]
            
            if x == goal[0] and y == goal[1]:
                found = True
            else:
                for i in range(len(delta)):
                    x2 = x + delta[i][0]
                    y2 = y + delta[i][1]
                    if x2 >= 0 and x2 < len(grid) and y2 >=0 and y2 < len(grid[0]):
                        if closed[x2][y2] == 0 and grid[x2][y2] == 0:
                            g2 = g + cost
                            open.append([g2, x2, y2])
                            closed[x2][y2] = 1
                            exp_counter += 1
                            expand[x2][y2] = exp_counter
    path = find_next(paths)
    for p in path:
        if len(p) == g2 + 1:
            found_path = p
            break
    n_grid = [[' ' for i in y] for y in grid] 
    for i in range(len(found_path)):
        x = found_path[i][0]
        y = found_path[i][1]
        if x == goal[1] and y == goal[0]:
            n_grid[y][x] = '*'
            break
        else:
            n_grid[y][x] = found_path[i+1][2]
    return n_grid # make sure you return the shortest path

for elem in search(grid, init, goal, cost):
    print elem