# ----------
# User Instructions:
# 
# Define a function, search() that returns a list
# in the form of [optimal path length, row, col]. For
# the grid shown below, your function should output
# [11, 4, 5].
#
# If there is no valid path from the start point
# to the goal, your function should return the string
# 'fail'
# ----------

# Grid format:
#   0 = Navigable space
#   1 = Occupied space

grid = [[0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1, 0]]
n_grid = [elem for elem in grid]

# Initialize expand table
expand = [[-1 for x in range(len(grid[0]))] for y in range(len(grid))]

init = [0, 0]
goal = [len(grid)-1, len(grid[0])-1]
cost = 1
l_open = []

delta = [[-1, 0], # go up
         [ 0,-1], # go left
         [ 1, 0], # go down
         [ 0, 1]] # go right

delta_name = ['^', '<', 'v', '>']


def search(grid, init, goal, cost):
    # ----------------------------------------
    # insert code here
    # ----------------------------------------
    l_open = [[0, init[0], init[1]]]
    exp_counter = 0
    expand[init[0]][init[1]] = exp_counter
    grid[init[0]][init[1]] = 1
    
    while True:
        smallest_g = 0
        smallest_g_index = 0
        for i in range(len(l_open)):
            if l_open[i][0] < smallest_g:
                smallest_g = l_open[i][0]
                smallest_g_index = i
                
        pos = l_open[smallest_g_index]
    
        if pos[1] == goal[0] and pos[2] == goal[1]:
            return pos
    
        # Make a list of possible moves
        moves = []
        for move in delta:
            new_y = pos[1] + move[0]
            new_x = pos[2] + move[1]
        
            if new_x < 0 or new_x >= len(grid[0]):
                continue
            if new_y < 0 or new_y >= len(grid):
                continue
        
            if grid[new_y][new_x] == 0:
                grid[new_y][new_x] = 1
                new_cost = pos[0] + cost
                exp_counter += 1
                expand[new_y][new_x] = exp_counter
                moves.append([new_cost, new_y, new_x])
    
        for move in moves:
            l_open.append(move)
        del l_open[smallest_g_index]
    
        if len(l_open) == 0:
            return 'fail'
    
    return

search_res = search(grid, init, goal, cost)
print 'Search result: ', search_res
print 'Expand table'
for e in expand:
    print e
    
#exp_max_end = max(max(expand))
#print 'max: ', exp_max_end
#end = False
#x = init[1]
#y = init[0]
#path = [['-' for xs in range(len(grid[0]))] for ys in range(len(grid))]

def myprint(new_x, new_y, new_p):
    #print 'new x: ', new_x
    #print 'new y: ', new_y
    #for elem in expand:
    #    print elem
    #print 'Appended path'
    #for elem in new_p:
    #    print elem
    #a = raw_input('')
    pass

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
                        myprint(x2, y2, new_p)        
    if found is False:
        return p
    else:
        new_paths = find_next(new_paths)
        return new_paths

op_path = search_res[0]
paths = [[[0,0]]]
new_paths = find_next(paths)               

for p in new_paths:
    if len(p) == op_path + 1:
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

for elem in n_grid:
    print elem