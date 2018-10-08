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
        [0, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0]]
n_grid = list(grid)

# Initialize expand table
expand = [[-1 for x in range(len(grid[0]))] for y in range(len(grid))]

init = [0, 0]
goal = [len(grid)-1, len(grid[0])-1]
print 'goal: ', goal
cost = 1
l_open = []

delta = [[-1, 0], # go up
         [ 0,-1], # go left
         [ 1, 0], # go down
         [ 0, 1]] # go right

delta_name = ['^', '<', 'v', '>']

def grid_reset():
    global grid
    grid = [[0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 0]]
   
def search_step_init():
    global l_open, grid
    
    l_open = [[0, init[0], init[1]]]
    grid[init[0]][init[1]] = 2
    
def search_step(grid, init, goal, cost):
    smallest_g = 0
    smallest_g_index = 0
    for i in range(len(l_open)):
        if l_open[i][0] < smallest_g:
            smallest_g = l_open[i][0]
            smallest_g_index = i
                
    print 'get next in open\r\n********'
    print l_open[smallest_g_index]
    pos = l_open[smallest_g_index]
    
    if pos[1] == goal[0] and pos[2] == goal[1]:
        grid_reset()
        search_step_init()
        return pos
    
    # Make a list of possible moves
    moves = []
    for move in delta:
        new_y = pos[1] + move[0]
        new_x = pos[2] + move[1]
        print '\tmove: ', move
        print '\tnew_y: ', new_y
        print '\tnew_x: ', new_x
        
        if new_x < 0 or new_x >= len(grid[0]):
            continue
        if new_y < 0 or new_y >= len(grid):
            continue
        
        if grid[new_y][new_x] == 0:
            grid[new_y][new_x] = 2
            new_cost = pos[0] + cost
            moves.append([new_cost, new_y, new_x])
    
    print '\r\npossible moves\r\n********'
    print moves
    
    for move in moves:
        l_open.append(move)
    del l_open[smallest_g_index]
    
    if len(l_open) == 0:
        grid_reset()
        search_step_init()
        return 'fail'
    
    return

def search(grid,init,goal,cost):
    # ----------------------------------------
    # insert code here
    # ----------------------------------------
    l_open = [[0, init[0], init[1]]]
    grid[init[0]][init[1]] = 2
    exp_counter = 0
    expand[init[0]][init[1]] = exp_counter
    
    while True:
        smallest_g = 0
        smallest_g_index = 0
        for i in range(len(l_open)):
            if l_open[i][0] < smallest_g:
                smallest_g = l_open[i][0]
                smallest_g_index = i
                
        print 'get next in open\r\n********'
        print l_open[smallest_g_index]
        pos = l_open[smallest_g_index]
    
        if pos[1] == goal[0] and pos[2] == goal[1]:
            return pos
    
    # Make a list of possible moves
        moves = []
        for move in delta:
            new_y = pos[1] + move[0]
            new_x = pos[2] + move[1]
            print '\tmove: ', move
            print '\tnew_y: ', new_y
            print '\tnew_x: ', new_x
        
            if new_x < 0 or new_x >= len(grid[0]):
                continue
            if new_y < 0 or new_y >= len(grid):
                continue
        
            if grid[new_y][new_x] == 0:
                grid[new_y][new_x] = 2
                new_cost = pos[0] + cost
                exp_counter += 1
                expand[new_y][new_x] = exp_counter
                moves.append([new_cost, new_y, new_x])
    
        print '\r\npossible moves\r\n********'
        print moves
    
        for move in moves:
            l_open.append(move)
        del l_open[smallest_g_index]
    
        if len(l_open) == 0:
            return 'fail'
    
    return

search_res = search(grid, init, goal, cost)
print search_res
op_path = search_res[0]
for elem in expand:
    print elem
a = raw_input('')
exp_max_end = max(max(expand))
print 'max: ', exp_max_end

end = False
x = init[1]
y = init[0]
path = [['-' for xs in range(len(grid[0]))] for ys in range(len(grid))]
paths = [[[0,0]]]

def myprint(new_x, new_y, new_p):
    print 'new x: ', new_x
    print 'new y: ', new_y
    for elem in expand:
        print elem
    print 'Appended path'
    for elem in new_p:
        print elem
    #a = raw_input('')
    
def find_next(p):
    found = False
    new_paths = []
    for pin in range(len(p)):
        x = p[pin][-1][0]
        y = p[pin][-1][1]
        for move, symbol in zip(delta, delta_name):
            new_p = p[pin][:]
            print 'new p', new_p
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
        print '***************'
        print 'New paths'
        for elem in new_paths:
            print elem
        new_paths = find_next(new_paths)
        return new_paths

new_paths = find_next(paths)               
print 'Paths after\r\n'
for p in new_paths:
    print p

print '\r\n____****____\r\n'
print 'Op path: ', op_path

for p in new_paths:
    if len(p) == op_path + 1:
        found_path = p
        break
print found_path 

for elem in n_grid:
    print elem

n_grid = [[str(i) for i in y] for y in n_grid]

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
'''
end = False
while end is False:
    exp_max = 0
    exp_name = 'X'
    new_x = 0
    new_y = 0
    for move, symbol in zip(delta, delta_name):
        x2 = x + move[1]
        y2 = y + move[0]
        print 'move: ', move
        print 'x2: ', x2
        print 'y2: ', y2
        #a = raw_input('')
        if (x2 >=0  and x2 < len(grid[0])):
            if (y2 >= 0) and (y2 < len(grid)):
                exp = expand[y2][x2]
                print 'exp: ', exp
                if exp > exp_max:
                    exp_max = exp
                    new_x = x2
                    new_y = y2
                    exp_name = symbol
         
    path[y][x] = exp_name
    x = new_x
    y = new_y

    if exp_max == exp_max_end:
        path[y][x] = 'X'
        end = True   

    
    print '****'
    for elem in expand:
        print elem
    print 'selected cell: ', exp_max
    #a = raw_input('')

for elem in path:
    print elem
        
'''        
        
'''
import matplotlib as mpl
from matplotlib import pyplot
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def updatefig(*args):
    global grid, img
    search_step(grid, init, goal, cost)
    img.set_array(grid)
    return img,

search_step_init()
fig = plt.figure()

# make a color map of fixed colors
cmap = mpl.colors.ListedColormap(['white','black','green', 'blue'])
bounds=[0, 1, 2, 3, 4]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

# tell imshow about color map so that only set colors are used
img = pyplot.imshow(grid,interpolation='nearest',
                    cmap = cmap,norm=norm, animated=True)

# make a color bar
pyplot.colorbar(img,cmap=cmap,
                norm=norm,boundaries=bounds,ticks=[-0,1,2,3,4])

ani = animation.FuncAnimation(fig, updatefig, interval=500, blit=True)
pyplot.show()
'''