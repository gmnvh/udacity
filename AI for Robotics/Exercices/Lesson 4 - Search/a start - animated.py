import numpy as  np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Initialize problem
grid = [[0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 1, 0, 1, 1, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0]]

# Problem configuration
init = [0, 0]
goal = [len(grid)-1, len(grid[0])-1]
cost = 1

# Robot move function
delta = [[-1, 0 ], # go up
         [ 0, -1], # go left
         [ 1, 0 ], # go down
         [ 0, 1 ]] # go right

delta_name = ['^', '<', 'v', '>']

# Initialize heuristic function
heuristic = [[9, 8, 7, 6, 5, 4],
             [8, 7, 6, 5, 4, 3],
             [7, 6, 5, 4, 3, 2],
             [6, 5, 4, 3, 2, 1],
             [5, 4, 3, 2, 1, 0]]

heuristic_old = [[0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]]

# Initialize expand table
expand = [[-1 for x in range(len(grid[0]))] for y in range(len(grid))]

# Change the block cells to -2 (so then can be colored in a different color)
np_grid = np.array(grid)
np_expand = np.array(expand)
np_expand = (np_grid * np_expand) -1

'''
    A* implementation
'''
def search(grid,init,goal,cost,heuristic):
    closed = [[0 for col in range(len(grid[0]))] for row in range(len(grid))]
    closed[init[0]][init[1]] = 1

    expand = [[-1 for col in range(len(grid[0]))] for row in range(len(grid))]
    action = [[-1 for col in range(len(grid[0]))] for row in range(len(grid))]

    x = init[0]
    y = init[1]
    g = 0
    f = heuristic[x][y]
    h = f
    open = [[f, g, h, x, y]]

    found = False  # flag that is set when search is complete
    resign = False # flag set if we can't find expand
    count = 0
    
    while not found and not resign:
        if len(open) == 0:
            resign = True
            return "Fail"
        else:
            open.sort()
            open.reverse()
            next = open.pop()
            x = next[3]
            y = next[4]
            g = next[1]
            expand[x][y] = count
            count += 1
            
            if x == goal[0] and y == goal[1]:
                found = True
            else:
                for i in range(len(delta)):
                    x2 = x + delta[i][0]
                    y2 = y + delta[i][1]
                    if x2 >= 0 and x2 < len(grid) and y2 >=0 and y2 < len(grid[0]):
                        if closed[x2][y2] == 0 and grid[x2][y2] == 0:
                            g2 = g + cost
                            h2 = heuristic[x2][y2]
                            f2 = g2 + h2
                            open.append([f2, g2, h2, x2, y2])
                            closed[x2][y2] = 1
                            action[x2][y2] = i

    return g, expand, action

'''
    Solving the problem
'''
g, expand, action = search(grid, init, goal, cost, heuristic)
np_expanded = np.array(expand)
np_action = np.array(action)
print '\r\nTotal cost: ', g
print '\r\nExpanded table\r\n', np_expanded
print '\r\nAction table\r\n', np_action

path = [[' ' for col in range(len(grid[0]))] for row in range(len(grid))]
x = goal[0]
y = goal[1]
path[init[0]][init[1]] = 'Start'
path[x][y] = 'Goal'
ani_path = []

while x != init[0] or y != init[1]:
    x2 = x - delta[action[x][y]][0]
    y2 = y - delta[action[x][y]][1]
    path[x2][y2] = delta_name[action[x][y]]
    ani_path.append([x2, y2, delta_name[action[x][y]]])
    x = x2
    y = y2

print '\r\nPath\r\n', np.array(path)

'''
    Create animated plot
'''
# Initialize global variables
x = 0
y = 0
ind = 0
ani_expanded = np.array(np_expand)
ani_parts = 0

def animated(*args):
    global x, y, ax, ind, g, ani_expanded, ani_parts, img
    
    # Show expanded animation
    if ani_parts == 0:
        itemindex = np.where(np_expanded==ind)
        ind = (ind + 1) % (np_expanded[goal[0], goal[1]]+1)

        ani_expanded[itemindex] = np_expanded[itemindex][0]
        img.set_array(ani_expanded)

        if ind == 0:
            ani_parts = 1
    
    # Show path animation
    if ani_parts == 1:
        spath = ani_path[ind]
        ax.text(spath[1], spath[0], spath[2], color='blue', ha='center', va='center')
        ind = (ind + 1) % (len(ani_path))
        
        if ind == 0:
            # Pause at the end for some seconds
            plt.pause(2) 
            
            # Clear everything to display again
            img.axes.clear()
            ani_parts = 0
            
            # Recreate the img
            img = ax.imshow(np_expand, interpolation='nearest', cmap = cmap, norm=norm,
                            animated=True)
            ani_expanded = np.array(np_expand)
            ax.text(init[1], init[0], 'S', color='blue', ha='center', va='top')
            ax.text(goal[1], goal[0], 'Goal', color='blue', ha='center', va='center')
                       
    return img, 
    
# Create plot
fig = plt.figure()
ax = fig.add_subplot(111)

# Make a color map of fixed colors
cmap = mpl.colors.ListedColormap(['black', 'white', '#5f9e8f'])

# Create the bounds
# -2 : blocked cells : black
# -1 : not expanded cells : white
# >0 : expanded cells: green
bounds=[-2, -1, 0, 1]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

# Create image
img = ax.imshow(np_expand, interpolation='nearest', cmap = cmap, norm=norm,
                 animated=True)
ani = animation.FuncAnimation(fig, animated, interval=600, blit=False)

# Print Start and goal
ax.text(init[1], init[0], 'S', color='blue', ha='center', va='top')
ax.text(goal[1], goal[0], 'Goal', color='blue', ha='center', va='center')

# Make a color bar
plt.colorbar(img, cmap=cmap, norm=norm, boundaries=bounds, ticks=[-2, -1, 0])
plt.show()