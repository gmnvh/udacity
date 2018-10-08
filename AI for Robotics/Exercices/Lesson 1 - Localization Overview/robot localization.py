# Robot localization

# Initial Uniform distribution
# i.e. given a space with 5 cells, the robot has 0.2 chances of been in a cell Xi
n = 5
p = [1.0/n]*n

# Now the robot can sense the cells (green or red)
# |G|R|R|G|G|
#
# If it senses red:
#                   red cells   * 0.6 
#                   green cells * 0.2

pHit = 0.6
pMiss = 0.2

world=['g','r','r','g','g']
Z = 'r'

new_p = []
for value, color in zip(p, world):
    hit = (Z == color)
    new_p.append(value * (hit * pHit + (1-hit) * pMiss))

print 'Probability distribution before and after the sensor'
print ["{0:0.2f}".format(i) for i in p]
print ["{0:0.2f}".format(i) for i in new_p]

# The new probability does not add to 1. It is necessary to do
# renormalization. So we need to divide each member by the sum of all.

new_p_sum = sum(new_p)
n_new_p = [x / new_p_sum for x in new_p]

print ''
print 'Sum of new p:', new_p_sum
print 'Normalizazed new p: ', ["{0:0.2f}".format(i) for i in n_new_p]
print 'Sum of normalized new p: ', sum(n_new_p)

# Let's create a function to return the normalized distribution after sense
def sense(p, Z):
    q = []
    for value, color in zip(p, world):
        hit = (Z == color)
        q.append(value * (hit * pHit + (1-hit) * pMiss))
    s = sum(q)
    q = [x / s for x in q]
    return q

# Let's process multiple sensor inputs
measurements = ['r', 'g']

new_p = p[:]
for measurement in measurements:
    new_p = sense(new_p, measurement)

print ''
print 'Probability distribution before and after the sensor measuments'
print ["{0:0.2f}".format(i) for i in p]
print ["{0:0.2f}".format(i) for i in new_p]

# Add robot motion assuming a cyclic world
#  |->|1|2|3|4|5| -|
#  |---------------|

new_p = sense(p, 'r')
moved_p = []
for k in range(len(new_p)):
    moved_p.append(new_p[k-1])

print ''
print 'Sensing red and moving one to the right'
print ["{0:0.2f}".format(i) for i in new_p]
print ["{0:0.2f}".format(i) for i in moved_p]

# Let's make it a function where U > 0 is shift to right and < 0 is shift to
# left
def move(p, U):
    U = U % len(p)
    q = p[-U:] + p[:-U]
    return q

# Simulate a complete move to right adding measurments
new_p = p
new_p = sense(new_p, 'r')
new_p = move(new_p, 1)
new_p = sense(new_p, 'g')

print ''
print 'Probability distribution moving 1 to the right with measurments'
print ["{0:0.2f}".format(i) for i in p]
print ["{0:0.2f}".format(i) for i in new_p]

# Inexact Motion
# Let's add some probability on moving to the right place
#
# If U = 2
# p(Xi+2|Xi) = 0.8
# p(Xi+1|Xi) = 0.1
# p(Xi+3|Xi) = 0.1

# Calculate the probability given the following distribuition
new_p = [0, 1, 0, 0, 0]

def inexactMove(p, U, pHit):
    U = U % len(p)
    q = p[-U:] + p[:-U]
    
    q1 = [x * pHit for x in q]
    q2 = move(q, 1)
    q2 = [x *(1-pHit)/2 for x in q2]
    q3 = move(q, -1)
    q3 = [x * (1-pHit)/2 for x in q3]
    
    ''' Udacity solution
    q = []
    for i in range(len(p)):
        s = pExact * p[(i-U) % len(p)]
        s = s + pOvershoot * p[(i-U-1) % len(p)]
        s = s + pUndershoot * p[(i-U+1) % len(p)]
        q.append(s)
    '''
    return [x+y+z for x,y,z in zip(q1,q2,q3)]

moved_new_p = inexactMove(new_p, 2, 0.8)

print '\r\nInexact Move Example 1'
print ["{0:0.2f}".format(i) for i in new_p]
print ["{0:0.2f}".format(i) for i in moved_new_p]

new_p = [0, 0.5, 0, 0.5, 0]
moved_new_p = inexactMove(new_p, 2, 0.8)

print '\r\nInexact Move Example 2'
print ["{0:0.2f}".format(i) for i in new_p]
print ["{0:0.2f}".format(i) for i in moved_new_p]

moved_new_p = inexactMove(p, 2, 0.8)
print '\r\nInexact Move Example 3'
print ["{0:0.2f}".format(i) for i in p]
print ["{0:0.2f}".format(i) for i in moved_new_p]

# Moving the robot twice
new_p = [0, 1, 0, 0, 0]
moved_new_p = inexactMove(new_p, 1, 0.8)
moved_new_p = inexactMove(moved_new_p, 1, 0.8)
print '\r\nMoving the robot twice'
print ["{0:0.2f}".format(i) for i in new_p]
print ["{0:0.2f}".format(i) for i in moved_new_p]

# Moving the robot 1000
# After too many inexact moves, the probability is uniform again
new_p = [0, 1, 0, 0, 0]
for i in range(1000):
    new_p = inexactMove(new_p, 1, 0.8)

print '\r\nMoving the robot 1000 times'
print ["{0:0.2f}".format(i) for i in new_p]

# Given the list motions=[1,1] which means the robot 
# moves right and then right again, compute the posterior 
# distribution if the robot first senses red, then moves 
# right one, then senses green, then moves right again, 
# starting with a uniform prior distribution.
measurements = ['r', 'g']
motions = [1, 1]

new_p = p[:]
for motion, Z in zip(motions, measurements):
    new_p = sense(new_p, Z)
    new_p = inexactMove(new_p, motion, 0.8)
    
print '\r\nDistribution given a list of motion and measurements: Example 1'
print ["{:4}".format(i) for i in world]
print ["{0:0.2f}".format(i) for i in p]
print ["{0:0.2f}".format(i) for i in new_p]

# Change the initial position
measurements = ['r', 'r']
motions = [1, 1]

new_p = p[:]
for motion, Z in zip(motions, measurements):
    new_p = sense(new_p, Z)
    new_p = inexactMove(new_p, motion, 0.8)
    
print '\r\nDistribution given a list of motion and measurements : Example 2'
print ["{:4}".format(i) for i in world]
print ["{0:0.2f}".format(i) for i in p]
print ["{0:0.2f}".format(i) for i in new_p]




    