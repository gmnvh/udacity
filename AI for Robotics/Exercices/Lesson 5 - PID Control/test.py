from math import *

radius = 25

x = 0.
y = 26.

if x < radius:
    # Area 1 - First turn
    dx = radius - x
    dy = radius - y
            
    alpha = atan(dy/dx)
    print alpha
    if alpha == 0.0:
        cte = -x
    else:
        cte = abs(((dy)/sin(alpha))) - radius
    print 'Area 1', cte
            
elif x < (3 * radius):
    # Area 2 - straigth line
    if y > radius:
        cte = y - (2*radius)
    else:
        cte = -y
    print 'Area 2', cte
else:
    # Area 3 - Second turn
    dx = (3*radius) - x
    dy = (radius) - y
            
    alpha = atan(dy/dx)
    print alpha
    if alpha == 0.0:
        cte = -x
    else:
        cte = abs((dy/sin(alpha))) - radius
    print 'Area 3', cte
