
def distanceFromLineGiven3Points(linePoint1, linePoint2, refPoint):
    '''
        Calculate the distance between a point and line. The line
        is given as two points.
    '''
    Rx = refPoint[0] - linePoint1[0]
    Ry = refPoint[1] - linePoint1[1]
    Px = linePoint2[0] - linePoint1[0]
    Py = linePoint2[1] - linePoint1[1]
    
    d = ((Ry * Px) - (Rx * Py)) / ((Px * Px) + (Py * Py))
    return d

def scalarProjectionGiven3Points(linePoint1, linePoint2, refPoint):
    '''
        Calculates the scalar projection (or scalar component) of 
        a Euclidean vector a in the direction of a Euclidean vector b.
    '''
    Rx = refPoint[0] - linePoint1[0]
    Ry = refPoint[1] - linePoint1[1]
    Px = linePoint2[0] - linePoint1[0]
    Py = linePoint2[1] - linePoint1[1]
    
    u = ((Rx * Px) + (Ry * Py)) / ((Px * Px) + (Py *Py))
    return u