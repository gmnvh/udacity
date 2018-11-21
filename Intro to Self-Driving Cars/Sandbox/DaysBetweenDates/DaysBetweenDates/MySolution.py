def isLeap(year):
    if year % 4 == 0:
        if year % 100 == 0:
            if year % 400 == 0:
                return True
            else:
                return False
        else:
            return True
    else:
        return False

def daysBeforeMonth(month, year):
    months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    rsp = sum(months[0:(month-1)])
    if isLeap(year) and month > 2:
        rsp += 1

    return rsp

def my_days_between_dates(y1, m1, d1, y2, m2, d2):
    """
    Calculates the number of days between two dates.
    """
    
    # TODO - by the end of this lesson you will have
    #  completed this function. You do not need to complete
    #  it yet though! 

    total_d1 = daysBeforeMonth(m1, y1)
    total_d1 += d1

    total_d2 = 0
    for i in range(y1, y2):
        total_d2 += (366) if isLeap(i) else (365)

    total_d2 += daysBeforeMonth(m2, y2)
    total_d2 += d2

    rsp = total_d2-total_d1
    return (rsp)
