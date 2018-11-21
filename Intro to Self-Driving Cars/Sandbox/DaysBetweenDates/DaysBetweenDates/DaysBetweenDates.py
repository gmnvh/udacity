from MySolution import *

def dateIsBefore(year1, month1, day1, year2, month2, day2):
    if year1 < year2:
        return True
    if year1 == year2:
        if month1 < month2:
            return True
        if month1 == month2:
            return day1 < day2
    return False

def nextDay(year, month, day):
    """
    Returns the year, month, day of the next day.
    """
    days_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    days_month_leap = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    month_days = (days_month_leap) if isLeap(year) else (days_month)
    day = day + 1
    
    if day > month_days[month-1]:
        day = 1
        month += 1
    
    if month > 12:
        month = 1
        year += 1
        
    return year, month, day

def days_between_dates(year1, month1, day1, year2, month2, day2):

    assert not dateIsBefore(year2, month2, day2, year1, month1, day1)

    days = 0
    while dateIsBefore(year1, month1, day1, year2, month2, day2):
        year1, month1, day1 = nextDay(year1, month1, day1)
        days += 1

    return days

def test_my_days_between_dates():
    
    # test same day
    assert(my_days_between_dates(2017, 12, 30,
                              2017, 12, 30) == 0)
    # test adjacent days
    assert(my_days_between_dates(2017, 12, 30, 
                              2017, 12, 31) == 1)
    # test new year
    assert(my_days_between_dates(2017, 12, 30, 
                              2018, 1,  1)  == 2)
    # test full year difference
    assert(my_days_between_dates(2012, 6, 29,
                              2013, 6, 29)  == 365)
    
    print("Congratulations! Your days_between_dates")
    print("function is working correctly!")

def test_days_between_dates():
    
    # test same day
    assert(days_between_dates(2017, 12, 30,
                              2017, 12, 30) == 0)
    # test adjacent days
    assert(days_between_dates(2017, 12, 30, 
                              2017, 12, 31) == 1)
    # test new year
    assert(days_between_dates(2017, 12, 30, 
                              2018, 1,  1)  == 2)
    # test full year difference
    assert(days_between_dates(2012, 6, 29,
                              2013, 6, 29)  == 365)
    
    print("Congratulations! Your days_between_dates")
    print("function is working correctly!")
    
test_my_days_between_dates()
test_days_between_dates()
