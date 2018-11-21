import unittest
from MySolution import *
from DaysBetweenDates import *

class Test_MySolution(unittest.TestCase):
    def test_A(self):
        try:
            test_my_days_between_dates()
        except:
            self.fail("Test failed")

if __name__ == '__main__':
    unittest.main()
