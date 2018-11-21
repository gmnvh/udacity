import unittest
from DaysBetweenDates import *

class Test_test_solution(unittest.TestCase):
    def test_nextDay(self):
        try:
            next = nextDay(2000, 12, 31)
            assert(next == (2001, 1, 1))
            next = nextDay(2011, 2, 28)
            assert(next == (2011, 3, 1))
            next = nextDay(2012, 2, 28)
            assert(next == (2012, 2, 29))
        except AssertionError as e:
            self.fail("Test failed")
        except:
            self.fail("Unknown error")

if __name__ == '__main__':
    unittest.main()
