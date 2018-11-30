import unittest
from Route_Planner import distance_between_points


class Test_test1(unittest.TestCase):
    def test_distance_between_points(self):
        try:
            assert(distance_between_points([0,0],[0,0]) == 0)
            assert(distance_between_points([-1,1],[3,4]) == 5)
        except:
            self.fail("distance_between_points failed")

if __name__ == '__main__':
    unittest.main()
