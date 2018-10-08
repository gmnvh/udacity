import sys
import ctypes
import logging
import datetime

sys.path.insert(0, r'../../driver')
from CanMessage import CanMessage
import CanSimulator

logger = logging.getLogger("GmMsg1E5")

class GmMsg1E5(CanMessage):
    def __init__(self):
        super(GmMsg1E5, self).__init__(0x1E5)
        
        
    def rx(self, data):
        str_wh_ag = (data[1] << 8) + data[2]
        str_wh_ag = ctypes.c_int16(str_wh_ag).value
        factor = 0.0625
        offset = 0
        str_wh_ag = offset + str_wh_ag * factor
        if str_wh_ag > 2047.9375 or str_wh_ag < -2048:
            raise Exception('Data out of range')
        


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    can_sim = CanSimulator.CanSimulator()
    diag_msg = GmMsg1E5()
    
    can_sim.rx_msg_subscrive(diag_msg)
    can_sim.join(100)