import logging
import CanSimulator


logger = logging.getLogger("CanMessage")

class CanMessage(object):
    id = 0
    tx_rate = 0

    def __init__(self, id):
        self.id = id
    
    def rx(self, data):
        logger.error('CanMessage should be extended to be used')
    
    def tx(self, data):
        logger.error('CanMessage should be extended to be used')


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    can_sim = CanSimulator.CanSimulator()
    diag_msg = CanMessage(0x778)
    
    can_sim.rx_msg_subscrive(diag_msg)
    can_sim.join(10)
