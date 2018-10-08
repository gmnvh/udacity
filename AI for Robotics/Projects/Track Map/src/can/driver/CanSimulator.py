'''
CanSimulator will create a group of threads to handle the reception and
perodic transmission of CAN messages.
'''

import time
import logging
import threading
from peak import can

RX_THREAD_NAME = 'CanRxThread'
logger = logging.getLogger("CanThread")

class CanRxThread(threading.Thread):
    '''
    CanRxThread is a thread to receive CAN messages.
    '''

    def __init__(self, can_driver):
        '''
        Initialization function for CANRxThread only initialize local
        variables.

        Args:
            can_driver: CAN Driver object
        '''
        threading.Thread.__init__(self, name=RX_THREAD_NAME)
        logger.debug('Creating CanRxThread')

        self.can_driver = can_driver
        self.subscribe = []
        return

    def run(self):
        '''
        The run method for this thread will check for new CAN messages and
        call the callback function for the message ID in the subcribed list.
        '''
        while True:
            msg_id, msg_data = self.can_driver.get_frame(rx_debug=False)
            for msg in self.subscribe:
                if msg.id == msg_id:
                    msg.rx(msg_data)

    def msg_subscribe(self, msg):
        '''
        Add a message to the subscribe list.

        Args:
            msg(CanMessage): CAN Message object
        '''
        self.subscribe.append(msg)


class CanSimulator(object):
    '''
    This class will create all threads to receive and send CAN messages.
    '''
    can_driver = None

    def __init__(self):
        # Initialize CAN Driver
        if CanSimulator.can_driver is None:
            CanSimulator.can_driver = can.CanDriver(can_filter=[0x155, 0x78B])
            logger.debug('CAN Driver initialized')

        # Check if CanRxThread already exist
        list_threads = threading.enumerate()
        for th in list_threads:
            if th.getName() == RX_THREAD_NAME:
                logger.debug('CanRxThread already exist')
                self.rx_thread = th
                return

        # Create CanRxThread
        self.rx_thread = CanRxThread(self.can_driver)
        self.rx_thread.daemon = True
        self.rx_thread.start()
        return

    def rx_msg_subscrive(self, msg):
        '''
        Subscribe RX CAN messages.

        Args:
            msg(CanMessage): CAN message object to be called in case message
                             is received.
        '''
        self.rx_thread.msg_subscribe(msg)

    def tx_msg_subscribe(self, msg):
        '''
        Subscribe TX CAN messages.

        Args:
            msg(CanMessage): CAN message object to be called in case message
                             is needs to be sent (periodically).
        '''
        pass

    def join(self, timeout=0):
        '''
        Wait all CanSimulator threads to resume.
        '''
        self.rx_thread.join(timeout)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    sim1 = CanSimulator()

    time.sleep(1)
    sim1.rx_thread.msg_subscribe(0x778)

    logger.debug('Delay main before subscribe for next message')
    time.sleep(2)

    sim2 = CanSimulator()
    sim2.rx_thread.msg_subscribe(0x78B)

    sim1.join(timeout=2)
