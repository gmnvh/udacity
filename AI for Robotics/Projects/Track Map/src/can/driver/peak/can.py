'''
CAN Driver for Peak hardware.
API can be downloaded in https://www.phytools.com/PCAN_APIs_s/1867.htm

Author: Gustavo Muller Nunes
'''

import os
import time
import logging
import threading
import pcanbasic as pcb

os.environ['PATH'] = os.path.dirname(pcb.__file__) + os.pathsep + \
                     os.environ['PATH']
logger = logging.getLogger("CanDriver")

class CanException(Exception):
    '''
        TODO [GMN] Implement exception class
    '''
    pass

def _log_tx(buf):
    logger.debug("TX: %s", repr(dict(ID=hex(buf.ID),
                                     MSGTYPE=buf.MSGTYPE,
                                     LEN=buf.LEN,
                                     DATA=list(hex(v) for v in buf.DATA))))

def _log_rx(buf, rx_debug=False):
    if rx_debug is True:
        logger.debug("RX: %s", repr(dict(ID=hex(buf[1].ID),
                                         MSGTYPE=buf[1].MSGTYPE,
                                         LEN=buf[1].LEN,
                                         DATA=list(hex(v)
                                                   for v in buf[1].DATA))))

class CanDriver(object):
    def __init__(self, channel=pcb.PCAN_USBBUS1, baudrate=pcb.PCAN_BAUD_500K,
                 can_filter=[0, 0]):
        self.channel = channel
        self.baudrate = baudrate
        self.pcan = pcb.PCANBasic()
        self.pcan.Initialize(channel, baudrate)
        self.pcan.FilterMessages(channel, can_filter[0], can_filter[1],
                                 pcb.PCAN_MODE_STANDARD)
        self.pcan.Reset(channel)
        self.stmin = 0

    def send_frame(self, can_id, data):
        '''
        Transmit a single 8 byte frame

        Args:
            can_id: CAN message id
            data:   Array of data to transmit up to 8 bytes in length - less
                    than 8 will be padded with zeros.

        Returns:
            Error code from PCAN tool is error occurred or zero if no error.
        '''
        if len(data) > 8:
            raise ReferenceError('Single frame length cannot exceed 8 bytes')
        buf = pcb.TPCANMsg()
        buf.ID = can_id
        buf.LEN = 8
        buf.MSGTYPE = pcb.PCAN_MESSAGE_STANDARD
        for idx in range(8):
            if idx < len(data):
                buf.DATA[idx] = data[idx]
            else:
                buf.DATA[idx] = 0
        _log_tx(buf)
        error = self.pcan.Write(self.channel, buf)
        if error != 0:
            logger.error('0x%X: %s', error, self.pcan.GetErrorText(error)[1])
        return error

    def get_frame(self, timeout=0, rx_debug=False):
        '''
        Get a single frame of 8 bytes from the PCAN tool buffer. The timeout
        will allow a determinate wait time before giving up on reception.

       Args:
            timeout:  Timeout in seconds - set to zero to disable timeout and
                      wait forever
            rx_debug: Enable a debug message with the received message. Only
                      to be enabled when messages are not very offen and for
                      debug purpose

        Returns:
            Response [id, data] if received
        '''
        start_time = time.time()
        while True:
            resp = self.pcan.Read(self.channel)
            if resp[1].ID != 0:
                _log_rx(resp, rx_debug)
                return resp[1].ID, resp[1].DATA
            if timeout != 0 and (time.time() - start_time) > timeout:
                raise CanException()

if __name__ == "__main__":
    response = None

    def cancel():
        global response
        response = raw_input()

    def receive():
        global pcan
        pcan.get_frame(rx_debug=True)

    logging.basicConfig(level=logging.DEBUG)
    logger.info("Test Code for CanDriver Peak - Press any key and ENTER to cancel")

    pcan = CanDriver(can_filter=[0x778, 0x779])

    user = threading.Thread(target=cancel)
    user.daemon = True
    user.start()

    rx = threading.Thread(target=receive)
    rx.daemon = True
    rx.start()

    while True:
        user.join(1)
        pcan.send_frame(0x779, [1, 2, 3, 4, 5, 6, 7, 8])
        if response != None:
            break
