import socket
import time
import random

UDP_IP = "172.30.47.170"
UDP_PORT = 5006
MESSAGE = "XXX"

print "UDP target IP:", UDP_IP
print "UDP target port:", UDP_PORT
print "message:", MESSAGE

sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP

#sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))

for i in range(400):
    print 'Sending message'
    #sock.connect((UDP_IP, UDP_PORT))
    #sock.send(MESSAGE)
    #sock.close()
    sock.sendto(str(random.random()), (UDP_IP, UDP_PORT))
    time.sleep(0.5)
