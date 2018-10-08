import socket

UDP_IP = "127.0.0.1"
UDP_PORT = 5005

print 'Getting ready'
print socket.gethostbyname(socket.gethostname())

sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
sock.bind((socket.gethostbyname(socket.gethostname()), UDP_PORT))


while True:
    data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
    print "received message:", data, ':', addr