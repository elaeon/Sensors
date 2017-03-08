#!/usr/bin/python2.7
from threading import Thread 
from SocketServer import ThreadingMixIn 
import socket
import logging

formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("test_server")
hdlr = logging.FileHandler('/tmp/server_echo.log')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)

# Multithreaded Python server : TCP Server Socket Thread Pool
class ClientThread(Thread): 
 
    def __init__(self, conn, ip, port): 
        Thread.__init__(self) 
        self.ip = ip 
        self.port = port
        self.conn = conn
 
    def run(self): 
        while True : 
            data = self.conn.recv(2048) 
            #conn.send(MESSAGE)  # echo 
            if not data: break
            logger.info("Data: {}".format(data))
        self.conn.close()

# Multithreaded Python server : TCP Server Socket Program Stub
HOST = '127.0.0.1' 
PORT = 2003
BUFFER_SIZE = 20

def single():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.listen(1)
    conn, addr = s.accept()
    while 1:
        data = conn.recv(1024)
        if not data: break
        #conn.sendall(data)
        print(data)
    conn.close()


def multi(): 
    tcpServer = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
    tcpServer.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) 
    tcpServer.bind((HOST, PORT)) 
    threads = [] 
     
    while True: 
        tcpServer.listen(4)
        (conn, (ip, port)) = tcpServer.accept()
        newthread = ClientThread(conn, ip, port) 
        newthread.start() 
        threads.append(newthread) 
     
    for t in threads: 
        t.join() 


if __name__ == '__main__':
    #single()
    multi()
