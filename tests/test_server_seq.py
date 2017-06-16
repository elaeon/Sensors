#!/usr/bin/python3
import asyncore
import socket
import time
import random

numbers = []
error = False

def test():
    global numbers
    global error
    numbers = sorted(map(int, (e for e in numbers if e != '')))
    for n0, n1 in zip(numbers, numbers[1:]):
        if abs(n0 - n1) > 1:
            print(numbers)
            print("ERROR LOST NUMBER", n0, n1)
            error = True
        elif error is False:
            numbers = []

class EchoHandler(asyncore.dispatcher_with_send):

    def handle_read(self):
        data = self.recv(8192)
        print(data)
        numbers.extend(data.split('\r\n\r\n'))
        test()


class EchoServer(asyncore.dispatcher):

    def __init__(self, host, port):
        asyncore.dispatcher.__init__(self)
        self.create_socket(socket.AF_INET, socket.SOCK_STREAM)
        self.set_reuse_addr()
        self.bind((host, port))
        self.listen(5)

    def handle_accept(self):
        pair = self.accept()
        if pair is None:
            return
        else:
            sock, addr = pair
            print('Incoming connection from %s' % repr(addr))
            handler = EchoHandler(sock)

if __name__ == '__main__':
    ip_address = input("ip:")
    server = EchoServer(ip_address, 8080)
    asyncore.loop()

