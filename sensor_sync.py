import asyncore, socket, threading
import time
import random
from queuelib import FifoDiskQueue


CHUNK_SIZE = 5
def generator():
    while True:
        yield "{}.{}".format(10, int(time.time()))


class Client(asyncore.dispatcher):

    def __init__(self, host):
        asyncore.dispatcher.__init__(self)
        self.create_socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connect( (host, 8080) )
        self.buffer = ""

        print("NEW")
        self.t = SenderThread(self)
        self.t.start()

    def handle_connect(self):
        print("conected")

    def handle_close(self):
        print("closed")
        self.close()
        fq = FifoDiskQueue("data.sqlite")
        for i in range(10):
            obj = next(data)
            fq.push(obj)
            print("SAVED", obj)
            time.sleep(1)
        fq.close()
        self.t.stop()

    def handle_read(self):
        print(self.recv(8192))

    def handle_error(self):
        print("lost")

    def writable(self):
        return (len(self.buffer) > 0)

    def handle_write(self):
        sent = self.send(self.buffer)
        self.buffer = self.buffer[sent:]

    def send_data(self, data):
        self.buffer = bytes(data, 'ascii')


class SenderThread(threading.Thread):
    _stop_t = False

    def __init__(self, client):
        super(SenderThread, self).__init__()
        self.client = client

    def stop(self):
        self._stop_t = True

    def run(self):
        counter = 0
        timestamps = []
        while self._stop_t == False:
            counter += 1
            time.sleep(1)
            timestamps.append(next(data))
            if counter % CHUNK_SIZE == 0:
                print("sending data from thread")
                timestamps.append('')
                self.client.send_data('\r\n\r\n'.join(timestamps))
                timestamps = []



data = generator()
while True:
    client = Client('192.168.52.152')
    asyncore.loop(timeout=1)
