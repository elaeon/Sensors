import asyncore, socket, threading
import time
import random
from queuelib import FifoDiskQueue


CHUNK_SIZE = 5
#def generator():
#    while True:
#        yield "{}.{}".format(10, int(time.time()))


class Client(asyncore.dispatcher):

    def __init__(self, host, port=8080, name="client_data", delay=1, data=None):
        asyncore.dispatcher.__init__(self)
        self.create_socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connect((host, port))
        self.buffer = ""
        self.name = name
        self.delay = delay
        self.data = data

        print("NEW")
        self.t = SenderThread(self, self.data)
        self.t.start()

    def handle_connect(self):
        print("conected")

    def handle_close(self):
        print("closed")
        self.close()
        fq = FifoDiskQueue("{}.fifo.sql".format(self.name))
        for i in range(10):
            obj = next(self.data)
            fq.push(obj)
            print("SAVED", obj)
            time.sleep(self.delay)
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

    def __init__(self, client, data):
        super(SenderThread, self).__init__()
        self.client = client
        self.data = data

    def stop(self):
        self._stop_t = True

    def run(self):
        counter = 0
        timestamps = []
        while self._stop_t == False:
            counter += 1
            time.sleep(1)
            timestamps.append(next(self.data))
            if counter % CHUNK_SIZE == 0:
                print("sending data from thread")
                timestamps.append('')
                self.client.send_data('\r\n\r\n'.join(timestamps))
                timestamps = []


class SyncData(object):
    def __init__(self, name, carbon_server, carbon_port=2003, delay=3, 
                delay_error_sensor=0.2, delay_error_connection=2):
        self.name = name
        self.DELAY = delay
        self.CARBON_SERVER = carbon_server
        self.CARBON_PORT = carbon_port
        self.logger = self.logging_setup()

    def logging_setup(self):
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        logger = logging.getLogger(self.name)
        hdlr = logging.FileHandler('/tmp/{}.log'.format(self.name))
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr)
        logger.setLevel(logging.INFO)
        return logger

    def run(self, generator):
        data = generator()
        while True:
            client = Client(self.CARBON_SERVER, self.CARBON_PORT, 
                            name=self.name, delay=self.delay, data=data)
            asyncore.loop(timeout=1)


class SyncDataFromDisk(SyncData):
    def slow(self, messages):
        for message in messages:
            time.sleep(.5)
            yield message

    def run(self):
        while True:
            queue = FifoDiskQueue("{}.fifo.sql".format(self.name))
            self.logger.info("Data saved: {}".format(len(queue)))
            if len(queue) > 0:
                messages = queue.pull()
                client = Client(self.CARBON_SERVER, self.CARBON_PORT, 
                            name=self.name, delay=self.delay, data=data)
                asyncore.loop(timeout=1)
            queue.close()
            time.sleep(3)
