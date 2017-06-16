import asyncore, socket, threading
import time
import logging

from queuelib import FifoDiskQueue

global_l = []


class Client(asyncore.dispatcher):

    def __init__(self, host, port=8080, name="client_data", delay=1,
                formater=None, batch_size=5):
        asyncore.dispatcher.__init__(self)
        self.create_socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connect((host, port))
        self.buffer = ""
        self.name = name
        self.delay = delay
        self.formater = formater

        print("NEW")
        self.t_net = SenderThread(self, self.formater, 
                            delay=self.delay, batch_size=batch_size)
        self.t_net.start()

    def handle_connect(self):
        print("conected")

    def handle_close(self):
        print("closed")
        self.close()
        self.t_net.stop()

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
        data_bytes = bytes('', 'ascii')
        data_string = []
        for elem in data:
            if isinstance(elem, str):
                data_string.append(elem)
            elif isinstance(elem, bytes):
                data_bytes = data_bytes + bytes('\r\n\r\n', 'ascii') + elem
        
        data_string.append('')
        self.buffer = bytes('\r\n\r\n'.join(data_string), 'ascii') + data_bytes

    @classmethod
    def cls_name(cls):
        return cls.__name__


class SenderThread(threading.Thread):
    _stop_t = False

    def __init__(self, client, formater, delay=1, batch_size=5):
        super(SenderThread, self).__init__()
        self.client = client
        self.delay = delay
        self.formater = formater
        self.batch_size = batch_size

    def stop(self):
        self._stop_t = True

    def run(self):
        global global_l
        counter = 0
        while self._stop_t == False:
            counter += 1
            time.sleep(round(self.delay - (self.delay/4.), 2))
            if counter % self.batch_size == 0:
                print("sending data from thread {}".format(self.client.cls_name()))
                self.client.send_data(global_l[:self.batch_size])
                global_l = global_l[self.batch_size:]
                counter = 0

            if len(global_l) > 4*self.batch_size:
                fq = FifoDiskQueue("{}.fifo.sql".format(self.client.name))
                for obj in global_l[:self.batch_size*2]:
                    print("SAVED: {}".format(obj))
                    fq.push(obj)
                global_l = global_l[self.batch_size*2:]
                fq.close()

            if (counter + 2) % self.batch_size == 0:
                fq = FifoDiskQueue("{}.fifo.sql".format(self.client.name))
                if len(fq) > 0:
                    for i in range(2):
                        for elem in fq.pull():
                            print("PULL ELEM", elem)
                            global_l.append(elem)
                        print("DB SIZE", len(fq))
                fq.close()


class GeneratorThread(threading.Thread):
    _stop_t = False

    def __init__(self, fn_data, formater, delay=1, batch_size=5):
        super(GeneratorThread, self).__init__()
        self.fn_data = fn_data
        self.delay = delay
        self.formater = formater
        self.batch_size = batch_size

    def stop(self):
        self._stop_t = True

    def run(self):
        while self._stop_t == False:
            time.sleep(self.delay)
            global_l.extend(list(self.formater.encode(self.fn_data())))
            print("generate data from thread, len: {}".format(len(global_l)))


class SyncData(object):
    def __init__(self, name, server, port=8080, delay=3, 
                delay_error_sensor=0.2, delay_error_connection=2, formater=None,
                batch_size=5):
        self.name = name
        self.delay = delay
        self.server = server
        self.port = port
        self.logger = self.logging_setup()
        self.formater = formater
        self.delay_error_connection = delay_error_connection
        self.batch_size = batch_size

    def logging_setup(self):
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        logger = logging.getLogger(self.name)
        hdlr = logging.FileHandler('/tmp/{}.log'.format(self.name))
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr)
        logger.setLevel(logging.INFO)
        return logger

    def run(self, fn_data):
        self.t = GeneratorThread(fn_data, self.formater, 
                                delay=self.delay, batch_size=self.batch_size)
        self.t.start()
        while True:
            client = Client(self.server, port=self.port, name=self.name, 
                            delay=self.delay, formater=self.formater,
                            batch_size=self.batch_size)
            asyncore.loop(timeout=1)
            time.sleep(self.delay_error_connection)
        self.t.stop()

