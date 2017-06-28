import asyncore, socket, threading
import time
import logging

from queuelib import FifoDiskQueue

global_l = []


class Client(asyncore.dispatcher):

    def __init__(self, host, port=8080, name="client_data", delay=1,
                formater=None, batch_size=5, delay_error_connection=5):
        asyncore.dispatcher.__init__(self)
        self.buffer = bytes("", 'ascii')
        self.name = name
        self.delay = delay
        self.host = host
        self.port = port
        self.batch_size = batch_size
        self.formater = formater
        self.connection_error = True
        self.delay_error_connection = delay_error_connection 
        self.init_connection()

    def init_connection(self):
        self.create_socket(socket.AF_INET, socket.SOCK_STREAM)
        while True:
            try:
                self.connect((self.host, self.port))
            except socket.error as e:
                self.connection_error = True
                print("ERROR", e)
                time.sleep(self.delay_error_connection)
            else:
                self.connection_error = False
                break

    def handle_connect(self):
        print("Checking conection")

    def handle_close(self):
        print("Connection closed")
        self.close()

    def handle_read(self):
        data = self.recv(8192)
        if len(data) == 0:
            self.handle_error()
        print("RESPONSE", data)

    def readable(self):
        return True

    def handle_error(self):
        self.connection_error = True
        time.sleep(self.delay_error_connection)
        self.handle_close()
        self.init_connection()

    def writable(self):
        return (len(self.buffer) > 0)

    def handle_write(self):
        sent = self.send(self.buffer)
        self.buffer = self.buffer[sent:]
        self.handle_error()

    def send_data(self, data):
        data_bytes = bytes('', 'ascii')
        data_string = []
        for elem in data:
            if isinstance(elem, str):
                data_string.append(elem)
            elif isinstance(elem, bytes):
                data_bytes = data_bytes + bytes('\r\n\r\n', 'ascii') + elem
        
        data_string.append('')
        self.buffer += bytes('\r\n\r\n'.join(data_string), 'ascii') + data_bytes

    @classmethod
    def cls_name(cls):
        return cls.__name__


class SenderThread(threading.Thread):
    _stop_t = False

    def __init__(self, client, formater):
        super(SenderThread, self).__init__()
        self.client = client
        self.formater = formater

    def stop(self):
        self._stop_t = True

    def run(self):
        global global_l
        counter = 0
        while self._stop_t == False:
            counter += 1
            time.sleep(round(self.client.delay - (self.client.delay/4.), 2))
            if counter % self.client.batch_size == 0 and self.client.connection_error is False:
                print("sending data from thread {}:{}".format(self.client.cls_name(), id(self.client)))
                self.client.send_data(global_l[:self.client.batch_size])
                global_l = global_l[self.client.batch_size:]
                counter = 0

            if len(global_l) > 4*self.client.batch_size:
                fq = FifoDiskQueue("{}.fifo.sql".format(self.client.name))
                for obj in global_l[:self.client.batch_size*2]:
                    print("SAVED: {}".format(obj))
                    fq.push(obj)
                global_l = global_l[self.client.batch_size*2:]
                fq.close()

            if (counter + 2) % self.client.batch_size == 0:
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
        client = Client(self.server, port=self.port, name=self.name, 
                       delay=self.delay, formater=self.formater,
                       batch_size=self.batch_size, 
                       delay_error_connection=5)
        print("Init sender")        
        t_net = SenderThread(client, self.formater)
        t_net.start()
        asyncore.loop(timeout=1, use_poll=True)
        print("STOP ALL")
        self.t.stop()
        t_net.stop()
