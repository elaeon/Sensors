import time
import socket
import logging

from queuelib import FifoDiskQueue
from sensor_writer import WriterDiskData, WriterMemoryData


class SyncData(object):
    def __init__(self, name, carbon_server, carbon_port=2003, delay=3, 
                delay_error_sensor=0.2, delay_error_connection=2):
        self.name = name
        self.DELAY = delay
        self.DELAY_ERROR_SENSOR = delay_error_sensor
        self.DELAY_ERROR_CONNECTION = delay_error_connection
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

    def send_msg(self, message):
        sock = socket.socket()
        try:
            sock.connect((self.CARBON_SERVER, self.CARBON_PORT))
            sock.sendall(message)
        except socket.error:
            self.logger.info("No se puede conectar a carbon {}:{}, {}".format(
                self.CARBON_SERVER, self.CARBON_PORT, message))
            return False
        except TypeError:
            return True
        else:
            return True
        finally:
            sock.close()

    def send_blocks_msg(self, messages):
        sock = socket.socket()
        tmp_msg = None
        try:
            sock.connect((self.CARBON_SERVER, self.CARBON_PORT))
            for message in messages:
                tmp_msg = message
                sock.sendall(tmp_msg)
        except socket.error:
            self.logger.info("No se puede conectar a carbon {}:{}".format(
                self.CARBON_SERVER, self.CARBON_PORT))
            return tmp_msg
        else:
            return True
        finally:
            sock.close()

    def run(self):
        pass

    def sync_failed(self, response, messages):
        sensor_writer = WriterDiskData(self.name)
        if response is None:
            sensor_writer.save(messages)
        elif response is not True:
            sensor_writer.save([response])

    def is_network_up(self):
        sock = socket.socket()
        try:
            sock.connect((self.CARBON_SERVER, self.CARBON_PORT))
        except socket.error:
            return False
        else:
            return True
        finally:
            sock.close()


class SyncDataFromDisk(SyncData):
    def run(self):
        while True:
            queue = FifoDiskQueue("{}.fifo.sql".format(self.name))
            #print(len(queue))
            if len(queue) > 0 and self.is_network_up():
                messages = queue.pull()
                response = self.send_blocks_msg(messages)
                if response is None or response is not True:
                    self.sync_failed(response, messages)
            queue.close()
            time.sleep(self.DELAY)


class SyncDataFromMemory(SyncData):
    def run(self, fn, batch_size=10, gen_data_every=1):
        queue_m = WriterMemoryData(self.name, batch_size=batch_size)
        while True:
            messages = queue_m.generate_data(fn, sleep=gen_data_every)
            response = self.send_blocks_msg(messages)
            #print(response)
            if response is None or response is not True:
                self.sync_failed(response, messages)
            
