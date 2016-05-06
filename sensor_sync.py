import time
import socket
import logging

from queuelib import FifoDiskQueue


class SyncData(object):
    def __init__(self, name, carbon_server, carbon_port=2003, delay=1, 
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
            self.logger.info("No se puede conectar a carbon {}:{}, {}".format(self.CARBON_SERVER, self.CARBON_PORT, message))
            return False
        except TypeError:
            return True
        else:
            return True
        finally:
            sock.close()

    def send_blocks_msg(self, messages):
        sock = socket.socket()
        try:
            sock.connect((self.CARBON_SERVER, self.CARBON_PORT))
            for message in messages:
                sock.sendall(message)
        except socket.error:
            self.logger.info("No se puede conectar a carbon {}:{}, {}".format(self.CARBON_SERVER, self.CARBON_PORT, message))
            return False
        else:
            return True
        finally:
            sock.close()

    def run(self):
        while True:
            queue = FifoDiskQueue("{}.fifo.sql".format(self.name))
            print(len(queue))
            message = queue.pop()
            while len(queue) >= 20:
                messages = [message]
                while len(messages) <= 19 and message is not None:
                    messages.append(queue.pop())
                if not self.send_blocks_msg(messages):
                    for message in messages:
                        queue.push(message)
                    break
            else:
                if not self.send_msg(message):
                    queue.push(message)
            queue.close()
            time.sleep(self.DELAY)

    def test(self):
        i = 0
        while i < 10:
            print(self.send_msg(""))
            time.sleep(self.DELAY)
            i += 1
