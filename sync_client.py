#!/usr/bin/python2.7
import os
import glob
import time

import socket
import logging

from queuelib.queue import FifoSQLiteQueue

#os.system('modprobe w1-gpio')
#os.system('modprobe w1-therm')

#base_dir = '/sys/bus/w1/devices/'
#device_folder = glob.glob(base_dir+"28*")[0]
#device_folder = base_dir + "28-0115524406ff"
#device_file = device_folder + '/w1_slave'

DELAY = 1
DELAY_ERROR_SENSOR = 0.2
DELAY_ERROR_CONNECTION = 2
CARBON_SERVER = 'localhost' #'192.168.52.50'
CARBON_PORT = 8080 #2003


def logging_setup():
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger('sensor A')
    hdlr = logging.FileHandler('/home/agmartinez/sensor_A.log')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger

#logger = logging_setup()

def read_temp_raw():
    with open(device_file,'r') as f:
        return f.readlines()

def send_msg(message):
    sock = socket.socket()
    try:
        sock.connect((CARBON_SERVER, CARBON_PORT))
        sock.sendall(message)
    except socket.error:
        #logger.info("No se puede conectar a carbon {}".format(len(data)))
        return False
    except TypeError:
        return True
    else:
        #logger.info("Datos enviados.")
        return True
    finally:
        sock.close()

def send_blocks_msg(messages):
    sock = socket.socket()
    try:
        sock.connect((CARBON_SERVER, CARBON_PORT))
        for message in messages:
            sock.sendall(message)
    except socket.error:
        #logger.info("No se puede conectar a carbon {}".format(len(data)))
        return False
    #except TypeError:
    #    return True
    else:
        #logger.info("Datos enviados.")
        return True
    finally:
        sock.close()

def read_temp():
   lines = read_temp_raw()
   while lines[0].strip()[-3:] != 'YES':
      time.sleep(DELAY_ERROR_SENSOR)
      lines = read_temp_raw()
   equals_pos = lines[1].find('t=')
   if equals_pos != -1:
      temp_string = lines[1][equals_pos+2:]
      temp_c = float(temp_string) / 1000.0
      return temp_c


def prepare_to_send_msg(message):
    if send_msg(message):
        return True
    else:
        print("DELAY")
        return False

if __name__ == '__main__':
    while True:
        queue = FifoSQLiteQueue("/tmp/my_program.fifo.sql")
        print(len(queue))
        message = queue.pop()
        while len(queue) >= 20:
            messages = [message]
            while len(messages) <= 19 and message is not None:
                messages.append(queue.pop())
            if not send_blocks_msg(messages):
                for message in messages:
                    queue.push(message)
                break
        else:
            if not prepare_to_send_msg(message):
                queue.push(message)
        queue.close()
        time.sleep(DELAY)
