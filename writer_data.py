import time
import os
from queuelib.queue import FifoSQLiteQueue

def get_temperature():
    import random
    return random.random() * 10

def messages(size):
    i = 0
    while i < size:
        timestamp = time.strftime("%Y-%M-%d-%H:%M:%S")
        yield "'system.{}.temperature_A {} {}'".format("test", get_temperature(), timestamp)
        i += 1

if __name__ == '__main__':
    while True:
        queue = FifoSQLiteQueue("/tmp/my_program.fifo.sql")
        for message in messages(10):
            queue.push(message)
            time.sleep(1)
        queue.close()
