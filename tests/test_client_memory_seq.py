#!/usr/bin/python3
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sensor_sync import SyncData
from formater import CarbonFormat


def generator():
    i = 0
    while True:
        yield i
        i = i + 1

N = generator()
def natural_numbers():
    return next(N)

if __name__ == '__main__':
    formater = SequenceTestFormat("seq")
    sensor_sync = SyncData("seq", "127.0.0.1", formater=formater, delay=1, 
                            batch_size=10, delay_error_connection=10)
    sensor_sync.run(natural_numbers)
