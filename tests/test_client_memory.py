#!/usr/bin/python2.7
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sensor_sync import SyncDataFromMemory


def random_data_tuple():
    import random
    return random.randint(0, 100), random.uniform(0, 30)


if __name__ == '__main__':
    sensor_sync = SyncDataFromMemory("temperature", "127.0.0.1")
    sensor_sync.run(random_data_tuple, batch_size=10, gen_data_every=2)

