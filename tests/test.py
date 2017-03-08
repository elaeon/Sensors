#!/usr/bin/python2.7
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sensor_sync import SyncDataFromMemory, SyncDataFromDisk

def random_data():
    import random
    return random.uniform(0, 30)


def random_data_tuple():
    import random
    return random.randint(0, 100), random.uniform(0, 30)


if __name__ == '__main__':
    sensor_sync = SyncDataFromMemory("temperature", "127.0.0.1")
    #sensor_sync.run(random_data, batch_size=10, gen_data_every=2)
    sensor_sync.run(random_data_tuple, batch_size=10, gen_data_every=2)
    #sensor_sync = SyncDataFromDisk("temperature", "192.168.1.73")
    #sensor_sync.run()
