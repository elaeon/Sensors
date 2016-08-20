#!/usr/bin/python2.7

from sensor_sync import SyncDataFromMemory


def read_temp():
    import random
    return random.uniform(0, 30)


if __name__ == '__main__':
    sensor_sync = SyncDataFromMemory("temperature", "192.168.1.73")
    sensor_sync.run(read_temp, batch_size=10, gen_data_every=2)
