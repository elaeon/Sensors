#!/usr/bin/python3
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sensor_sync import SyncData
from formater import CarbonFormat


def random_data_tuple():
    import random
    return random.randint(0, 100)#, random.uniform(0, 30)


if __name__ == '__main__':
    formater = CarbonFormat("temperature", alternate_names=("humidity_A", "temperature_A"))
    sensor_sync = SyncData("temperature", "127.0.0.1", formater=formater, delay=1, 
                            batch_size=10, delay_error_connection=10)
    sensor_sync.run(random_data_tuple)
