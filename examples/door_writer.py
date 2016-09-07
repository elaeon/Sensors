#!/usr/bin/python2.7
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sensor_sync import SyncDataFromDisk
from utils import get_settings

settings = get_settings(__file__)
CARBON_HOST = settings.get("server", "carbon_server")
SENSOR_NAME = "puerta"

if __name__ == '__main__':
    sensor_sync = SyncDataFromDisk(SENSOR_NAME, CARBON_HOST)
    sensor_sync.run()
