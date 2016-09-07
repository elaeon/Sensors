#!/usr/bin/python2.7
from sensor_sync import SyncDataFromDisk
from utils import get_settings

settings = get_settings(__file__)
CARBON_HOST = settings.get("server", "carbon_server")
SENSOR_NAME = "temperature_low_one"

if __name__ == '__main__':
    sensor_sync = SyncDataFromDisk(SENSOR_NAME, CARBON_HOST)
    sensor_sync.run()
