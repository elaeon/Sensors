#!/usr/bin/python2.7
from sensor_sync import SyncDataFromDisk

CARBON_HOST = "192.168.1.73"
SENSOR_NAME = "temperature_humidity"

if __name__ == '__main__':
    sensor_sync = SyncDataFromDisk(SENSOR_NAME, CARBON_HOST)
    sensor_sync.run()
