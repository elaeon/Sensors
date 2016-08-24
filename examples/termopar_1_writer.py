#!/usr/bin/python2.7

CARBON_HOST = "192.168.1.73"
SENSOR_NAME = "temperature_low_one"

if __name__ == '__main__':
    sensor_sync = SyncDataFromDisk(SENSOR_NAME, CARBON_HOST)
    sensor_sync.run()
