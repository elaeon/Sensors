#!/usr/bin/python2.7
from sensor_sync import SyncData

if __name__ == '__main__':
    sync_data = SyncData("temperature_low_one", "192.168.52.50")
    sync_data.run()
