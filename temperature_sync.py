#!/usr/bin/python2.7
from sensor_sync import SyncData

if __name__ == '__main__':
    sync_data = SyncData("temperature", ip_adress)
    sync_data.run()
