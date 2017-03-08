import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sensor_sync import SyncDataFromDisk


if __name__ == '__main__':
    sensor_sync = SyncDataFromDisk("temperature", "127.0.0.1")
    sensor_sync.run()
