#!/usr/bin/python3
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import RPi.GPIO as io
io.setmode(io.BCM)

from sensor_sync import SyncData
from formater import CarbonFormat
from utils import get_settings

settings = get_settings(__file__)
CARBON_HOST = settings.get("server", "carbon_server")
CARBON_PORT = int(settings.get("server", "carbon_port"))
SENSOR_NAME = "puerta"

door_pin = 17 #GPIO17. osea pin11
io.setup(door_pin, io.IN, pull_up_down=io.PUD_UP)  # activate input with PullUp

def check_door():
    return 1 if io.input(door_pin) else 0


if __name__ == '__main__':
    formater = CarbonFormat(SENSOR_NAME)
    sensor_sync = SyncData(SENSOR_NAME, CARBON_HOST, port=CARBON_PORT, formater=formater, delay=2, 
                            batch_size=10, delay_error_connection=10)
    sensor_sync.run(check_door)
