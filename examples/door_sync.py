#!/usr/bin/python2.7
import RPi.GPIO as io
io.setmode(io.BCM)

from sensor_sync import SyncDataFromMemory
from utils import get_settings

settings = get_settings(__file__)
CARBON_HOST = settings.get("server", "carbon_server")
SENSOR_NAME = "puerta"

door_pin = 17 #GPIO17. osea pin11
io.setup(door_pin, io.IN, pull_up_down=io.PUD_UP)  # activate input with PullUp

def check_door():
    return 1 if io.input(door_pin) else 0


if __name__ == '__main__':
    sensor_sync = SyncDataFromMemory(SENSOR_NAME, CARBON_HOST)
    sensor_sync.run(check_door, batch_size=10, gen_data_every=2)
