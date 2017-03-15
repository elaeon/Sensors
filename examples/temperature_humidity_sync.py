#!/usr/bin/python2.7
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sensor_sync import SyncDataFromMemory
from utils import get_settings, two_point_calibration

settings = get_settings(__file__)
CARBON_HOST = settings.get("server", "carbon_server")
SENSOR_NAME = "temperature_humidity"
DEVICE_NUMBER = settings.get("sensor_termopar", "device_number")
GPIO = '22'

import Adafruit_DHT

RETRIES = 10
DELAY_ERROR_SENSOR = .5

raw_low = settings.get("two_point_calibration", "raw_low")
raw_high = settings.get("two_point_calibration", "raw_high")
ref_low = settings.get("two_point_calibration", "ref_low")
ref_high = settings.get("two_point_calibration", "ref_high")

def get_humidity_temperature():
    humidity, temperature = Adafruit_DHT.read_retry(
        Adafruit_DHT.AM2302, GPIO, retries=RETRIES, delay_seconds=DELAY_ERROR_SENSOR)
    humidity = two_point_calibration(humidity, raw_low, raw_high, ref_low, ref_high)
    return humidity, temperature

if __name__ == '__main__':
    sensor_sync = SyncDataFromMemory(SENSOR_NAME, CARBON_HOST)
    sensor_sync.run(get_humidity_temperature, batch_size=10, gen_data_every=2)
