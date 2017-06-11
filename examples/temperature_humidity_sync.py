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

raw_low_h = settings.get("two_point_calibration_1", "raw_low")
raw_high_h = settings.get("two_point_calibration_1", "raw_high")
ref_low_h = settings.get("two_point_calibration_1", "ref_low")
ref_high_h = settings.get("two_point_calibration_1", "ref_high")

raw_low_t = settings.get("two_point_calibration_2", "raw_low")
raw_high_t = settings.get("two_point_calibration_2", "raw_high")
ref_low_t = settings.get("two_point_calibration_2", "ref_low")
ref_high_t = settings.get("two_point_calibration_2", "ref_high")

def get_humidity_temperature():
    humidity, temperature = Adafruit_DHT.read_retry(
        Adafruit_DHT.AM2302, GPIO, retries=RETRIES, delay_seconds=DELAY_ERROR_SENSOR)
    humidity = two_point_calibration(humidity, raw_low_h, raw_high_h, ref_low_h, ref_high_h)
    temperature = two_point_calibration(temperature, raw_low_t, raw_high_t, ref_low_t, ref_high_t)
    return humidity, temperature

if __name__ == '__main__':
    sensor_sync = SyncDataFromMemory(SENSOR_NAME, CARBON_HOST)
    sensor_sync.run(get_humidity_temperature, batch_size=10, gen_data_every=2)
