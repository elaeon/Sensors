#!/usr/bin/python2.7
from sensor_sync import SyncDataFromMemory

CARBON_HOST = "192.168.1.73"
SENSOR_NAME = "temperature_humidity"
DEVICE_NUMBER = "28-01155244f3ff"

import Adafruit_DHT

RETRIES = 10
DELAY_ERROR_SENSOR = .5

def get_humidity_temperature():
    humidity, temperature = Adafruit_DHT.read_retry(
        Adafruit_DHT.AM2302, '17', retries=RETRIES, delay_seconds=DELAY_ERROR_SENSOR)
    return humidity, temperature

if __name__ == '__main__':
    sensor_sync = SyncDataFromMemory(SENSOR_NAME, CARBON_HOST)
    sensor_sync.run(get_humidity_temperature, batch_size=10, gen_data_every=2)