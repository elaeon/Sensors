import time
import Adafruit_DHT
import platform

from queuelib import FifoDiskQueue

RETRIES = 5
DELAY_ERROR_SENSOR = .1

def get_humidity_temperature():
    humidity, temperature = Adafruit_DHT.read_retry(
        Adafruit_DHT.AM2302, '17', retries=RETRIES, delay_seconds=DELAY_ERROR_SENSOR)

    return humidity, temperature

def messages(node, size):
    i = 0
    while i < size:
        h, t = get_humidity_temperature()
        if h is None or t is None:
            continue
        timestamp = int(time.time())
        yield "system.{}.temperature_A {} {}\n".format(node, t, timestamp)
        yield "system.{}.humidity_A {} {}\n".format(node, h, timestamp)
        i += 1

if __name__ == '__main__':
    node = platform.node().replace('.', '-')
    while True:
        queue = FifoDiskQueue("temperature.fifo.sql")
        for message in messages(node, 10):
            queue.push(message)
            time.sleep(1)
        queue.close()
