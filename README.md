# sensors
Example
temperature_sync.py
{{{
#!/usr/bin/python2.7
from sensor_sync import SyncData

if __name__ == '__main__':
    sync_data = SyncData("temperature_low_one", ip_adress)
    sync_data.run()
}}}

temperature_writer.py
{{{
#!/usr/bin/python2.7
import os
import glob
import time

from sensor_writer import WriterData

os.system('modprobe w1-gpio')
os.system('modprobe w1-therm')

base_dir = '/sys/bus/w1/devices/'
#device_folder = glob.glob(base_dir+"28*")[0]
device_folder = base_dir+"28-01155244f3ff"
device_file = device_folder + '/w1_slave'

def read_temp_raw():
   f = open(device_file,'r')
   lines = f.readlines()
   f.close()
   return lines

def read_temp():
   lines = read_temp_raw()
   while lines[0].strip()[-3:] != 'YES':
      time.sleep(0.2)
      lines = read_temp_raw()
   equals_pos = lines[1].find('t=')
   if equals_pos != -1:
      temp_string = lines[1][equals_pos+2:]
      temp_c = float(temp_string) / 1000.0
      return temp_c

def messages_fn(node, size):
    i = 0
    while i < size:
        t = read_temp()
        timestamp = int(time.time())
        yield (t, timestamp)
        i += 1

import Adafruit_DHT

RETRIES = 10
DELAY_ERROR_SENSOR = .5

def get_humidity_temperature():
    humidity, temperature = Adafruit_DHT.read_retry(
        Adafruit_DHT.AM2302, '17', retries=RETRIES, delay_seconds=DELAY_ERROR_SENSOR)

    return humidity, temperature

#def messages_fn(node, size):
#    i = 0
#    while i < size:
#        h, t = get_humidity_temperature()
#        if h is None or t is None:
#            continue
#        timestamp = int(time.time())
#        yield (t, timestamp, "temperature_A")
#        yield (h, timestamp, "humidity_A")
#        i += 1


if __name__ == '__main__':
    sensor_writer = WriterData("temperature_low_one")
    sensor_writer.run(10, messages_fn)
}}}
