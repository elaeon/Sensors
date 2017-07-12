#!/usr/bin/python3
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sensor_sync import SyncData
from formater import CarbonFormat
from utils import get_settings
import os

settings = get_settings(__file__)
CARBON_HOST = settings.get("server", "carbon_server")
CARBON_PORT = int(settings.get("server", "carbon_port"))
SENSOR_NAME = "temperature_low_one"
DEVICE_NUMBER = settings.get("sensor_termopar", "device_number")

os.system('modprobe w1-gpio')
os.system('modprobe w1-therm')

base_dir = '/sys/bus/w1/devices/'
device_folder = base_dir + DEVICE_NUMBER
device_file = device_folder + '/w1_slave'

def read_temp_raw():
    with open(device_file,'r') as f:
        lines = f.readlines()
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


if __name__ == '__main__':
    formater = CarbonFormat(SENSOR_NAME)
    sensor_sync = SyncData(SENSOR_NAME, CARBON_HOST, port=CARBON_PORT, formater=formater, delay=2, 
                            batch_size=10, delay_error_connection=10)
    sensor_sync.run(read_temp)
