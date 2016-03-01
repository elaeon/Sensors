#!/usr/bin/python2.7
import os
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

if __name__ == '__main__':
    sensor_writer = WriterData("temperature")
    sensor_writer.run(10, messages_fn)
