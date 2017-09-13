# sensors
Sensors, is a library for send datatime series from a device to a server over the network or saved into the device.

Example.
First define a function where the data is generated.
```python
#!/usr/bin/python2.7
#This funtion return random numbers
def random_data():
    import random
    return random.uniform(0, 30)
```
Then, call a sync
```python
if __name__ == '__main__':
    formater = CarbonFormat(SENSOR_NAME)
    sensor_sync = SyncData(SENSOR_NAME, CARBON_HOST, port=CARBON_PORT, formater=formater, delay=2, 
                            batch_size=10, delay_error_connection=10)
    sensor_sync.run(read_temp)
```
