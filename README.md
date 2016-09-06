# sensors
Sensors, is a library for send datatime series over the network, for example data generated from temperature sensors.

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
    sensor_sync = SyncDataFromMemory(sensor_name, ip_adress, port)
    #send data with a chunk size of 10, each element of this chunk is generated every 2 seconds.
    sensor_sync.run(random_data, batch_size=10, gen_data_every=2)
```
Run the script in background.
In a separated file, build another sync, this sync will send data previously stored, derivated from network errors conections. The sensor name must be the same as SyncDataFromMemory

```python
#!/usr/bin/python2.7
if __name__ == '__main__':
    sensor_sync = SyncDataFromDisk(sensor_name, ip_adress, port)
    sensor_sync.run()
```
