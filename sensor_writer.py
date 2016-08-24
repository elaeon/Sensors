import time
import platform
from queuelib import FifoDiskQueue, FifoMemoryQueue

class WriterData(object):
    def __init__(self, sensor_name, node=platform.node().replace('.', '-'), batch_size=10):
        self.node = node
        self.sensor_name = sensor_name
        self.root_name = 'sensors'
        self.batch_size = batch_size
        
    def msg_format(self, message):
        v, timestamp, name = message
        return "{}.{}.{} {} {}\n".format(self.root_name, self.node, name, v, timestamp)

    def run(self, fn, sleep=1):
        pass

    def messages_fn(self, fn):
        i = 0
        while i < self.batch_size:
            value = fn()
            timestamp = int(time.time())
            values = value if isinstance(value, tuple) else (value,)
            if len(values) > 1:
                sensors_names = ("humidity_A", "temperature_A")
            else:
                sensors_names = (self.sensor_name,)
            for value, sensor_name in zip(values, sensors_names):
                yield (value, timestamp, sensor_name)
                i += 1

    def generate_data(self, fn, sleep=1):
        for message in self.messages_fn(fn):
            yield self.msg_format(message)
            time.sleep(sleep)


class WriterMemoryData(WriterData):
    def run(self, fn, sleep=1):          
        while True: 
            self.generate_data(fn, sleep=sleep)


class WriterDiskData(WriterData):
    def __init__(self, sensor_name, node=platform.node().replace('.', '-')):
        super(WriterDiskData, self).__init__(sensor_name, node=node)
        self.database_name = "{}.fifo.sql".format(self.sensor_name)

    def run(self, fn, sleep=1):
        while True:
            queue = FifoDiskQueue(self.database_name)
            for message in self.generate_data(fn, sleep=sleep):
                queue.push(message)
            queue.close()

    def save(self, messages):
        queue = FifoDiskQueue(self.database_name)
        for message in messages:
            queue.push(message)
        queue.close()
