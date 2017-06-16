import time
import platform
from queuelib import FifoDiskQueue, FifoMemoryQueue

class BaseFormat(object):
    def __init__(self, sensor_name, node=platform.node().replace('.', '-'), 
                root_name='sensors', alternate_names=None):
        self.node = node
        self.sensor_name = sensor_name
        self.root_name = root_name
        self.alternate_names = alternate_names
        
    def msg_format(self, message):
        pass

    def encode(self, value):
        pass

    def run(self, fn, sleep=1):
        pass

    def message_fn(self, value):
        timestamp = int(time.time())
        values = value if isinstance(value, tuple) else (value,)
        if len(values) > 1:
            sensors_names = self.alternate_names
        else:
            sensors_names = (self.sensor_name,)
        for value, sensor_name in zip(values, sensors_names):
            yield (value, timestamp, sensor_name)


class CarbonFormat(BaseFormat):
    def msg_format(self, messages):
        for msg in messages:
            v, timestamp, name = msg
            yield "{}.{}.{} {} {}\n".format(self.root_name, self.node, name, v, timestamp)

    def encode(self, value):
        return self.msg_format(self.message_fn(value))


class SequenceTestFormat(BaseFormat):
    def msg_format(self, messages):
        for msg in messages:
            v, timestamp, name = msg
            yield str(v)

    def encode(self, value):
        return self.msg_format(self.message_fn(value))

