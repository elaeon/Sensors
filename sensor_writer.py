import time
import platform
from queuelib import FifoDiskQueue

class WriterData(object):
    def __init__(self, sensor_name):
        self.node = platform.node().replace('.', '-')
        self.sensor_name = sensor_name
        self.database_name = "{}.fifo.sql".format(self.sensor_name)

    def msg_format(self, message):
        if len(message) == 2:
            v, timestamp = message
            name = self.sensor_name
        elif len(message) == 3:
            v, timestamp, name = message
        return "system.{}.{} {} {}\n".format(self.node, name, v, timestamp)

    def run(self, num_messages_second, messages_fn, sleep=1):
        while True:
            queue = FifoDiskQueue(self.database_name)
            for message in messages_fn(self.node, num_messages_second):
                queue.push(self.msg_format(message))
                time.sleep(sleep)
            queue.close()
