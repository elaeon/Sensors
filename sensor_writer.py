import time
import platform
from queuelib import FifoDiskQueue

class WriterData(object):
    def __init__(self, sensor_name):
        self.node = platform.node().replace('.', '-')
        self.sensor_name = sensor_name

    def msg_format(self, v, timestamp):
        return "system.{}.{} {} {}\n".format(self.node, self.sensor_name, v, timestamp)

    def run(self, database_name, num_messages_second, messages_fn, sleep=1):
        while True:
            queue = FifoDiskQueue(database_name)
            for t, timestamp in messages_fn(self.node, num_messages_second):
                queue.push(self.msg_format(t, timestamp))
                time.sleep(sleep)
            queue.close()
