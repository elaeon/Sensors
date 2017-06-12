import os
import sqlite3
from time import sleep
from threading import get_ident


class FifoMemoryQueue(object):
    def __init__(self):
        self._db = []

    def push(self, item):
        if not isinstance(item, bytes):
            raise TypeError('Unsupported type: {}'.format(type(item).__name__))
        self._db.append(item)

    def pop(self):
        self._db.pop()

    def pull(self, batch_size=10):
        return self._db[:batch_size]

    def close(self):
        self._db = []

    def __len__(self):
        return len(self._db)

    
class FifoDiskQueue(object):

    _create = (
            'CREATE TABLE IF NOT EXISTS fifoqueue ' 
            '('
            '  id INTEGER PRIMARY KEY AUTOINCREMENT,'
            '  item BLOB'
            ')'
            )
    _size = 'SELECT COUNT(*) FROM fifoqueue'
    _iterate = 'SELECT id, item FROM queue'
    _push = 'INSERT INTO fifoqueue (item) VALUES (?)'
    _write_lock = 'BEGIN IMMEDIATE'
    _pull = 'SELECT id, item FROM fifoqueue ORDER BY id LIMIT 1'
    _del = 'DELETE FROM fifoqueue WHERE id = ?'
    _peek = (
            'SELECT item FROM queue '
            'ORDER BY id LIMIT 1'
            )

    def __init__(self, path):
        self.path = os.path.abspath(path)
        self._connection_cache = {}
        with self._get_conn() as conn:
            conn.execute(self._create)

    def __len__(self):
        with self._get_conn() as conn:
            l = next(conn.execute(self._size))[0]
        return l

    def __iter__(self):
        with self._get_conn() as conn:
            for id, obj_buffer in conn.execute(self._iterate):
                yield loads(str(obj_buffer))

    def _get_conn(self):
        id = get_ident()
        if id not in self._connection_cache:
            self._connection_cache[id] = sqlite3.Connection(self.path, 
                    timeout=60)
            self._connection_cache[id].text_factory = bytes
        return self._connection_cache[id]

    def push(self, obj):
        obj_buffer = bytes(obj, 'ascii')
        with self._get_conn() as conn:
            conn.execute(self._push, (obj_buffer,)) 

    def pull(self, sleep_wait=True):
        keep_pooling = True
        wait = 0.1
        max_wait = 2
        tries = 0
        with self._get_conn() as conn:
            while keep_pooling:
                conn.execute(self._write_lock)
                cursor = conn.execute(self._pull)
                try:
                    id, obj_buffer = cursor.next()
                    yield obj_buffer
                    conn.execute(self._del, (id,))
                    keep_pooling = False
                except StopIteration:
                    conn.commit() # unlock the database
                    if not sleep_wait:
                        keep_pooling = False
                        continue
                    tries += 1
                    sleep(wait)
                    wait = min(max_wait, tries/10 + wait)

    def peek(self):
        with self._get_conn() as conn:
            cursor = conn.execute(self._peek)
            try:
                return loads(str(cursor.next()[0]))
            except StopIteration:
                return None

    def close(self):
        for k in self._connection_cache.keys():
            self._connection_cache[k].close()
        self._connection_cache = {}

    
class LifoDiskQueue(FifoDiskQueue):
    _create = (
        'CREATE TABLE IF NOT EXISTS lifoqueue '
        '(id INTEGER PRIMARY KEY AUTOINCREMENT, item BLOB)'
    )
    _pop = 'SELECT id, item FROM lifoqueue ORDER BY id DESC LIMIT 1'
    _size = 'SELECT COUNT(*) FROM lifoqueue'
    _push = 'INSERT INTO lifoqueue (item) VALUES (?)'
    _del = 'DELETE FROM lifoqueue WHERE id = ?'
