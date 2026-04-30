import time
from collections import OrderedDict

class LLMCache:
    def __init__(self, capacity=1000, ttl_seconds=60):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.ttl = ttl_seconds

    def get(self, key):
        if key in self.cache:
            entry, timestamp = self.cache[key]
            if time.time() - timestamp <= self.ttl:
                self.cache.move_to_end(key)
                return entry
            else:
                del self.cache[key]
        return None

    def put(self, key, value):
        if key in self.cache:
            del self.cache[key]
        elif len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
        self.cache[key] = (value, time.time())
