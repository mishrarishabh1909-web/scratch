import uuid
import time

class StreamReceiver:
    def __init__(self):
        pass

    def receive(self, task):
        task['arrival_time'] = time.time()
        task['ingestion_id'] = str(uuid.uuid4())
        return task
