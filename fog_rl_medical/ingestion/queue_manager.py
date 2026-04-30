import queue

class QueueManager:
    def __init__(self):
        # 4 Priority lanes (1 to 4)
        self.queues = {
            1: queue.Queue(),
            2: queue.Queue(),
            3: queue.Queue(),
            4: queue.Queue()
        }

    def enqueue(self, task, priority):
        if priority in self.queues:
            self.queues[priority].put(task)

    def dequeue(self):
        for p in sorted(self.queues.keys()):
            if not self.queues[p].empty():
                return self.queues[p].get()
        return None

    def get_queue_depths(self):
        return [self.queues[p].qsize() for p in sorted(self.queues.keys())]
