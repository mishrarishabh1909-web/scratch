import time

class TaskDispatcher:
    def __init__(self):
        self.active_tasks = {}

    def dispatch(self, task, node_id):
        # Mock YAFS DES event dispatch
        self.active_tasks[task.task_id] = {
            'task': task,
            'node_id': node_id,
            'dispatch_time': time.time(),
            'status': 'running'
        }

    def complete_task(self, task_id):
        if task_id in self.active_tasks:
            self.active_tasks[task_id]['status'] = 'completed'
            self.active_tasks[task_id]['completion_time'] = time.time()
