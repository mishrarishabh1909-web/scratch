class CapacityBufferAgent:
    def __init__(self):
        self.buffer_pool = 200.0
        
    def request_burst(self, amount):
        if self.buffer_pool >= amount:
            self.buffer_pool -= amount
            return True
        return False
        
    def replenish(self, amount):
        self.buffer_pool = min(200.0, self.buffer_pool + amount)
        
