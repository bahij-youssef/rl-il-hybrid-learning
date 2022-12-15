from collections import deque
import random

# Represents the experience memory of an agent
class ReplayMemory():

    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)

    def add(self, exp):
        self.buffer.append(exp)

    def sample(self, batch_size):
        return random.sample(self.buffer, min(len(self.buffer),batch_size))