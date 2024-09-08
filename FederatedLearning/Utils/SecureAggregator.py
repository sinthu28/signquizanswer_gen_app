import numpy as np
import logging

class SecureAggregator:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    def add_noise(self, data):
        noise = np.random.laplace(0, 1/self.epsilon, data.shape)
        return data + noise

    def secure_aggregate(self, client_updates):
        logging.info("SecureAggregator: Adding noise for differential privacy...")
        noisy_updates = [self.add_noise(update) for update in client_updates]
        return np.mean(noisy_updates, axis=0)