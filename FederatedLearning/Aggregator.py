import numpy as np
import logging

class Aggregator:
    def __init__(self, secure=False):
        self.secure = secure

    def aggregate(self, client_updates):
        logging.info("Aggregator: Aggregating client updates...")
        num_clients = len(client_updates)
        avg_weights = [np.mean([client_update[layer] for client_update in client_updates], axis=0)
                       for layer in range(len(client_updates[0]))]

        return avg_weights