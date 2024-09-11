import logging
from Aggregator import Aggregator

class Server:
    def __init__(self, model, clients, aggregator: Aggregator):
        self.model = model
        self.clients = clients
        self.aggregator = aggregator

    def receive_updates(self, client_updates):
        logging.info("Server: Receiving updates from clients...")
        self.aggregator.aggregate(client_updates)

    def update_model(self, global_weights):
        logging.info("Server: Updating global model...")
        self.model.set_weights(global_weights)

    def distribute_model(self):
        logging.info("Server: Distributing global model to clients...")
        global_weights = self.model.get_weights()
        for client in self.clients:
            client.update_model(global_weights)