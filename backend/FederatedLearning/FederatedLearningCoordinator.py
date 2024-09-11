import logging
from FederatedLearning.Client import Client
from FederatedLearning.Server import Server
from FederatedLearning.Aggregator import Aggregator

class FederatedLearningCoordinator:
    def __init__(self, clients, server, num_rounds=10):
        self.clients = clients
        self.server = server
        self.num_rounds = num_rounds

    def start_training(self):
        logging.info("Federated Learning Coordinator: Starting training...")
        for round_num in range(self.num_rounds):
            logging.info(f"--- Round {round_num + 1} ---")
            
            client_updates = [client.send_model_update() for client in self.clients]
            
            global_weights = self.server.receive_updates(client_updates)
            
            self.server.update_model(global_weights)
            
            self.server.distribute_model()

        logging.info("Federated Learning Coordinator: Training completed.")