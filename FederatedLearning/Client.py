import logging

class Client:
    def __init__(self, client_id, model, data, epochs=1, batch_size=32):
        self.client_id = client_id
        self.model = model
        self.data = data
        self.epochs = epochs
        self.batch_size = batch_size

    def train_local_model(self):
        logging.info(f"Client {self.client_id}: Starting local training...")
        X_train, y_train = self.data
        self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size)
        logging.info(f"Client {self.client_id}: Local training completed.")
        return self.model.get_weights()

    def send_model_update(self):
        weights = self.train_local_model()
        return weights