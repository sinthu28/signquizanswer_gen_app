import logging
import requests

class Communication:
    def __init__(self, server_url):
        self.server_url = server_url

    def send_data(self, data):
        try:
            logging.info("Sending data to server...")
            response = requests.post(self.server_url, json=data)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logging.error(f"Error sending data: {e}")
            raise e

    def receive_data(self):
        try:
            logging.info("Receiving data from server...")
            response = requests.get(self.server_url)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logging.error(f"Error receiving data: {e}")
            raise e