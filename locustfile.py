from locust import HttpUser, task, between

class QuickstartUser(HttpUser):
    wait_time = between(1, 2)  # Simulated users will wait 1-2 seconds between tasks

    @task
    def predict_sentiment(self):
        self.client.post("/predict", json={"text": "I love this!"})
