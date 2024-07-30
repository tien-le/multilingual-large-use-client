import time
import requests
import json
import math

import numpy as np
from numpy.linalg import norm


class TensorFlowServingClient:
    def __init__(self, server_url, model_name):
        self.server_url = server_url.rstrip('/')
        self.model_name = model_name

    def _get_url(self, version=None, endpoint='predict'):
        url = f"{self.server_url}/v1/models/{self.model_name}"
        if version:
            url += f"/versions/{version}"
        if endpoint:
            if endpoint=="metadata":
                url += "/metadata"
            else:
                url += f":{endpoint}"
        return url

    def predict(self, input_texts, version=None):
        url = self._get_url(version=version, endpoint='predict')
        payload = json.dumps({"instances": input_texts})
        response = requests.post(url, data=payload, headers={"Content-Type": "application/json"})
        return response.json()

    def get_model_status(self):
        url = self._get_url(self.model_name, endpoint=None)
        response = requests.get(url)
        return response.json()

    def get_model_metadata(self):
        url = self._get_url(endpoint='metadata')
        response = requests.get(url)
        return response.json()

    def get_versioned_model_status(self, version):
        url = self._get_url(version=version, endpoint=None)
        response = requests.get(url)
        return response.json()

    def get_health_check(self, version):
        return self.get_versioned_model_status(version=version)

    def get_embeddings(self, input_texts, batch_size=32, version=None):
        embeddings = []
        num_batches = math.ceil(len(input_texts) / batch_size)

        for i in range(num_batches):
            print("-" * 36)
            print("Processing batch i=", i)
            batch_texts = input_texts[i*batch_size : (i+1)*batch_size]
            # instances = {"instances": batch_texts}
            print("batch_texts:", batch_texts)
            response = self.predict(batch_texts, version)
            # print("response:", response)
            batch_embeddings = response.get("predictions", [])
            embeddings.extend(batch_embeddings)
        return embeddings


def get_cosine_similarity(a, b):
        """Estimate the cosine similarity.
        """
        if (not a) or (not b):
            print("Cannot estimate the cosine similarity")
            return -0.1

        A = np.array(a)
        B = np.array(b)
        return np.dot(A, B)/(norm(A) * norm(B))


# Example usage
if __name__ == "__main__":
    start_time = time.time()

    model_name = "use-multilingual-large"
    model_version = "2"
    client = TensorFlowServingClient(model_name=model_name, server_url="http://localhost:8501")

    # Predict
    # input_texts = ["How you doing?", "la vie est belle"]
    input_texts = ["La vie est magnifique.", "La vie est merveilleuse.", "La vie est splendide.", "La vie est pleine de joie.", "La vie est éblouissante.", "La vie est radieuse.", "La vie est un cadeau.", "La vie est une bénédiction.", "La vie est une aventure.", "La vie est une fête.", "La vie est un enchantement.", "La vie est pleine de bonheur.", "La vie est une merveille.", "La vie est une poésie.", "La vie est un délice.", "La vie est douce.", "La vie est charmante.", "La vie est agréable.", "La vie est un rêve éveillé.", "La vie est pleine de magie.", "La vie est joyeuse.", "La vie est un plaisir.", "La vie est lumineuse.", "La vie est exaltante.", "La vie est enchanteresse.", "La vie est sereine.", "La vie est ravissante.", "La vie est épanouissante.", "La vie est enchantée.", "La vie est harmonieuse.", "La vie est passionnante.", "La vie est envoûtante.", "La vie est une chanson.", "La vie est une symphonie.", "La vie est divine.", "La vie est une aventure fascinante.", "La vie est une promenade enchantée.", "La vie est en pleine floraison.", "La vie est pleine de surprises agréables.", "La vie est comme un beau tableau.", "La vie est une comédie divine.", "La vie est une danse de joie.", "La vie est un jardin fleuri.", "La vie est pleine d'éclat.", "La vie est un spectacle merveilleux.", "La vie est une expérience sublime.", "La vie est une harmonie parfaite.", "La vie est un songe merveilleux.", "La vie est une mélodie douce.", "La vie est une cascade de plaisirs."]
    prediction = client.predict(input_texts).get("predictions", [])
    vector1 = prediction[0]
    print("Prediction:", prediction[0][:3])
    print("Duration (s):", time.time()-start_time)
    print("-" * 72)

    # Get model status
    model_status = client.get_model_status()
    print("Model Status:", model_status)
    print("-" * 72)

    # Get model metadata
    model_metadata = client.get_model_metadata()
    print("Model Metadata:", model_metadata)
    print("-" * 72)

    # Get versioned model status
    versioned_model_status = client.get_versioned_model_status(model_version)
    print("Versioned Model Status:", versioned_model_status)
    print("-" * 72)

    # Health check
    health_check = client.get_health_check(model_version)
    print("Health Check:", health_check)
    print("-" * 72)


    # Batch Processing
    # The get_embeddings method divides the input texts into batches of a specified size (batch_size).
    # For each batch, it formats the texts as required by the model, sends a request to TensorFlow Serving,
    # and collects the embeddings from the response.
    start_time = time.time()
    input_texts = ["La vie est magnifique.", "La vie est merveilleuse.", "La vie est splendide.", "La vie est pleine de joie.", "La vie est éblouissante.", "La vie est radieuse.", "La vie est un cadeau.", "La vie est une bénédiction.", "La vie est une aventure.", "La vie est une fête.", "La vie est un enchantement.", "La vie est pleine de bonheur.", "La vie est une merveille.", "La vie est une poésie.", "La vie est un délice.", "La vie est douce.", "La vie est charmante.", "La vie est agréable.", "La vie est un rêve éveillé.", "La vie est pleine de magie.", "La vie est joyeuse.", "La vie est un plaisir.", "La vie est lumineuse.", "La vie est exaltante.", "La vie est enchanteresse.", "La vie est sereine.", "La vie est ravissante.", "La vie est épanouissante.", "La vie est enchantée.", "La vie est harmonieuse.", "La vie est passionnante.", "La vie est envoûtante.", "La vie est une chanson.", "La vie est une symphonie.", "La vie est divine.", "La vie est une aventure fascinante.", "La vie est une promenade enchantée.", "La vie est en pleine floraison.", "La vie est pleine de surprises agréables.", "La vie est comme un beau tableau.", "La vie est une comédie divine.", "La vie est une danse de joie.", "La vie est un jardin fleuri.", "La vie est pleine d'éclat.", "La vie est un spectacle merveilleux.", "La vie est une expérience sublime.", "La vie est une harmonie parfaite.", "La vie est un songe merveilleux.", "La vie est une mélodie douce.", "La vie est une cascade de plaisirs."]

    # Get embeddings
    embeddings = client.get_embeddings(input_texts=input_texts, batch_size=8)
    vector2 = embeddings[0]
    print("Embeddings:", embeddings[0][:3])
    print("Duration (s):", time.time()-start_time)
    print("-" * 72)

    cosine_sim = get_cosine_similarity(prediction[0], embeddings[0])
    print("Cosine similarity:", cosine_sim)
