from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from typing import List
import json
import numpy as np

class ModelEncoder:
    def __init__(self):
        self.endpoint_name = "minilm-embedding"
        self.model_id = "sentence-transformers/all-MiniLM-L6-v2"
        self.embedding_dimension = 384
        self.encoder = Predictor(endpoint_name=self.endpoint_name, serializer=JSONSerializer())
    
    def embed_docs(self, docs) -> List[List[float]]:
        out = self.encoder.predict({"inputs": docs})
        decoded_string = out.decode('utf-8')
        array = json.loads(decoded_string)
        embeddings = np.mean(np.array(array), axis=1)
        return embeddings.tolist()