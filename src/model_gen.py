from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from typing import List
from flask import jsonify
from model_enc import ModelEncoder
from vector import Vector
import os
import json

class ModelGenerator:
    def __init__(self):
        self.endpoint_name = "llama-2-generator"
        self.predictor = Predictor(endpoint_name=self.endpoint_name, serializer=JSONSerializer())
        self.model_encoder = ModelEncoder()
        self.vector_store = Vector(
            api_key=os.getenv('PINECONE_API_KEY'),
            environment="gcp-starter",
            index_name="rag-aws-poc-index"
        )
        self.max_section_len = 2000
        self.separator = "\n"

    def construct_unified_context(self, contexts:List[str]) -> str:
        chosen_sections = []
        chosen_sections_len = 0
        for text in contexts:
            text = text.strip()
            chosen_sections_len+=len(text)+2
            if chosen_sections_len >self.max_section_len:
                break
            chosen_sections.append(text)
        concatenated_doc = self.separator.join(chosen_sections)
        print(f"Chunks used {len(chosen_sections)}, chunks: \n {concatenated_doc}")
        return concatenated_doc

    def create_payload(self, question, context_str) -> dict:
        #llama2 compatible prompt
        prompt_template = """Answer the following QUESTION based on the CONTEXT given. If you do not know the answer and the CONTEXT doesn't
        contain the answer truthfully say "I don't know".

        CONTEXT:
        {context}

        ANSWER:
        """
        text_input = prompt_template.replace("{context}", context_str).replace("{question}", question)
        payload = {
            "inputs":
            [
                [
                {"role": "system", "content": text_input},
                {"role": "user", "content": question},
                ]
            ],
            "parameters": {"max_new_tokens":256, "top_p": 0.9, "temperature":0.6, "return_full_text":False}
        }
        return payload
    
    def predict_with_enriched_context(self, question):
        query_vec = self.model_encoder.embed_docs(question)[0]
        print("queryvec")
        print(query_vec)
        top_k_results = self.vector_store.query(query_vec)
        print("topkresults")
        print(top_k_results)
        contexts = [match.metadata["text"] for match in top_k_results.matches]
        contexts_str = self.construct_unified_context(contexts)
        payload = self.create_payload(question, contexts_str)
        out = self.predictor.predict(payload, custom_attributes="accept_eula=true")
        decoded_string = out.decode('utf-8')
        json_response = json.loads(decoded_string)
        return jsonify({"response": json_response[0]['generation']['content'], "prompt": payload['inputs'][0][0]['content']})