import pinecone
import time
from typing import List
from tqdm.auto import tqdm
import os
from model_enc import ModelEncoder

class Vector:
    def __init__(self, api_key, environment, index_name):
        #os.getenv('PINECONE_API_KEY') 
        self.api_key = api_key
        self.environment = environment # "gcp-starter"
        self.index_name = index_name #"rag-aws-poc-index"
        self.k = 4
        self.embedding_dimension = 384 #must match ModelEncoder embeding dimension
        self.batch_size = 2
        self.model_encoder = ModelEncoder()
        print(api_key)
        pinecone.init(api_key=api_key, environment=environment)
        self.index = pinecone.Index(index_name)

    
    def query(self, query_vec):
        top_k_results = self.index.query(query_vec, top_k=self.k, include_metadata=True)
        return top_k_results
        
    def create_vector_store(self):
        if self.index_name in pinecone.list_indexes():
            pinecone.delete_index(self.index_name)
        pinecone.create_index(name=self.index_name, dimension=self.embedding_dimension, metric='cosine')
        while not pinecone.describe_index(self.index_name).status['ready']:
            time.sleep(1)
        return "Index updated"
    
    def embeed_chunks(self, chunks):
        print("Embedded each chunk and save it to vector store")
        for i in tqdm(range(0, len(chunks), self.batch_size)):
            i_end = min(i+ self.batch_size, len(chunks))
            ids = [str(x) for x in range(i, i_end)]
            metas = [{"text": text, "category": metadata} for text, metadata in zip(chunks['page_content'][i:i_end], chunks['metadata'][i:i_end])]
            texts = [text for text in chunks['page_content'][i:i_end].tolist()]
            embeddings = self.model_encoder.embed_docs(texts)
            records = zip(ids, embeddings, metas)
            self.index.upsert(vectors=records)

        print("Chunks embedded to vector store Succesfully")    
        print(self.index.describe_index_stats())

    def chunks_to_vector_store(self, chunks):
        self.create_vector_store()
        self.embeed_chunks(chunks)
        print("Index updated")