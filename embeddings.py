from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
from  openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()


API_KEY = os.getenv('EMBEDDING_API_KEY')


class Embedder:


    def __init__(self,embedder_type:str='api', model_name: str = "intfloat/multilingual-e5-small",device: str = 'cpu', model_name_api:str='text-embedding-3-small',base_url:str='https://api.gapgpt.app/v1',api_key:str=API_KEY):
        self.embedder_type=embedder_type
        if self.embedder_type!='api':
            self.device = device
            self.model= SentenceTransformer(model_name,device=device)
            print(f"[Embedder] Model loaded on {self.device}")

        self.model_name = model_name_api
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )

        

    def embed_text(self, texts: List[str])-> np.ndarray:
        """
        Embed a list of text chunks.
        Returns a numpy array of shape (num_chunks, embedding_dim)
        
        """
         
        embedded_text =self.model.encode(
            ["passage: " + t for t in texts],
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False
            )

        return embedded_text
    

    def embed_query(self,query: str) -> np.ndarray:
        """
        Embed a single query string.
        Returns a 1D numpy array.
        
        """
        embedded_query= self.model.encode(["query: " + query],convert_to_numpy=True, normalize_embeddings=True)[0]
        return embedded_query
    



    

    def embed_text_api(self,texts: List[str])->np.ndarray:
        
        prefixed_texts = [f"passage: {t}" for t in texts]
        
        response = self.client.embeddings.create(
            model=self.model_name,
            input=prefixed_texts
        )
        
        # Extract embeddings from the response
        embeddings = [item.embedding for item in response.data]
        
        # Convert to numpy array
        return np.array(embeddings)


    def embed_query_api(self, query: str) -> np.ndarray:
        """
        Embed a single query string.
        Returns a 1D numpy array.
        """
        prefixed_query = f"query: {query}"
        
        response = self.client.embeddings.create(
            model=self.model_name,
            input=[prefixed_query]  
        )
        
        embedding = response.data[0].embedding
        
        return np.array(embedding)