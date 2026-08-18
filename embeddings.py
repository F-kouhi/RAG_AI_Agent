from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
from  openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()


API_KEY = os.getenv('EMBEDDING_API_KEY')


class Embedder:
    def __init__(
        self,
        embedder_type: str = 'local',
        # مسیر پوشه محلی که مدل را در مرحله قبل در آن ذخیره کردیم
        local_model_path: str = "./local_models/multilingual-e5-small", 
        device: str = 'cpu',
        model_name_api: str = 'text-embedding-3-small',
        base_url: str = 'https://api.gapgpt.app/v1',
        api_key: str = API_KEY
    ):
        self.embedder_type = embedder_type
        if self.embedder_type == 'api':
            self.model_name = model_name_api
            self.client = OpenAI(
                base_url=base_url,
                api_key=api_key
            )
        else:
            self.device = device
            # لود مدل به صورت کاملاً آفلاین بدون اتصال به اینترنت
            self.model = SentenceTransformer(
                local_model_path, 
                device=device, 
                local_files_only=True
            )
            print(f"[Embedder] Local model loaded from {local_model_path} on {self.device}")

        

        

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



# # بخش تست آفلاین بودن مدل امبدینگ( در صورت نیاز از کامنت دربیارید)
# if __name__ == "__main__":
#     # تست لود کاملا آفلاین مدل از پوشه محلی
#     try:
#         print("در حال تست لود آفلاین مدل...")
#         embedder = Embedder(
#             embedder_type='local',
#             local_model_path="./local_models/multilingual-e5-small"
#         )
        
#         # تست تولید امبدینگ
#         test_text = ["سلام دنیا"]
#         vector = embedder.embed_text(test_text)
        
#         print("✅ تست موفقیت‌آمیز بود!")
#         print(f"ابعاد بردار امبدینگ: {vector.shape}")
        
#     except Exception as e:
#         print(f"❌ خطا در اجرای آفلاین: {e}")

