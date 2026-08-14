from embeddings import Embedder
from pdf_to_txt import DockumentProcessor
from pathlib import Path
import numpy as np
import json
from typing import List, Dict
import re
from hazm import sent_tokenize  

DOC_PATH = Path("Data/")  
CHUNKS_PATH = Path("Data/chunks.json")
METADATA_PATH = Path("Data/metadata.json")
EMBEDDINGS_PATH = Path("Data/embeddings.npy")


class EmbeddingBuilder:
    """
    Offline document preprocessing for Persian RAG:
    - Clean text
    - Chunk into overlapping sentences
    - Embed
    - Save chunks, embeddings, metadata
    """

    def __init__(self, chunk_size: int = 700, overlap_sentences: int = 2, batch_size: int = 64):
        self.chunk_size = chunk_size
        self.overlap_sentences = overlap_sentences
        self.batch_size = batch_size
        self.embedder = Embedder()
    
   



   

    def chunk_doc(self, pages: List[Dict],doc_id):
        """
        pages: [{"page_number": int, "text": str}, ...]
        """
        chunks = []
        metadata = []
        doc_char_offset = 0  # global char index across pages


      

        for page in pages:
            page_number = page["page_number"]
            
            sentences = sent_tokenize(page['text'])
            current_chunk = []
            char_start = doc_char_offset

            for _ ,  sent in enumerate(sentences):
                current_chunk.append(sent)
                current_len = sum(len(s) for s in current_chunk) + len(current_chunk)



                if current_len >= self.chunk_size:
                    chunk_text = " ".join(current_chunk)
                    chunks.append(chunk_text)

                    metadata.append({
                        "doc_id": doc_id, 
                        "char_start": char_start,
                        "char_end": char_start + len(chunk_text),
                        "sentences": current_chunk.copy(),
                        "page_number": page_number
                    })

                    # overlap: keep last N sentences
                    overlap_sents = current_chunk[-self.overlap_sentences:] if self.overlap_sentences > 0 else []
                    char_start += len(" ".join(current_chunk[:-self.overlap_sentences])) if self.overlap_sentences > 0 else len(chunk_text)
                    current_chunk = overlap_sents

            # final chunk for page
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)
                metadata.append({
                    "doc_id": doc_id, 
                    "char_start": char_start,
                    "char_end": char_start + len(chunk_text),
                    "sentences": current_chunk.copy(),
                    "page_number": page_number
                })

            doc_char_offset += len(page['text']) + 1  

        return chunks, metadata

    def embed_chunks(self, chunks: List[str]) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i+self.batch_size]
            emb = self.embedder.embed_text_api(batch)
            all_embeddings.append(emb)
        return np.vstack(all_embeddings)

    from pathlib import Path


    def build(self):

        print("[Building] ....")
        all_chunks = []
        all_metadata = []
        all_embeddings = []

        for pdf_path in DOC_PATH.glob("*.pdf"):
            print(f"[Build] Processing {pdf_path.name}")
            doc_id = pdf_path.stem

            processed_path = Path(f"Data/processed_{doc_id}.json")

            self.pdf_processor = DockumentProcessor(
                pdf_path=pdf_path,
                output_path=processed_path
            )
            self.pdf_processor.pdf_to_json()

            pages = json.loads(processed_path.read_text(encoding="utf-8"))

            chunks, metadata = self.chunk_doc(pages, doc_id)
            embeddings = self.embed_chunks(chunks)

            all_chunks.extend(chunks)
            all_metadata.extend(metadata)
            all_embeddings.append(embeddings)

        np.save(EMBEDDINGS_PATH, np.vstack(all_embeddings))

        with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False)

        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(all_metadata, f, ensure_ascii=False)




if __name__ == '__main__':
    embd = EmbeddingBuilder()

    embd.build()