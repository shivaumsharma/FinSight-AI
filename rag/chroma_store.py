import chromadb
from sentence_transformers import (SentenceTransformer)

class ChromaVectorStore:
  def __init__(self,collection_name="financial_transcripts",embedding_model="all-MiniLM-L6-v2"):

    self.collection_name=(collection_name)
    self.embedding_model_name=(embedding_model)
    self.embedding_model=(SentenceTransformer(self.embedding_model_name))
    self.chroma_client=(chromadb.Client())
    self.collection=(self.chroma_client.get_or_create_collection(name=self.collection_name))

  def add_documents(self,documents):
    embeddings=(self.embedding_model.encode(documents).tolist())
    document_ids=[f"doc_{i}" for i in range(len(documents))]
    self.collection.add(documents=documents,embeddings=embeddings,ids=document_ids)

  def query_documents(self,query,n_results=3):
    query_embedding=(self.embedding_model.encode(query).tolist())
    results=(self.collection.query(query_embeddings=[query_embedding],n_results=n_results))

    return results 

  