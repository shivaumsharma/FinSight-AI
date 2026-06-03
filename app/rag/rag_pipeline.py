from app.rag.transcript_loader import (TranscriptLoader)
from app.rag.Text_chunker import (TextChunker)
from app.rag.chroma_store import (ChromaVectorStore)

class RAGPipeline:
  def __init__(self): 
    self.chunker=(TextChunker())
    self.vector_store=(ChromaVectorStore())

  def ingest_transcript(self,transcript_path):
    transcript_loader=TranscriptLoader(transcript_path)
    transcript_text=(transcript_loader.load_transcript())
    chunks=(self.chunker.chunk_text(transcript_text))
    self.vector_store.add_documents(chunks)

    return chunks
  
  def query_pipeline(self,query,n_results=3):
    results=(self.vector_store.query_documents(query=query,n_results=n_results))
    retrieved_chunks=(results["documents"][0])
    return retrieved_chunks
