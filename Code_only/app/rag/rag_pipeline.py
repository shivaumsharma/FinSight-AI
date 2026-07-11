"""
Production RAG Pipeline

Responsibilities
----------------
✓ Ingest transcript
✓ Avoid duplicate ingestion
✓ Query vector database
✓ Return retrieved chunks
"""

from app.rag.transcript_loader import TranscriptLoader
from app.rag.Text_chunker import FinancialTranscriptChunker
from app.rag.chroma_store import ChromaVectorStore


class RAGPipeline:

    def __init__(self):

        self.vector_store = ChromaVectorStore()

    # =========================================================
    # Transcript Ingestion
    # =========================================================

    def ingest_transcript(
        self,
        transcript_path,
        company="",
        quarter=""
    ):

        loader = TranscriptLoader(transcript_path)

        transcript = loader.load_transcript()

        chunker = FinancialTranscriptChunker(
            company=company,
            quarter=quarter
        )

        chunks = chunker.chunk_text(transcript)

        existing_docs = self.vector_store.count()

        if existing_docs == 0:

            print("Creating Vector Database...")

            self.vector_store.add_documents(chunks)

        else:

            print("Existing Vector Store Found")
            print("Skipping Transcript Ingestion")

        return chunks

    # =========================================================
    # Retrieval
    # =========================================================

    def query_pipeline(
        self,
        query,
        n_results=20,
        where=None
    ):

        results = self.vector_store.query_documents(

            query=query,

            n_results=n_results,

            where=where

        )

        retrieved = []

        documents = results.get("documents", [[]])[0]

        metadatas = results.get("metadatas", [[]])[0]

        for doc, meta in zip(documents, metadatas):

            retrieved.append({

                "text": doc,

                "metadata": meta

            })

        return retrieved