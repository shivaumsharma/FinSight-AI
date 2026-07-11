"""
Manual smoke test for the RAG pipeline.

Fixed: the transcript path previously pointed at
"app/data/apple_q2.txt", which does not exist (the real file lives
at "app/data/transcripts/apple_q2.txt"), so this script raised
FileNotFoundError before it could test anything. Now sourced from
the same ticker_resolver mapping the rest of the app uses so the two
can never drift apart again.
"""

from app.rag.rag_pipeline import RAGPipeline
from app.core.ticker_resolver import get_transcript_path

if __name__ == "__main__":

    rag_pipeline = RAGPipeline()
    transcript_path = get_transcript_path("AAPL")
    rag_pipeline.ingest_transcript(transcript_path)

    query = "What did the management say about AI demand?"
    retrieved_chunks = rag_pipeline.query_pipeline(query=query, n_results=3)

    print("\n=== Retrieved Chunks ===\n")
    for chunk in retrieved_chunks:
        print(chunk)
        print("\n--------\n")
