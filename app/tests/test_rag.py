"""
Manual smoke test for the RAG pipeline.

Ingests a real SEC disclosure for AAPL (via SECEdgarClient, live
network call) rather than a hand-authored transcript file.
"""

from app.rag.rag_pipeline import RAGPipeline

if __name__ == "__main__":

    rag_pipeline = RAGPipeline()
    chunks, disclosure = rag_pipeline.ingest_company_disclosure("AAPL")

    if not disclosure:
        print("No SEC disclosure found for AAPL.")
    else:
        print(f"Ingested {disclosure['form']} filed {disclosure['filing_date']}")

        query = "What did the management say about AI demand?"
        retrieved_chunks = rag_pipeline.query_pipeline(query=query, ticker="AAPL", n_results=3)

        print("\n=== Retrieved Chunks ===\n")
        for chunk in retrieved_chunks:
            print(chunk)
            print("\n--------\n")
