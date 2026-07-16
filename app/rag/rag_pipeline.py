"""
Production RAG Pipeline

Responsibilities
----------------
- Fetch a real SEC disclosure for any ticker (via SECEdgarClient)
- Avoid duplicate ingestion, scoped per company
- Query vector database
- Return retrieved chunks
"""

from app.data.sec_edgar_client import SECEdgarClient
from app.rag.Text_chunker import FinancialTranscriptChunker
from app.rag.chroma_store import ChromaVectorStore


class RAGPipeline:

    def __init__(self):

        self.vector_store = ChromaVectorStore()
        self.sec_client = SECEdgarClient()

    # =========================================================
    # Disclosure Ingestion
    # =========================================================

    def ingest_company_disclosure(self, ticker):
        """
        Fetches this ticker's most useful recent SEC disclosure (an
        8-K earnings-release exhibit, falling back to a 10-Q/10-K's
        MD&A section) and ingests it into the vector store, unless
        this company already has chunks stored.

        Returns (chunks, disclosure_metadata). disclosure_metadata is
        None if SEC has no CIK / no qualifying filing for this ticker
        (e.g. some foreign private issuers file 20-F/6-K instead) --
        callers should treat that as "no evidence available" rather
        than an error, the same way a missing transcript used to be
        handled.
        """

        disclosure = self.sec_client.fetch_company_disclosure(ticker)

        if not disclosure:
            return [], None

        if self.vector_store.company_has_documents(ticker):
            # Already ingested in an earlier query -- skip re-chunking
            # and re-embedding, but still return disclosure metadata
            # (cheap: fetch_company_disclosure is disk-cached) so
            # citations always have a source URL/filing date, not
            # just on the first query for this ticker.
            return [], disclosure

        chunker = FinancialTranscriptChunker(
            company=ticker,
            quarter=disclosure["filing_date"],
        )

        chunks = chunker.chunk_text(disclosure["text"])

        self.vector_store.add_documents(chunks)

        return chunks, disclosure

    # =========================================================
    # Retrieval
    # =========================================================

    def query_pipeline(
        self,
        query,
        ticker,
        n_results=20,
    ):

        results = self.vector_store.query_documents(
            query=query,
            n_results=n_results,
            where={"company": ticker},
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
