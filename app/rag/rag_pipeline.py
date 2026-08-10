"""
Production RAG Pipeline

Responsibilities
----------------
- Fetch a real disclosure for any ticker -- SEC EDGAR (SECEdgarClient)
  for US-listed tickers, NSE India (NSEFilingsClient) for .NS-listed
  ones (see ingest_company_disclosure's own docstring for the branch)
- Avoid duplicate ingestion, scoped per company
- Query vector database
- Return retrieved chunks
"""

from app.data.nse_filings_client import NSEFilingsClient
from app.data.sec_edgar_client import SECEdgarClient
from app.rag.Text_chunker import FinancialTranscriptChunker
from app.rag.chroma_store import ChromaVectorStore


class RAGPipeline:

    def __init__(self):

        self.vector_store = ChromaVectorStore()
        self.sec_client = SECEdgarClient()
        self.nse_client = NSEFilingsClient()

    # =========================================================
    # Disclosure Ingestion
    # =========================================================

    def ingest_company_disclosure(self, ticker):
        """
        Fetches this ticker's most useful recent disclosure -- for a
        US-listed ticker, an 8-K earnings-release exhibit falling back
        to a 10-Q/10-K's MD&A section (SECEdgarClient); for a .NS-listed
        ticker, NSE's own most recent "Outcome of Board Meeting"/
        "Financial Result Updates" announcement (NSEFilingsClient) --
        and ingests it into the vector store, unless this company's
        stored chunks are already from that exact filing. Everything
        downstream of the returned dict (chunking, embedding, citation
        building) is entirely source-agnostic; this branch is the
        entire SEC-vs-NSE integration point.

        Previously this skipped ingestion whenever the company had
        ANY chunks stored at all, regardless of which filing they
        came from -- fetch_company_disclosure() always re-checks the
        source for whatever filing is CURRENTLY the latest (its own
        disk cache is keyed per accession number, not "per ticker"),
        but the vector store never got refreshed to match, so a ticker
        queried once would keep serving that same filing's content
        indefinitely, even years later once new filings existed. Now
        compares accession numbers -- a unique ID for a specific
        filing, from either source -- so ingestion actually re-runs
        when a newer filing is available, and stays skipped (cheap)
        when it isn't.

        Returns (chunks, disclosure_metadata). disclosure_metadata is
        None if the relevant source has no qualifying filing for this
        ticker (e.g. a US foreign-private-issuer filing 20-F/6-K
        instead, or NSE being unreachable/having nothing to show) --
        callers should treat that as "no evidence available" rather
        than an error, the same way a missing transcript used to be
        handled.
        """

        if (ticker or "").upper().endswith(".NS"):
            disclosure = self.nse_client.fetch_company_disclosure(ticker[:-3])
        else:
            disclosure = self.sec_client.fetch_company_disclosure(ticker)

        if not disclosure:
            return [], None

        latest_accession = disclosure.get("accession_number")
        existing_accession = self.vector_store.get_ingested_accession(ticker)

        if existing_accession is not None and existing_accession == latest_accession:
            # Already have chunks from exactly this filing -- skip
            # re-chunking/re-embedding, but still return disclosure
            # metadata (cheap: fetch_company_disclosure is disk-cached)
            # so citations always have a source URL/filing date.
            return [], disclosure

        # Otherwise: either nothing is ingested yet, a newer filing is
        # available, or the existing chunks predate accession-number
        # tracking (unknown vintage, existing_accession is None but
        # chunks may still exist). Clear out anything stale before
        # ingesting fresh chunks -- unconditionally, not only when
        # existing_accession is known, because chunk_ids are built
        # from company+filing_date and would otherwise collide with
        # old chunks for the SAME filing_date, silently blocking
        # add_documents() from writing the new (accession-tagged)
        # chunks at all. delete_company_documents() on a company with
        # nothing stored is a harmless no-op.
        self.vector_store.delete_company_documents(ticker)

        chunker = FinancialTranscriptChunker(
            company=ticker,
            quarter=disclosure["filing_date"],
            accession_number=latest_accession,
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
