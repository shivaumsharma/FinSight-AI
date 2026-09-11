"""
Unit tests for RAGPipeline (app/rag/rag_pipeline.py) -- ingestion
routing, accession-based skip/re-ingest logic, retrieval shape, and the
per-ticker locking added as a fix for a real ingestion race (see
rag_pipeline.py's own comment on _lock_for_ticker).

Previously this file was a manual __main__-gated smoke script with
zero real assertions (needs live SEC EDGAR network access and the full
embedding-model stack) -- it read as coverage in a file listing but
wasn't, and was excluded from CI for exactly that reason. Rewritten to
mock at the same three seams RAGPipeline.__init__ constructs
(ChromaVectorStore, SECEdgarClient, NSEFilingsClient) plus
FinancialTranscriptChunker, so this now runs in milliseconds with no
network access and belongs in CI -- no longer --ignore'd in
.github/workflows/tests.yml.
"""

import importlib.util
import sys
import threading
import time
import types

# app/rag/chroma_store.py imports chromadb and sentence_transformers at
# MODULE level (transitively pulled in by `from app.rag.rag_pipeline
# import RAGPipeline` below), even though this file never constructs a
# real ChromaVectorStore -- every test here replaces it with a fake via
# _build_pipeline. Those two packages are deliberately NOT in this
# project's lightweight CI dependency list (see tests.yml's own
# extensive comment on why: keeping torch/chromadb out keeps CI fast
# and network-independent), so importing them for real here would
# either break CI collection or force CI to grow multi-GB heavier for
# a module this file already fully mocks around. Install minimal fake
# stand-ins ONLY if the real packages genuinely aren't installed (a
# normal dev machine with the full stack still gets the real ones,
# unchanged) -- just enough surface for chroma_store.py's own import
# statements to succeed; the actual objects are never touched, since
# ChromaVectorStore itself is mocked out below.
if importlib.util.find_spec("chromadb") is None:
    sys.modules["chromadb"] = types.ModuleType("chromadb")
if importlib.util.find_spec("sentence_transformers") is None:
    fake_st = types.ModuleType("sentence_transformers")
    fake_st.SentenceTransformer = object
    sys.modules["sentence_transformers"] = fake_st

from app.rag import rag_pipeline as rag_pipeline_module
from app.rag.rag_pipeline import RAGPipeline


class _FakeChunk:
    def __init__(self, text, chunk_id, metadata):
        self.text = text
        self.chunk_id = chunk_id
        self.metadata = metadata


class _FakeChunker:
    """Stands in for FinancialTranscriptChunker -- one chunk per call,
    carrying enough metadata (company, accession_number) for
    _FakeVectorStore's get_ingested_accession/add_documents to behave
    like the real ChromaVectorStore does against real chunk metadata."""

    def __init__(self, company, quarter, accession_number):
        self.company = company
        self.accession_number = accession_number

    def chunk_text(self, text):
        return [_FakeChunk(
            text=text,
            chunk_id=f"{self.company}_{self.accession_number}_0",
            metadata={"company": self.company, "accession_number": self.accession_number},
        )]


class _FakeVectorStore:
    """Implements exactly the 4 methods RAGPipeline calls on
    self.vector_store, backed by a plain dict instead of a real Chroma
    collection + embedding model. delete_delay lets a test widen the
    delete-then-add race window on purpose; call_log records call
    order/count for the concurrency test below."""

    def __init__(self, delete_delay=0.0, call_log=None):
        self._by_ticker = {}
        self._lock = threading.Lock()  # protects the fake's own dict, not what's under test
        self.delete_delay = delete_delay
        self.call_log = call_log if call_log is not None else []

    def get_ingested_accession(self, ticker):
        with self._lock:
            chunks = self._by_ticker.get(ticker)
        if not chunks:
            return None
        return chunks[0]["metadata"].get("accession_number")

    def delete_company_documents(self, ticker):
        self.call_log.append("delete")
        if self.delete_delay:
            time.sleep(self.delete_delay)
        with self._lock:
            self._by_ticker.pop(ticker, None)

    def add_documents(self, chunks):
        self.call_log.append("add")
        with self._lock:
            for chunk in chunks:
                ticker = chunk.metadata["company"]
                self._by_ticker.setdefault(ticker, []).append(
                    {"text": chunk.text, "metadata": chunk.metadata}
                )

    def query_documents(self, query, n_results, where):
        ticker = where["company"]
        with self._lock:
            chunks = list(self._by_ticker.get(ticker, []))[:n_results]
        return {
            "documents": [[c["text"] for c in chunks]],
            "metadatas": [[c["metadata"] for c in chunks]],
        }


class _FakeDisclosureClient:
    def __init__(self, disclosure):
        self._disclosure = disclosure
        self.calls = []

    def fetch_company_disclosure(self, ticker):
        self.calls.append(ticker)
        return self._disclosure


def _disclosure(accession="ACC-1", text="Some filing text.", filing_date="2026-01-01"):
    return {
        "text": text,
        "form": "8-K",
        "filing_date": filing_date,
        "accession_number": accession,
        "source_url": "https://example.com/filing.htm",
    }


def _build_pipeline(monkeypatch, store, sec_client=None, nse_client=None):
    """Patches the three classes RAGPipeline.__init__ constructs, then
    builds a real RAGPipeline -- its __init__ picks up the fakes via
    normal name resolution in the rag_pipeline module namespace."""
    monkeypatch.setattr(rag_pipeline_module, "ChromaVectorStore", lambda: store)
    monkeypatch.setattr(rag_pipeline_module, "SECEdgarClient", lambda: sec_client or _FakeDisclosureClient(None))
    monkeypatch.setattr(rag_pipeline_module, "NSEFilingsClient", lambda: nse_client or _FakeDisclosureClient(None))
    monkeypatch.setattr(rag_pipeline_module, "FinancialTranscriptChunker", _FakeChunker)
    return RAGPipeline()


def test_ingest_routes_a_dot_ns_ticker_to_the_nse_client(monkeypatch):
    sec_client = _FakeDisclosureClient(None)
    nse_client = _FakeDisclosureClient(_disclosure())
    pipeline = _build_pipeline(monkeypatch, _FakeVectorStore(), sec_client, nse_client)

    pipeline.ingest_company_disclosure("RELIANCE.NS")

    assert nse_client.calls == ["RELIANCE"]  # .NS suffix stripped before calling
    assert sec_client.calls == []


def test_ingest_routes_a_plain_ticker_to_the_sec_client(monkeypatch):
    sec_client = _FakeDisclosureClient(_disclosure())
    nse_client = _FakeDisclosureClient(None)
    pipeline = _build_pipeline(monkeypatch, _FakeVectorStore(), sec_client, nse_client)

    pipeline.ingest_company_disclosure("AAPL")

    assert sec_client.calls == ["AAPL"]
    assert nse_client.calls == []


def test_ingest_returns_empty_when_no_disclosure_is_found(monkeypatch):
    pipeline = _build_pipeline(monkeypatch, _FakeVectorStore(), _FakeDisclosureClient(None))
    chunks, disclosure = pipeline.ingest_company_disclosure("AAPL")
    assert chunks == []
    assert disclosure is None


def test_ingest_writes_chunks_on_first_ingestion(monkeypatch):
    store = _FakeVectorStore()
    pipeline = _build_pipeline(monkeypatch, store, _FakeDisclosureClient(_disclosure(accession="ACC-1")))

    chunks, disclosure = pipeline.ingest_company_disclosure("AAPL")

    assert len(chunks) == 1
    assert disclosure["accession_number"] == "ACC-1"
    assert store.get_ingested_accession("AAPL") == "ACC-1"


def test_ingest_skips_reingest_when_accession_already_matches(monkeypatch):
    store = _FakeVectorStore()
    sec_client = _FakeDisclosureClient(_disclosure(accession="ACC-1"))
    pipeline = _build_pipeline(monkeypatch, store, sec_client)

    pipeline.ingest_company_disclosure("AAPL")
    store.call_log.clear()
    chunks, disclosure = pipeline.ingest_company_disclosure("AAPL")

    assert chunks == []  # no re-chunking
    assert disclosure["accession_number"] == "ACC-1"  # metadata still returned
    assert store.call_log == []  # neither delete nor add ran the second time


def test_ingest_reingests_when_a_newer_accession_is_available(monkeypatch):
    store = _FakeVectorStore()
    sec_client = _FakeDisclosureClient(_disclosure(accession="ACC-1", text="old text"))
    pipeline = _build_pipeline(monkeypatch, store, sec_client)
    pipeline.ingest_company_disclosure("AAPL")

    sec_client._disclosure = _disclosure(accession="ACC-2", text="new text")
    chunks, disclosure = pipeline.ingest_company_disclosure("AAPL")

    assert len(chunks) == 1
    assert chunks[0].text == "new text"
    assert store.get_ingested_accession("AAPL") == "ACC-2"


def test_query_pipeline_maps_documents_and_metadata_into_dicts(monkeypatch):
    store = _FakeVectorStore()
    pipeline = _build_pipeline(monkeypatch, store, _FakeDisclosureClient(_disclosure(text="evidence text")))
    pipeline.ingest_company_disclosure("AAPL")

    retrieved = pipeline.query_pipeline(query="What did management say?", ticker="AAPL", n_results=5)

    assert retrieved == [{"text": "evidence text", "metadata": {"company": "AAPL", "accession_number": "ACC-1"}}]


def test_concurrent_ingestion_for_the_same_new_filing_only_deletes_and_adds_once(monkeypatch):
    """Regression test for the RAG ingestion atomicity fix: two
    RAGPipeline instances (mirroring two separate requests, each
    constructing their own RAGPipeline but sharing the same underlying
    store) both deciding a re-ingest is needed for the SAME new filing
    at the same time. Before the per-ticker lock, both callers ran
    delete-then-add independently and could interleave -- this asserts
    the lock collapses that into exactly one delete/add cycle: the
    second caller's lock-protected accession check sees the first
    caller's write already landed and skips re-deleting/re-adding
    entirely, rather than racing it.

    delete_delay widens the window between delete() and add() on
    purpose -- without the lock, that's exactly the gap a second
    thread's own delete() could land in.
    """
    call_log = []
    store = _FakeVectorStore(delete_delay=0.05, call_log=call_log)
    store.add_documents([_FakeChunk("stale text", "AAPL_ACC-OLD_0", {"company": "AAPL", "accession_number": "ACC-OLD"})])
    call_log.clear()

    sec_client = _FakeDisclosureClient(_disclosure(accession="ACC-NEW", text="fresh text"))
    pipeline_a = _build_pipeline(monkeypatch, store, sec_client)
    pipeline_b = _build_pipeline(monkeypatch, store, sec_client)

    barrier = threading.Barrier(2)

    def run(pipeline):
        barrier.wait()  # both threads reach ingest_company_disclosure at the same instant
        pipeline.ingest_company_disclosure("AAPL")

    threads = [threading.Thread(target=run, args=(p,)) for p in (pipeline_a, pipeline_b)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert call_log.count("delete") == 1
    assert call_log.count("add") == 1
    assert store.get_ingested_accession("AAPL") == "ACC-NEW"
