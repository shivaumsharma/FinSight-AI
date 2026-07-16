"""
Persistent Chroma Vector Store

Stores:

- embeddings
- documents
- metadata

Supports:

- persistent storage
- metadata filtering
- future hybrid retrieval
"""

import os
import chromadb

from sentence_transformers import SentenceTransformer

# Process-wide singleton cache, mirroring app/core/llm_provider.py.
# ChromaVectorStore is reconstructed fresh (via a new RAGPipeline())
# inside RAGTool.run() on every single request, so without this the
# embedding model gets reloaded from disk on every query.
_shared_embedding_models = {}


def _get_shared_embedding_model(name):
    if name not in _shared_embedding_models:
        _shared_embedding_models[name] = SentenceTransformer(name)
    return _shared_embedding_models[name]


class ChromaVectorStore:

    def __init__(
        self,
        collection_name="financial_transcripts",
        embedding_model="BAAI/bge-base-en-v1.5",
        persist_directory="vector_db"
    ):

        self.embedding_model = _get_shared_embedding_model(
            embedding_model
        )

        os.makedirs(persist_directory, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    #############################################################

    def add_documents(self, chunks):

        if len(chunks) == 0:
            return

        documents = [chunk.text for chunk in chunks]

        ids = [chunk.chunk_id for chunk in chunks]

        metadatas = [chunk.metadata for chunk in chunks]

        embeddings = self.embedding_model.encode(
            documents,
            normalize_embeddings=True
        ).tolist()

        # Scoped to this batch's own company rather than fetching
        # every id in the whole collection -- as more tickers get
        # ingested over a session's lifetime, a global fetch here
        # gets steadily slower for no benefit, since chunk IDs are
        # namespaced by company and can't collide across companies.
        companies = {meta.get("company") for meta in metadatas}
        existing = set()
        for company in companies:
            existing |= set(
                self.collection.get(where={"company": company})["ids"]
            )

        new_documents = []
        new_embeddings = []
        new_ids = []
        new_metadata = []

        for doc, emb, idx, meta in zip(
            documents,
            embeddings,
            ids,
            metadatas
        ):

            if idx not in existing:

                new_documents.append(doc)
                new_embeddings.append(emb)
                new_ids.append(idx)
                new_metadata.append(meta)

        if len(new_ids):

            self.collection.add(

                documents=new_documents,

                embeddings=new_embeddings,

                ids=new_ids,

                metadatas=new_metadata

            )

    #############################################################

    def query_documents(

        self,

        query,

        n_results=10,

        where=None

    ):

        query_embedding = self.embedding_model.encode(

            query,

            normalize_embeddings=True

        ).tolist()

        results = self.collection.query(

            query_embeddings=[query_embedding],

            n_results=n_results,

            where=where

        )

        return results

    #############################################################

    def count(self):

        return self.collection.count()

    #############################################################

    def company_has_documents(self, company):
        """
        Whether this specific company already has chunks stored --
        NOT whether the collection as a whole is non-empty. Using a
        global count() to decide "already ingested" caused chunk IDs
        to collide across companies whenever company/quarter weren't
        threaded through to the chunker (they previously weren't),
        silently dropping every company's chunks after the first one
        ever ingested.
        """

        existing = self.collection.get(
            where={"company": company},
            limit=1,
        )

        return len(existing.get("ids", [])) > 0

    #############################################################

    def reset(self):

        self.client.delete_collection(

            self.collection.name

        )

        self.collection = self.client.get_or_create_collection(

            self.collection.name

        )