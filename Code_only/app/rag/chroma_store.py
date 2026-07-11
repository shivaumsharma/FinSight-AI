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


class ChromaVectorStore:

    def __init__(
        self,
        collection_name="financial_transcripts",
        embedding_model="BAAI/bge-base-en-v1.5",
        persist_directory="vector_db"
    ):

        self.embedding_model = SentenceTransformer(
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

        existing = set(
            self.collection.get()["ids"]
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

    def reset(self):

        self.client.delete_collection(

            self.collection.name

        )

        self.collection = self.client.get_or_create_collection(

            self.collection.name

        )