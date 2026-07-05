import chromadb
from sentence_transformers import SentenceTransformer


class RAGService:
    """
    Singleton service that owns the embedding model
    and Chroma collection.
    """

    def __init__(
        self,
        collection_name="financial_transcripts",
        embedding_model="all-MiniLM-L6-v2",
    ):

        self.embedding_model = SentenceTransformer(embedding_model)

        self.client = chromadb.Client()

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    def embed(self, texts):

        if isinstance(texts, str):
            texts = [texts]

        return self.embedding_model.encode(texts).tolist()