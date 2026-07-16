from sentence_transformers import CrossEncoder

# Process-wide singleton, mirroring app/core/llm_provider.py.
# EvidenceReranker is constructed fresh inline inside RAGTool.run() on
# every single request, so without this the CrossEncoder weights get
# reloaded from disk on every query.
_shared_model = None


def _get_shared_model():
    global _shared_model
    if _shared_model is None:
        _shared_model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
    return _shared_model


class EvidenceReranker:

    def __init__(self):

        self.model = _get_shared_model()

    def rerank(self,query,chunks,top_k=5):
        if not chunks:
            return []
        pairs = [(query,chunk["text"])for chunk in chunks]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(chunks, scores),key=lambda x: x[1],reverse=True)

        return [

            chunk

            for chunk, _ in ranked[:top_k]

        ]

       