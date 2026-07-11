from sentence_transformers import CrossEncoder


class EvidenceReranker:

    def __init__(self):

        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

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

       