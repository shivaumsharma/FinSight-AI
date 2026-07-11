# from .base_tool import BaseTool
# from app.rag.rag_pipeline import RAGPipeline


# class RAGTool(BaseTool):

#     name = "rag_tool"
#     description = "Retrieves relevant transcript chunks."

#     def run(self, **kwargs):

#         query = kwargs.get("question")
#         transcript_path = kwargs.get("transcript_path")

#         pipeline = RAGPipeline()

#         pipeline.ingest_transcript(transcript_path)

#         chunks = pipeline.query_pipeline(query)

#         return {
#             "retrieved_chunks": chunks
#         }

from .base_tool import BaseTool
from app.rag.rag_pipeline import RAGPipeline


class RAGTool(BaseTool):

    name = "rag_tool"
    description = "Retrieves relevant transcript chunks."

    def run(self, **kwargs):

        print("\n===== RAG TOOL =====")
        print(kwargs)

        query = kwargs.get("question")
        transcript_path = kwargs.get("transcript_path")

        print("query =", repr(query))
        print("transcript_path =", repr(transcript_path))

        pipeline = RAGPipeline()

        pipeline.ingest_transcript(transcript_path)

        chunks = pipeline.query_pipeline(query)

        return {
            "retrieved_chunks": chunks
        }