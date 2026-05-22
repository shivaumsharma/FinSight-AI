from app.rag.rag_pipeline import(RAGPipeline)
from app.rag.report_generator import(ReportGenerator)

if __name__=="__main__":
  rag_pipeline=(RAGPipeline())
  transcript_path=("data/apple_q2.txt")
  rag_pipeline.ingest_transcript(transcript_path)
  query=("What did the management say about AI demand?")
  retrived_chunks=(rag_pipeline.query_pipeline(query=query,n_results=3))

  print("\n===Retrived Chunks===\n")

  for chunk in retrived_chunks:
    print(chunk)
    print("\n--------\n")

    report_generator=(ReportGenerator(api_key="YOUR_OPENAI_API_KEY"))

    generated_response=(report_generator.generate_response(query=query,retrived_chunks=retrived_chunks))

    print("\n===Generated Response===\n")
    print(generated_response)