import streamlit as st
from app.rag.rag_pipeline import (RAGPipeline)


st.set_page_config(page_title="Finsight AI",layout="wide")

st.title("Finsight AI")
st.subheader("AI-Powered Financial Research Platform")

st.sidebar.header("Transcript Settings")

transcript_path=st.sidebar.text_input("Transcript Path",value="app/data/apple_q2.txt")

query=st.text_input("Ask Financial Question",placeholder="What did management say about AI demand?")

if st.button("Run RAG Pipeline"):
  with st.spinner("Processing Transcript..."):
    rag_pipeline=(RAGPipeline())
    rag_pipeline.ingest_transcript(transcript_path)
    retrieved_chunks=(rag_pipeline.query_pipeline(query=query,n_results=3))
  st.success("Semantic retrieval completed")
  st.subheader("Retrieved Context")
  st.metric("Retrieved Chunks",len(retrieved_chunks))

  for i,chunk in enumerate(retrieved_chunks):
    st.markdown(f"""###Chunk{i+1}  {chunk}""")

st.sidebar.title("Finsight AI")
st.sidebar.markdown("AI-Powered Equity Research Platform")
col1,col2=st.columns(2)
st.markdown("---")
st.caption("Built using RAG,ChromaDB,Finbert and DCF valuation models")
st.sidebar.markdown("""
### Example Queries
- What did management say about AI demand?
- What drove margin expansion?
- How is China performing?
""")
with open(transcript_path,"r") as file:
    transcript_text=file.read()

st.text_area(
    "Transcript Preview",
    transcript_text[:1000],
    height=300
)