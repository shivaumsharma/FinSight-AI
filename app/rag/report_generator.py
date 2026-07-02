from transformers import pipeline

class ReportGenerator:
  def __init__(self,model_name="google/flan-t5-base"):
    self.generator=pipeline("text2text-generation",model=model_name)
  
  def generate_response(self,query,retrieved_chunks):
    context="\n\n".join(retrieved_chunks)
    prompt=f"""You are an expert equity research analyst.
    Answer ONLY using the information contained in the transcript.
    If the transcript does not contain enough information, reply:
    "The transcript does not contain enough information to answer this question:      {query}
    Answer:
        """
    
    response=self.generator(prompt,max_new_tokens=256,do_sample=False)
    generated_response=response[0]["generated_text"]

    return generated_response
