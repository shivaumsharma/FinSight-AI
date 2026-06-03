from transformers import pipeline

class ReportGenerator:
  def __init__(self,model_name="google/flan-t5-base"):
    self.generator=pipeline("text-generation",model=model_name)
  
  def generate_response(self,query,retrieved_chunks):
    context="\n\n".join(retrieved_chunks)
    prompt=f"""You are a professional equity research analyst.Use the provided financial transcript context to answer the user question accurately.Context:{context} Question:{query}
    Answer:
        """
    
    response=self.generator(prompt,max_length=256,do_sample=False)
    generated_response=response[0]["generated_text"]

    return generated_response
