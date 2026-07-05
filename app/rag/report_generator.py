from transformers import pipeline

class ReportGenerator:
  def __init__(self,model_name="google/flan-t5-base"):
    self.generator=pipeline("text2text-generation",model=model_name)
  
  def generate_response(self,query,retrieved_chunks):
    context="\n\n".join(retrieved_chunks)
    prompt = f"""
You are a senior equity research analyst.

Context:
{context}

Question:
{query}

Write a concise answer in 4-6 sentences using ONLY the context.

Answer:
"""
    print("=" * 80)
    print(prompt)
    print("=" * 80)
    
    response=self.generator(prompt,max_new_tokens=256,do_sample=False,repetition_penalty=2.0,no_repeat_ngram_size=3)
    generated_response=response[0]["generated_text"]

    return generated_response
