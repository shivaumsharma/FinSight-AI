from openai import OpenAI

class ReportGenerator:
  def __init__(self,api_key):
    self.client=(OpenAI(api_key=api_key))

  def generate_response(self,query,retrieved_chunks):
    context="\n\n".join(retrieved_chunks)

    prompt=f"""
    you are an Ai financial research analyst. Use the following retrieved earnings transcript context to answer the query.Retrived Context:{context} User Query:{query} Generate a clear,professional,financially grounded response."""

    response=(self.client.chat.completions.create(model="gpt-4.1-mini",
    messages=[{"role":"system","content":"you are a professional equity research analyst"},{"role":"user","content":prompt}]))

    generate_response=(response.choices[0].message.content)
    return self.generate_response
