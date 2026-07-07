from transformers import pipeline
import time 


class ReportGenerator:

    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct"):

        self.generator = pipeline(
            "text-generation",
            model=model_name,
            return_full_text=False
        )

    def generate(self,prompt: str)-> str:
        start = time.perf_counter()
        response = self.generator(
            prompt,
            max_new_tokens=250,
            do_sample=False,
            repetition_penalty=1.2,
            return_full_text=False
        )

        return response[0]["generated_text"]