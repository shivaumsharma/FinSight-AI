from transformers import pipeline


class ReportGenerator:

    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct"):

        self.generator = pipeline(
            "text-generation",
            model=model_name,
            return_full_text=False
        )

    def generate(
        self,
        prompt: str
    ) -> str:

        response = self.generator(
            prompt,
            max_new_tokens=220,
            do_sample=False,
            temperature=0.2,
            repetition_penalty=1.15,
            return_full_text=False
        )

        return response[0]["generated_text"]