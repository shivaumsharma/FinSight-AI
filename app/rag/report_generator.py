from transformers import pipeline
import time 


class ReportGenerator:
    """
    Thin wrapper around a local HF text-generation pipeline.

    The model is now loaded lazily on first use instead of in
    __init__. Several tools (ReportTool, LLMPlanner) create their own
    ReportGenerator-family objects; lazy loading means constructing
    one is cheap and the actual multi-second model load only happens
    once, on first call to generate().
    """

    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct"):
        self.model_name = model_name
        self._generator = None

    @property
    def generator(self):
        if self._generator is None:
            self._generator = pipeline(
                "text-generation",
                model=self.model_name,
                return_full_text=False
            )
        return self._generator

    def generate(self, prompt: str, max_new_tokens: int = 700) -> str:
        """
        max_new_tokens defaults to 700, not 250.

        250 was cutting the report off mid-section: the system prompt
        instructs a ~350-word report across 5 headed sections, and if
        the model is not yet using an efficient tokenizer, 250 tokens
        (a good deal less than 350 words) as GENERATION budget nearly
        guarantees Financial Outlook / Investment Recommendation never
        get written at all -- which is exactly what completeness_score
        was catching. This is a ceiling, not a target: the pipeline
        still stops early at the model's own end-of-turn token for
        short outputs (e.g. the planner's one-line JSON), so this is
        safe to share between ReportTool and LLMPlanner.
        """
        start = time.perf_counter()
        response = self.generator(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.15,
            return_full_text=False
        )

        return response[0]["generated_text"]