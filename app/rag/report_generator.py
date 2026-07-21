import os

from huggingface_hub import hf_hub_download
from llama_cpp import Llama


class ReportGenerator:
    """
    Thin wrapper around a local llama.cpp text-completion model.

    Runs the same Qwen2.5-1.5B-Instruct model as before, now through
    llama.cpp (via llama-cpp-python) on a Q8_0-quantized GGUF instead
    of a raw HF `transformers` pipeline -- llama.cpp is purpose-built
    for fast CPU inference, and this cut the narrative call (~3300
    prompt tokens, up to 700 generated) from ~257s to ~65s on this
    machine, with no change to any caller's prompt or to generate()'s
    signature.

    Q8_0 (near-lossless 8-bit), not the smaller/faster Q4_K_M: Q4_K_M
    was tried first and is ~30% faster, but on the real MSFT narrative
    prompt (logs/narrative_debug.jsonl) it produced a 2200-char
    Executive Summary that consumed the entire generation budget and
    silently dropped the other four report sections -- confirmed via
    direct A/B against Q8_0 on the identical prompt, not assumed to be
    a one-off. Q8_0 wrote all five sections correctly on every prompt
    tested. The narrative call already dominates report latency, so
    trading some of the 4-bit speedup for reliably-complete reports is
    the right tradeoff here.

    Uses raw text completion, NOT llama.cpp's chat-formatted API --
    every caller (LLMPlanner, company_resolver, narrative_builder)
    was already prompting the old HF pipeline as freeform text
    continuation, not through a chat template (the pipeline was
    called with a plain string, not a messages list). Switching to
    chat-formatted input here would silently change how every one of
    those prompts is interpreted -- an unrelated behavior change that
    doesn't belong in a "swap the inference backend" change.

    The model is loaded lazily on first use, matching the previous
    implementation: several tools (ReportTool, LLMPlanner) create
    their own ReportGenerator-family objects, so lazy loading keeps
    construction cheap and defers the actual model load to first
    call to generate().
    """

    MODEL_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
    MODEL_FILE = "qwen2.5-1.5b-instruct-q8_0.gguf"

    # Largest observed real prompt (narrative_builder's, with a full
    # RESEARCH CONTEXT + news block) runs ~3300 tokens; 8192 leaves
    # comfortable headroom for a ticker with more retrieved evidence
    # plus the up-to-700-token generation budget. llama.cpp evicts the
    # OLDEST tokens first if the context fills, which would silently
    # cut off the DATA fields at the start of the prompt rather than
    # erroring -- sized generously to avoid that rather than to hit it
    # exactly.
    N_CTX = 8192

    def __init__(self, model_repo: str = MODEL_REPO, model_file: str = MODEL_FILE):
        self.model_repo = model_repo
        self.model_file = model_file
        self._generator = None

    @property
    def generator(self) -> Llama:
        if self._generator is None:
            model_path = hf_hub_download(repo_id=self.model_repo, filename=self.model_file)
            self._generator = Llama(
                model_path=model_path,
                n_ctx=self.N_CTX,
                n_threads=os.cpu_count(),
                verbose=False,
                # llama.cpp's repeat_penalty only looks back this many
                # tokens by default (64) -- far short of this app's
                # ~700-token generation budget. HF's repetition_penalty
                # (the old backend) scans the ENTIRE generated sequence,
                # not a fixed window. Left at the 64-token default, the
                # model "forgets" content it wrote earlier in a long
                # narrative and starts repeating whole sections
                # verbatim once they fall outside that window --
                # confirmed via direct A/B output comparison against
                # the old backend, not assumed. Set to N_CTX so the
                # penalty covers the full context, matching the old
                # backend's effectively-global behavior.
                last_n_tokens_size=self.N_CTX,
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
        was catching. This is a ceiling, not a target: generation
        still stops early at the model's own end-of-turn token for
        short outputs (e.g. the planner's one-line JSON), so this is
        safe to share between ReportTool and LLMPlanner.

        temperature=0.0 is llama.cpp's greedy decoding, matching the
        old backend's do_sample=False. repeat_penalty carries over the
        same 1.15 value from the old backend's repetition_penalty.
        """
        response = self.generator(
            prompt,
            max_tokens=max_new_tokens,
            temperature=0.0,
            repeat_penalty=1.15,
            echo=False,
        )
        return response["choices"][0]["text"]
