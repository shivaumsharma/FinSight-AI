"""
Manual smoke test for FinBERT sentiment scoring -- not a pytest test
(no assertions, downloads/runs the real model). Guarded behind
__main__ so pytest can safely import this module during collection
without triggering a model load as a side effect (matching the same
pattern already used in test_rag.py).
"""

from app.nlp.finbert import FinBERT

if __name__ == "__main__":
    model = FinBERT()

    text = """
    Revenue increased 32%.
    Margins expanded.
    Management raised guidance.
    """

    print(model.analyze(text))