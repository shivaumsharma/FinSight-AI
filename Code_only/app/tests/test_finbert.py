from app.nlp.finbert import FinBERT


model = FinBERT()

text = """
Revenue increased 32%.
Margins expanded.
Management raised guidance.
"""

print(model.analyze(text))