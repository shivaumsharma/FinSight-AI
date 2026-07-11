import pandas as pd 
from transformers import pipeline

class FinBERTSentimentAnalyzer:
  def __init__(self,model_name="ProsusAI/finbert"):
    self.model_name=(model_name)
    self.sentiment_pipeline=(pipeline("text-classification",model=self.model_name))
  
  def analyze_sentiment(self,text):
    sentiment_result=(self.sentiment_pipeline(text))
    sentiment_label=(sentiment_result[0]["label"])
    sentiment_score=(sentiment_result[0]["score"])
    return {"label":sentiment_label,"score":sentiment_score}
