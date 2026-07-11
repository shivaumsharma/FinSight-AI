import os
import pandas as pd 

class TranscriptLoader:
  def __init__(self,transcript_path):
    self.transcript_path=(transcript_path)
  def load_transcript(self):
    with open(self.transcript_path,"r",encoding="utf-8") as file:
      transcript_text=(file.read())
      return transcript_text
    
