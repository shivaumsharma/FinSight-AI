from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd

class TextChunker:
  def __init__(self,chunk_size=500,chunk_overlap=50):
    self.chunk_size=(chunk_overlap)
    self.text_splitter=(RecursiveCharacterTextSplitter(chunk_size=self.chunk_size,chunk_overlap=self.chunk_overlap))
    def chunk_text(self,text):
      chunks=(self.text_splitter.split_text(text))
      return chunks 