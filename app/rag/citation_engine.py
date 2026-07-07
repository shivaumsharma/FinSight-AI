"""
citation_engine.py

Builds structured citations from retrieved chunks.
"""

from typing import List, Dict


class CitationEngine:

    def build(self, chunks):

        citations = []

        for i, chunk in enumerate(chunks):

            citations.append({

                "id": chunk["metadata"].get(

                    "chunk_id",

                    f"Chunk {i+1}"

                ),

                "speaker": chunk["metadata"].get(

                    "speaker",

                    "Unknown"

                ),

                "section": chunk["metadata"].get(

                    "section",

                    "Unknown"

                ),

                "importance": chunk["metadata"].get(

                    "importance",

                    0.5

                ),

                "text": chunk["text"][:300]

            })

        return citations
