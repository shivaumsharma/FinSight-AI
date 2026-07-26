"""
citation_engine.py

Builds structured citations from retrieved chunks.
"""

from typing import List, Dict, Optional


class CitationEngine:

    def build(self, chunks, disclosure: Optional[dict] = None):
        """
        disclosure: the {"form", "filing_date", "source_url", ...}
        dict from SECEdgarClient.fetch_company_disclosure (see
        RAGPipeline.ingest_company_disclosure) that these chunks were
        ingested from -- every chunk in one call shares the same
        source filing, so this is looked up once, not per-chunk.
        Attached to each citation so "Retrieved Evidence" entries in
        report_data_builder.py's _references() can carry a real
        filing_date/source_url instead of url: None, matching the
        document-level "SEC Filing" reference entries right next to
        them. Optional (default None) so callers without a disclosure
        on hand (none currently, but nothing requires one) still work.
        """

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

                "text": chunk["text"][:300],

                "filing_date": (disclosure or {}).get("filing_date"),

                "source_url": (disclosure or {}).get("source_url"),

            })

        return citations
