"""
FinancialTranscriptChunker

Replaces simple RecursiveCharacterTextSplitter.

This chunker:

1. Parses transcript speaker boundaries
2. Creates semantic chunks
3. Generates metadata
4. Scores importance
5. Returns structured TranscriptChunk objects
"""

import re
from dataclasses import dataclass, asdict
from typing import List


# -------------------------------------------------------
# Transcript Chunk
# -------------------------------------------------------

@dataclass
class TranscriptChunk:

    chunk_id: str
    speaker: str
    section: str
    importance: float
    text: str
    company: str = ""
    quarter: str = ""
    metadata: dict = None

    def to_dict(self):
        return asdict(self)


# -------------------------------------------------------
# Financial Transcript Chunker
# -------------------------------------------------------

class FinancialTranscriptChunker:

    def __init__(
        self,
        company="",
        quarter=""
    ):

        self.company = company
        self.quarter = quarter

    # ---------------------------------------------------

    def chunk_text(
        self,
        transcript: str
    ) -> List[TranscriptChunk]:

        speaker_blocks = self._split_by_speaker(transcript)

        chunks = []

        for idx, block in enumerate(speaker_blocks):

            speaker = block["speaker"]
            text = block["text"]

            section = self._detect_section(text)

            importance = self._importance_score(text)

            metadata = {
                "company": self.company,
                "quarter": self.quarter,
                "speaker": speaker,
                "section": section,
                "importance": importance,
            }

            chunk = TranscriptChunk(

                chunk_id=f"{self.company}_{self.quarter}_{idx}",

                speaker=speaker,

                section=section,

                importance=importance,

                text=text,

                company=self.company,

                quarter=self.quarter,

                metadata=metadata

            )

            chunks.append(chunk)

        return chunks

    # ---------------------------------------------------

    def _split_by_speaker(self, transcript):

        pattern = re.compile(

            r"(Operator|CEO|CFO|Analyst|Unknown)\s*:\s*",

            flags=re.IGNORECASE

        )

        matches = list(pattern.finditer(transcript))

        blocks = []

        if not matches:

            return [

                {

                    "speaker": "Unknown",

                    "text": transcript.strip()

                }

            ]

        for i, match in enumerate(matches):

            speaker = match.group(1)

            start = match.end()

            end = (

                matches[i + 1].start()

                if i + 1 < len(matches)

                else len(transcript)

            )

            text = transcript[start:end].strip()

            if text:

                blocks.append(

                    {

                        "speaker": speaker,

                        "text": text

                    }

                )

        return blocks

    # ---------------------------------------------------

    def _detect_section(self, text):

        t = text.lower()

        if any(

            k in t

            for k in [

                "guidance",

                "outlook",

                "forecast"

            ]

        ):

            return "Guidance"

        if any(

            k in t

            for k in [

                "revenue",

                "sales",

                "margin",

                "profit",

                "eps"

            ]

        ):

            return "Financial Performance"

        if any(

            k in t

            for k in [

                "artificial intelligence",

                "ai",

                "machine learning",

                "generative"

            ]

        ):

            return "AI"

        if any(

            k in t

            for k in [

                "cloud",

                "azure",

                "aws"

            ]

        ):

            return "Cloud"

        if any(

            k in t

            for k in [

                "capital",

                "buyback",

                "dividend"

            ]

        ):

            return "Capital Allocation"

        return "General Discussion"

    # ---------------------------------------------------

    def _importance_score(self, text):

        score = 0.5

        text = text.lower()

        important_words = [

            "guidance",

            "revenue",

            "margin",

            "eps",

            "cash flow",

            "free cash flow",

            "buyback",

            "capital allocation",

            "growth",

            "forecast",

            "ai",

            "artificial intelligence",

            "enterprise",

            "demand",

            "operating income"

        ]

        for word in important_words:

            if word in text:

                score += 0.05

        score = min(score, 1.0)

        return round(score, 2)