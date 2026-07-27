"""
Unit tests for FinancialTranscriptChunker (app/rag/Text_chunker.py) --
speaker-block splitting for earnings-call-style transcripts, the
paragraph-grouping fallback for SEC press-release text with no speaker
markers, and the section/importance heuristics that feed retrieval
metadata.
"""

from app.rag.Text_chunker import FinancialTranscriptChunker


def test_chunk_text_splits_on_speaker_markers():
    transcript = (
        "Operator: Welcome to the call.\n"
        "CEO: Revenue grew twenty percent this quarter.\n"
        "CFO: We are raising full year guidance.\n"
    )
    chunks = FinancialTranscriptChunker(company="TEST", quarter="2024-Q1").chunk_text(transcript)
    speakers = [c.speaker for c in chunks]
    assert speakers == ["Operator", "CEO", "CFO"]
    assert "Revenue grew twenty percent" in chunks[1].text


def test_chunk_text_falls_back_to_paragraphs_when_no_speaker_markers():
    # SEC 8-K exhibit text has no "Speaker:" markers at all.
    press_release = "Revenue increased significantly.\n\nManagement remains confident in the outlook."
    chunks = FinancialTranscriptChunker(company="TEST", quarter="2024-Q1").chunk_text(press_release)
    assert len(chunks) >= 1
    assert all(c.speaker for c in chunks)


def test_paragraph_fallback_tags_ceo_attribution():
    text = 'The results were strong, said Jane Doe, Apple\'s CEO, during the announcement.'
    chunks = FinancialTranscriptChunker(company="TEST", quarter="2024-Q1").chunk_text(text)
    assert chunks[0].speaker == "CEO"


def test_paragraph_fallback_defaults_to_management_without_attribution():
    text = "Revenue for the quarter increased year over year across all segments."
    chunks = FinancialTranscriptChunker(company="TEST", quarter="2024-Q1").chunk_text(text)
    assert chunks[0].speaker == "Management"


def test_many_small_paragraphs_are_grouped_up_to_the_size_cap():
    # Each paragraph individually is well under MAX_CHUNK_CHARS (800);
    # only grouping several of them together crosses the cap, which is
    # what should trigger a new chunk boundary.
    paragraph = "Sentence about results and outlook for the quarter. " * 5  # ~270 chars
    text = "\n".join([paragraph] * 6)  # ~1,620 chars total
    chunker = FinancialTranscriptChunker(company="TEST", quarter="2024-Q1")
    chunks = chunker.chunk_text(text)
    assert len(chunks) > 1
    assert all(len(c.text) <= chunker.MAX_CHUNK_CHARS for c in chunks)


def test_detect_section_classifies_guidance_language():
    chunker = FinancialTranscriptChunker()
    assert chunker._detect_section("We are raising our full year outlook and guidance.") == "Guidance"


def test_detect_section_classifies_ai_language():
    chunker = FinancialTranscriptChunker()
    assert chunker._detect_section("Generative AI demand across our cloud platform accelerated.") == "AI"


def test_detect_section_defaults_to_general_discussion():
    chunker = FinancialTranscriptChunker()
    assert chunker._detect_section("Thank you for joining us today.") == "General Discussion"


def test_importance_score_increases_with_keyword_density():
    chunker = FinancialTranscriptChunker()
    plain = chunker._importance_score("We had a normal quarter.")
    rich = chunker._importance_score(
        "Revenue growth accelerated, margin expanded, guidance and forecast improved, "
        "free cash flow and buyback increased, capital allocation and AI demand strong."
    )
    assert rich > plain
    assert rich <= 1.0


def test_chunk_metadata_carries_accession_number_for_staleness_detection():
    chunker = FinancialTranscriptChunker(company="AAPL", quarter="2024-Q1", accession_number="0001234567-24-000001")
    chunks = chunker.chunk_text("Operator: Welcome.\nCEO: Results were strong.\n")
    assert chunks[0].metadata["accession_number"] == "0001234567-24-000001"
