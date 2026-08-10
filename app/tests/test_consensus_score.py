"""Unit tests for app/reporting/consensus_score.py."""

from app.reporting.consensus_score import LOW_SAMPLE_THRESHOLD, build_consensus_report


def _rating(firm, rating):
    return {"firm": firm, "rating": rating, "raw_grade": rating.title(), "date": "2026-01-01"}


def test_no_institutional_ratings_returns_none():
    assert build_consensus_report("Buy", []) is None


def test_insufficient_data_rating_returns_none():
    assert build_consensus_report("Insufficient Data", [_rating("Morgan Stanley", "BUY")]) is None


def test_small_sample_flags_low_sample_size():
    assert LOW_SAMPLE_THRESHOLD == 3
    ratings = [_rating("Morgan Stanley", "BUY")]
    report = build_consensus_report("Buy", ratings)
    assert report["total_count"] == 1
    assert report["low_sample_size"] is True
    assert report["sample_size_note"] is not None
    assert "1 covering institution" in report["sample_size_note"]


def test_broad_sample_does_not_flag_low_sample_size():
    ratings = [_rating(f"Firm {i}", "BUY") for i in range(LOW_SAMPLE_THRESHOLD)]
    report = build_consensus_report("Buy", ratings)
    assert report["total_count"] == LOW_SAMPLE_THRESHOLD
    assert report["low_sample_size"] is False
    assert report["sample_size_note"] is None


def test_score_and_agreement_counts():
    ratings = [
        _rating("Morgan Stanley", "BUY"),
        _rating("Goldman Sachs", "BUY"),
        _rating("JPMorgan", "HOLD"),
        _rating("Barclays", "SELL"),
    ]
    report = build_consensus_report("Buy", ratings)
    assert report["score"] == 50
    assert report["agreeing_count"] == 2
    assert report["disagreeing_count"] == 2
    assert report["total_count"] == 4
    assert report["low_sample_size"] is False
