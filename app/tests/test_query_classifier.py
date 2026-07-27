"""
Unit tests for QueryClassifier and RetrievalPlanner (app/reasoning/) --
the rule-based intent detection that decides how a retrieval query
gets expanded before hitting the vector store.
"""

import pytest

from app.reasoning.query_classifier import QueryClassifier
from app.reasoning.retrieval_planner import RetrievalPlanner


@pytest.mark.parametrize("question,expected_intent", [
    ("Should I invest in Apple?", "investment"),
    ("What is Apple's intrinsic value?", "valuation"),
    ("What did management say on the earnings call?", "transcript"),
    ("What was Apple's revenue and margin?", "financial"),
    ("Tell me about Apple.", "general"),
])
def test_query_classifier_intents(question, expected_intent):
    assert QueryClassifier().classify(question) == expected_intent


def test_query_classifier_investment_keywords_take_priority_over_financial():
    # "worth" (investment) and "revenue" (financial) both appear --
    # investment is checked first and should win.
    intent = QueryClassifier().classify("Is Apple worth buying given its revenue growth?")
    assert intent == "investment"


def test_retrieval_planner_expands_investment_queries():
    query = RetrievalPlanner().build("Should I invest in Apple?", "investment")
    assert "guidance" in query
    assert "free cash flow" in query
    assert query.startswith("Should I invest in Apple?")


def test_retrieval_planner_passes_through_general_intent_unmodified():
    question = "Tell me about Apple."
    assert RetrievalPlanner().build(question, "general") == question
