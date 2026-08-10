"""Unit tests for app/core/retry.py."""

import requests

from app.core import retry


def _http_error(status_code, headers=None):
    response = requests.Response()
    response.status_code = status_code
    response.headers = headers or {}
    return requests.exceptions.HTTPError(response=response)


def test_succeeds_on_first_attempt_without_sleeping(monkeypatch):
    sleeps = []
    monkeypatch.setattr(retry.time, "sleep", lambda s: sleeps.append(s))

    result = retry.retry_on_transient_error(lambda: "ok")

    assert result == "ok"
    assert sleeps == []


def test_transient_error_then_success(monkeypatch):
    monkeypatch.setattr(retry.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise requests.exceptions.ConnectionError("connection reset")
        return "recovered"

    result = retry.retry_on_transient_error(flaky, max_retries=2)

    assert result == "recovered"
    assert calls["n"] == 3


def test_always_fails_reraises_the_original_exception_type(monkeypatch):
    monkeypatch.setattr(retry.time, "sleep", lambda s: None)

    def always_times_out():
        raise requests.exceptions.Timeout("timed out")

    try:
        retry.retry_on_transient_error(always_times_out, max_retries=2)
        assert False, "expected Timeout to propagate"
    except requests.exceptions.Timeout:
        pass


def test_exhausted_retries_calls_fn_max_retries_plus_one_times(monkeypatch):
    monkeypatch.setattr(retry.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def always_fails():
        calls["n"] += 1
        raise requests.exceptions.ConnectionError("nope")

    try:
        retry.retry_on_transient_error(always_fails, max_retries=2)
    except requests.exceptions.ConnectionError:
        pass

    assert calls["n"] == 3  # initial attempt + 2 retries


def test_5xx_and_429_are_retried():
    for status in (429, 500, 503):
        calls = {"n": 0}

        def flaky(status=status):
            calls["n"] += 1
            if calls["n"] < 2:
                raise _http_error(status)
            return "ok"

        result = retry.retry_on_transient_error(flaky, max_retries=1, backoff_base_seconds=0)
        assert result == "ok"
        assert calls["n"] == 2


def test_non_retryable_4xx_fails_fast_with_zero_sleeps(monkeypatch):
    sleeps = []
    monkeypatch.setattr(retry.time, "sleep", lambda s: sleeps.append(s))
    calls = {"n": 0}

    def bad_auth():
        calls["n"] += 1
        raise _http_error(401)

    try:
        retry.retry_on_transient_error(bad_auth, max_retries=3)
        assert False, "expected HTTPError to propagate"
    except requests.exceptions.HTTPError as e:
        assert e.response.status_code == 401

    assert calls["n"] == 1  # no retries spent on a certain failure
    assert sleeps == []


def test_retry_after_header_is_honored_over_exponential_backoff(monkeypatch):
    sleeps = []
    monkeypatch.setattr(retry.time, "sleep", lambda s: sleeps.append(s))
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise _http_error(429, headers={"Retry-After": "7"})
        return "ok"

    result = retry.retry_on_transient_error(flaky, max_retries=1)

    assert result == "ok"
    assert sleeps == [7.0]
