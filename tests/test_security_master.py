"""ISIN -> ticker resolution -- no real network; OpenFIGI's HTTP call is faked."""

import requests

import security_master


class _FakeResponse:
    def __init__(self, status_code, json_data, headers=None):
        self.status_code = status_code
        self._json_data = json_data
        self.headers = headers or {}

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeSession:
    """Records every POST and returns canned responses in order (or via a handler fn)."""

    def __init__(self, responses=None, handler=None):
        self._responses = list(responses or [])
        self._handler = handler
        self.calls = []

    def post(self, url, json, headers, timeout):
        self.calls.append({"url": url, "json": json, "headers": headers})
        if self._handler:
            return self._handler(json)
        return self._responses.pop(0)


def test_us_listing_resolves_with_high_confidence():
    session = _FakeSession([
        _FakeResponse(200, [{"data": [{"ticker": "GOOGL", "name": "Alphabet Inc", "exchCode": "US", "securityType": "Common Stock"}]}]),
    ])
    result = security_master.resolve_isins(["US02079K3059"], provider=security_master.OpenFIGIProvider(session))
    assert result["US02079K3059"]["ticker"] == "GOOGL"
    assert result["US02079K3059"]["confidence"] == "high"


def test_non_us_listing_resolves_with_best_effort_confidence():
    session = _FakeSession([
        _FakeResponse(200, [{"data": [{"ticker": "SU", "name": "Schneider Electric SE", "exchCode": "FP", "securityType": "Common Stock"}]}]),
    ])
    result = security_master.resolve_isins(["FR0000121972"], provider=security_master.OpenFIGIProvider(session))
    assert result["FR0000121972"]["ticker"] == "SU"
    assert result["FR0000121972"]["confidence"] == "best_effort"


def test_us_candidate_is_preferred_over_non_us_when_both_present():
    session = _FakeSession([
        _FakeResponse(200, [{"data": [
            {"ticker": "SNYNF", "name": "Sony", "exchCode": "US"},  # US OTC listing, should win
            {"ticker": "6758", "name": "Sony", "exchCode": "JT"},
        ]}]),
    ])
    result = security_master.resolve_isins(["JP3435000009"], provider=security_master.OpenFIGIProvider(session))
    assert result["JP3435000009"]["ticker"] == "SNYNF"
    assert result["JP3435000009"]["confidence"] == "high"


def test_not_found_resolves_to_none():
    session = _FakeSession([_FakeResponse(200, [{"warning": "No identifier found."}])])
    result = security_master.resolve_isins(["XX0000000000"], provider=security_master.OpenFIGIProvider(session))
    assert result["XX0000000000"] is None


def test_resolution_is_cached_second_call_makes_no_request():
    session = _FakeSession([
        _FakeResponse(200, [{"data": [{"ticker": "GOOGL", "name": "Alphabet", "exchCode": "US"}]}]),
    ])
    provider = security_master.OpenFIGIProvider(session)

    first = security_master.resolve_isins(["US02079K3059"], provider=provider)
    second = security_master.resolve_isins(["US02079K3059"], provider=provider)

    assert first == second
    assert len(session.calls) == 1  # not called again for the cached ISIN


def test_batching_splits_more_than_ten_isins_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENFIGI_API_KEY", raising=False)
    isins = [f"US{i:010d}" for i in range(14)]

    def handler(jobs):
        return _FakeResponse(200, [{"data": [{"ticker": f"T{i}", "exchCode": "US"}]} for i in range(len(jobs))])

    session = _FakeSession(handler=handler)
    result = security_master.resolve_isins(isins, provider=security_master.OpenFIGIProvider(session))

    assert len(session.calls) == 2  # 10 + 4
    assert len(session.calls[0]["json"]) == 10
    assert len(session.calls[1]["json"]) == 4
    assert len(result) == 14


def test_rate_limit_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr(security_master.time, "sleep", lambda seconds: None)  # don't actually wait in tests
    session = _FakeSession([
        _FakeResponse(429, None, headers={"retry-after": "1"}),
        _FakeResponse(200, [{"data": [{"ticker": "GOOGL", "exchCode": "US"}]}]),
    ])
    result = security_master.resolve_isins(["US02079K3059"], provider=security_master.OpenFIGIProvider(session))
    assert result["US02079K3059"]["ticker"] == "GOOGL"
    assert len(session.calls) == 2


def test_network_failure_resolves_batch_to_none_without_crashing():
    class _RaisingSession:
        def post(self, *args, **kwargs):
            raise requests.exceptions.ConnectionError("no network")

    result = security_master.resolve_isins(["US02079K3059"], provider=security_master.OpenFIGIProvider(_RaisingSession()))
    assert result["US02079K3059"] is None
