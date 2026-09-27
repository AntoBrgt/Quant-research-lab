"""Form 4 insider features -- synthetic XML / data sets, no network."""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

import config
import insider_form4 as f4


@pytest.fixture(autouse=True)
def isolated_form4(tmp_path, monkeypatch):
    monkeypatch.setattr(f4, "FORM4_DIR", tmp_path / "form4")
    monkeypatch.setattr(config, "RAW_DATA_DIR", tmp_path / "raw")
    return tmp_path


def _owner(cik, director=False, officer=False, ten_pct=False, flag_style="1"):
    t, f = ("true", "false") if flag_style == "bool" else ("1", "0")
    return (
        f"<reportingOwner><reportingOwnerId><rptOwnerCik>{cik}</rptOwnerCik><rptOwnerName>X</rptOwnerName></reportingOwnerId>"
        f"<reportingOwnerRelationship><isDirector>{t if director else f}</isDirector><isOfficer>{t if officer else f}</isOfficer>"
        f"<isTenPercentOwner>{t if ten_pct else f}</isTenPercentOwner></reportingOwnerRelationship></reportingOwner>"
    )


def _trans(code, shares, price, date="2024-03-01", acq="A"):
    return (
        "<nonDerivativeTransaction><securityTitle><value>Common Stock</value></securityTitle>"
        f"<transactionDate><value>{date}</value></transactionDate>"
        f"<transactionCoding><transactionFormType>4</transactionFormType><transactionCode>{code}</transactionCode></transactionCoding>"
        f"<transactionAmounts><transactionShares><value>{shares}</value></transactionShares>"
        f"<transactionPricePerShare><value>{price}</value></transactionPricePerShare>"
        f"<transactionAcquiredDisposedCode><value>{acq}</value></transactionAcquiredDisposedCode></transactionAmounts>"
        "</nonDerivativeTransaction>"
    )


def _form4(owners: list[str], transactions: list[str], issuer_cik=320193, derivative="") -> bytes:
    return (
        "<?xml version=\"1.0\"?><ownershipDocument><documentType>4</documentType>"
        f"<issuer><issuerCik>{issuer_cik:010d}</issuerCik><issuerTradingSymbol>AAPL</issuerTradingSymbol></issuer>"
        + "".join(owners)
        + f"<nonDerivativeTable>{''.join(transactions)}</nonDerivativeTable>"
        + (f"<derivativeTable>{derivative}</derivativeTable>" if derivative else "")
        + "</ownershipDocument>"
    ).encode()


# ---------------------------------------------------------------- XML parsing


def test_parse_keeps_only_officer_director_open_market_buys_and_sells():
    xml = _form4(
        [_owner(111, director=True)],
        [_trans("P", 1000, 10.0), _trans("S", 500, 12.0, acq="D"), _trans("F", 300, 11.0, acq="D"), _trans("M", 50, 1.0)],
        derivative="<derivativeTransaction><transactionCoding><transactionCode>P</transactionCode></transactionCoding></derivativeTransaction>",
    )
    parsed = f4.parse_form4_xml(xml, "0001-24-000001", "2024-03-04")
    assert list(parsed.columns) == f4.TRANSACTION_COLUMNS
    assert parsed["trans_code"].tolist() == ["P", "S"]  # tax withholding / exercises / derivatives dropped
    assert parsed["value"].tolist() == [10_000.0, 6_000.0]
    assert (parsed["filing_date"] == "2024-03-04").all() and (parsed["issuer_cik"] == 320193).all()


def test_parse_drops_ten_percent_owners_who_are_not_officers_or_directors():
    xml = _form4([_owner(222, ten_pct=True)], [_trans("P", 1000, 10.0)])
    assert f4.parse_form4_xml(xml, "a", "2024-03-04").empty


def test_parse_accepts_true_false_flags_and_namespaces():
    xml = _form4([_owner(333, officer=True, flag_style="bool")], [_trans("P", 10, 5.0)])
    xml = xml.replace(b"<ownershipDocument>", b'<ownershipDocument xmlns="http://example.com/ns">')
    parsed = f4.parse_form4_xml(xml, "a", "2024-03-04")
    assert parsed["is_officer"].tolist() == [True] and parsed["owner_cik"].tolist() == [333]


# ---------------------------------------------------------------- per-issuer fetch


class FakeSEC:
    def __init__(self, filings):
        # filings: list of (accession, filing_date, form, xml)
        self.filings = filings
        self.calls = []

    def __call__(self, url):
        self.calls.append(url)
        if url.endswith("company_tickers.json"):
            return json.dumps({"0": {"cik_str": 320193, "ticker": "AAPL"}, "1": {"cik_str": 1067983, "ticker": "BRK.B"}}).encode()
        if "submissions" in url:
            recent = {
                "form": [f[2] for f in self.filings],
                "accessionNumber": [f[0] for f in self.filings],
                "filingDate": [f[1] for f in self.filings],
                "primaryDocument": ["xslF345X05/form4.xml"] * len(self.filings),
            }
            return json.dumps({"filings": {"recent": recent, "files": []}}).encode()
        for acc, _, _, xml in self.filings:
            if f"/{acc.replace('-', '')}/form4.xml" in url:
                assert "xslF345" not in url  # the raw XML, not the rendered view
                return xml
        raise AssertionError(f"unexpected URL {url}")


def test_ticker_map_uses_yahoo_share_class_style():
    ciks = f4.ticker_to_cik(FakeSEC([]))
    assert ciks["AAPL"] == 320193 and ciks["BRK-B"] == 1067983


def test_issuer_fetch_parses_form4_only_and_caches_per_accession():
    xml = _form4([_owner(111, director=True)], [_trans("P", 100, 10.0)])
    sec = FakeSEC([("0001-24-000001", "2024-03-04", "4", xml), ("0001-24-000002", "2024-03-05", "4/A", xml),
                   ("0001-18-000001", "2018-01-04", "4", xml)])
    tx = f4.fetch_issuer_transactions(320193, "2019-01-01", sec)
    assert tx["accession"].tolist() == ["0001-24-000001"]  # amendment and pre-window filing skipped
    sec.calls.clear()
    f4.fetch_issuer_transactions(320193, "2019-01-01", sec)
    assert all("submissions" in url for url in sec.calls)  # filed Form 4s come from the cache


# ---------------------------------------------------------------- bulk data sets


def _bulk_zip(subs, owners, trans) -> bytes:
    def tsv(frame):
        return frame.to_csv(sep="\t", index=False)

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("SUBMISSION.tsv", tsv(pd.DataFrame(subs, columns=["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERCIK"])))
        archive.writestr("REPORTINGOWNER.tsv", tsv(pd.DataFrame(owners, columns=["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP"])))
        archive.writestr("NONDERIV_TRANS.tsv", tsv(pd.DataFrame(
            trans, columns=["ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE", "TRANS_SHARES", "TRANS_PRICEPERSHARE"])))
    return buffer.getvalue()


def _sample_zip() -> bytes:
    return _bulk_zip(
        subs=[("A1", "04-MAR-2024", "4", "0000320193"), ("A2", "05-MAR-2024", "4/A", "0000320193"),
              ("A3", "06-MAR-2024", "4", "0000320193")],
        owners=[("A1", "0000000111", "Director,Officer"), ("A2", "0000000111", "Director"), ("A3", "0000000222", "TenPercentOwner")],
        trans=[("A1", "01-MAR-2024", "P", "100", "10"), ("A1", "01-MAR-2024", "F", "5", "10"),
               ("A2", "01-MAR-2024", "P", "100", "10"), ("A3", "01-MAR-2024", "P", "100", "10")],
    )


def test_bulk_zip_keeps_original_form4_officer_director_p_s_only():
    tx = f4.parse_bulk_zip(_sample_zip())
    assert tx["accession"].tolist() == ["A1"]  # 4/A and 10%-owner filings dropped
    row = tx.iloc[0]
    assert row["filing_date"] == "2024-03-04" and row["trans_code"] == "P" and row["value"] == 1000.0
    assert int(row["issuer_cik"]) == 320193 and bool(row["is_director"]) and bool(row["is_officer"])


def test_bulk_loader_caches_quarters_and_reports_coverage_end():
    calls = []

    def fetch(url):
        calls.append(url)
        if url == f4.BULK_INDEX_URL:
            return b'<a href="/files/x/2024q1_form345.zip">q1</a>'
        if url.endswith("2024q1_form345.zip"):
            return _sample_zip()
        raise AssertionError(url)

    tx, coverage_end = f4.load_bulk_transactions("2024-01-15", "2024-05-01", fetch)  # 2024q2 not published yet
    assert len(tx) == 1 and coverage_end == "2024-03-31"
    calls.clear()
    f4.load_bulk_transactions("2024-01-15", "2024-03-20", fetch)
    assert calls == []  # the parsed quarter is cached


# ---------------------------------------------------------------- features


def _tx(owner, code, filing_date, value, accession=None, trans_date=None, shares=None):
    shares = shares if shares is not None else value / 10.0
    return {"accession": accession or f"{owner}-{filing_date}-{code}", "filing_date": filing_date, "issuer_cik": 320193,
            "owner_cik": owner, "is_director": True, "is_officer": False, "trans_code": code,
            "trans_date": trans_date or filing_date, "shares": shares, "price": 10.0, "value": value}


def _features(transactions, dates, **kwargs):
    keys = pd.DataFrame({"date": pd.to_datetime(dates), "ticker": "AAPL", "log_market_cap": np.log(1e6)})
    return f4.insider_features(keys, pd.DataFrame(transactions), {"AAPL": 320193}, **kwargs).set_index("date")


def test_transactions_count_from_their_filing_date_not_their_trade_date():
    tx = [_tx(1, "P", "2024-03-10", 1000, trans_date="2024-02-01")]  # traded Feb 1, filed Mar 10 (late filer)
    out = _features(tx, ["2024-03-09", "2024-03-10"])
    assert out.loc["2024-03-09", "insider_buyers_90d"] == 0
    assert out.loc["2024-03-10", "insider_buyers_90d"] == 1


def test_window_distinct_buyers_cluster_and_net_value():
    tx = [_tx(1, "P", "2024-01-05", 1000), _tx(1, "P", "2024-02-05", 1000),  # same buyer twice
          _tx(2, "P", "2024-02-10", 3000), _tx(3, "P", "2024-03-01", 1000),
          _tx(4, "S", "2024-03-02", 2000),
          _tx(5, "P", "2023-09-01", 9999)]  # outside the 90-day window
    out = _features(tx, ["2024-03-05"]).iloc[0]
    assert out["insider_buyers_90d"] == 3 and out["insider_cluster_buy"] == 1
    assert out["insider_net_buy_value_90d_mcap"] == pytest.approx((1000 + 1000 + 3000 + 1000 - 2000) / 1e6)


def test_joint_filing_value_is_counted_once():
    tx = [_tx(1, "P", "2024-03-01", 5000, accession="J1"), _tx(2, "P", "2024-03-01", 5000, accession="J1")]
    out = _features(tx, ["2024-03-05"]).iloc[0]
    assert out["insider_buyers_90d"] == 2 and out["insider_net_buy_value_90d_mcap"] == pytest.approx(5000 / 1e6)


def test_unknown_issuer_or_uncovered_date_is_nan_not_zero():
    tx = [_tx(1, "P", "2024-03-01", 1000)]
    out = _features(tx, ["2024-03-05", "2024-07-05"], coverage_end="2024-06-30")
    assert out.loc["2024-03-05", "insider_buyers_90d"] == 1
    assert out.loc["2024-07-05", f4.FEATURE_COLUMNS].isna().all()
    keys = pd.DataFrame({"date": pd.to_datetime(["2024-03-05"]), "ticker": ["ZZZZ"]})
    assert f4.insider_features(keys, pd.DataFrame(tx), {"AAPL": 320193})[f4.FEATURE_COLUMNS].isna().all().all()
