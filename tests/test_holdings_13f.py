"""13F holdings provider -- fully offline (fake SEC fetcher + fake CUSIP resolver)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

import config
from institutional_research import holdings_13f as h13f
from institutional_research import universe
from institutional_research.schemas import OUTPUT_COLUMNS

NS = "http://www.sec.gov/edgar/document/thirteenf/informationtable"


def _info_row(name, cusip, shares, value, put_call=None, amt_type="SH", cls="COM"):
    pc = f"<putCall>{put_call}</putCall>" if put_call else ""
    return (
        f"<infoTable><nameOfIssuer>{name}</nameOfIssuer><titleOfClass>{cls}</titleOfClass>"
        f"<cusip>{cusip}</cusip><value>{value}</value>"
        f"<shrsOrPrnAmt><sshPrnamt>{shares}</sshPrnamt><sshPrnamtType>{amt_type}</sshPrnamtType></shrsOrPrnAmt>"
        f"{pc}<investmentDiscretion>SOLE</investmentDiscretion>"
        f"<votingAuthority><Sole>{shares}</Sole><Shared>0</Shared><None>0</None></votingAuthority></infoTable>"
    )


def _table(rows: list[str], prefixed: bool = False) -> bytes:
    if prefixed:  # some filers use an ns1: prefix -- parsing must not care
        body = "".join(rows).replace("<", "<ns1:").replace("<ns1:/", "</ns1:")
        return f'<ns1:informationTable xmlns:ns1="{NS}">{body}</ns1:informationTable>'.encode()
    return f'<informationTable xmlns="{NS}">{"".join(rows)}</informationTable>'.encode()


# ---------------------------------------------------------------- parsing


def test_parse_drops_options_and_bonds_and_sums_duplicate_cusips():
    xml = _table([
        _info_row("APPLE INC", "037833100", 100, 20_000),
        _info_row("APPLE INC", "037833100", 50, 10_000),  # second sub-manager row
        _info_row("APPLE INC", "037833100", 999, 5_000, put_call="Call"),
        _info_row("SOME BOND", "000000AA1", 1_000_000, 1_000_000, amt_type="PRN"),
    ])
    parsed = h13f.parse_information_table(xml)
    assert parsed["cusip"].tolist() == ["037833100"]
    assert parsed.iloc[0]["shares"] == 150
    assert parsed.iloc[0]["value"] == 30_000


def test_parse_is_namespace_prefix_agnostic():
    xml = _table([_info_row("MICROSOFT CORP", "594918104", 10, 4_000)], prefixed=True)
    assert h13f.parse_information_table(xml)["cusip"].tolist() == ["594918104"]


# ---------------------------------------------------------------- change logic


def _holdings(rows):
    return pd.DataFrame(rows, columns=["cusip", "name_of_issuer", "title_of_class", "shares", "value"])


def test_flow_adjustment_makes_index_inflows_neutral():
    # Every position +5% shares (fund inflows) except B (+30%) and C (-30%).
    prev = _holdings([(c, c, "COM", 1000, 1000 * 10) for c in "ABCDEFG"])
    cur = _holdings(
        [("A", "A", "COM", 1050, 10500), ("B", "B", "COM", 1300, 13000), ("C", "C", "COM", 700, 7000)]
        + [(c, c, "COM", 1050, 10500) for c in "DEFG"]
    )
    changes = h13f.compare_quarters(cur, prev).set_index("cusip")
    assert changes.loc["A", "status"] == "UNCHANGED"  # +5% raw, but that's just the filer's flow
    assert changes.loc["B", "status"] == "ADDED"
    assert changes.loc["C", "status"] == "REDUCED"
    assert changes.loc["A", "adjusted_change"] == pytest.approx(0.0)


def test_stock_split_is_not_mistaken_for_buying():
    prev = _holdings([("S", "S", "COM", 1000, 400_000)] + [(c, c, "COM", 100, 1000) for c in "XYZ"])
    # 4:1 split: 4x shares, implied price 400 -> ~100 (value moved +2%)
    cur = _holdings([("S", "S", "COM", 4000, 408_000)] + [(c, c, "COM", 100, 1000) for c in "XYZ"])
    row = h13f.compare_quarters(cur, prev).set_index("cusip").loc["S"]
    assert row["split_factor"] == 4
    assert row["status"] == "UNCHANGED"


def test_new_and_exited_positions():
    prev = _holdings([("OLD", "OLD CO", "COM", 100, 1000), ("K", "K", "COM", 100, 1000)])
    cur = _holdings([("NEW", "NEW CO", "COM", 100, 1000), ("K", "K", "COM", 100, 1000)])
    changes = h13f.compare_quarters(cur, prev).set_index("cusip")
    assert changes.loc["NEW", "status"] == "NEW"
    assert changes.loc["OLD", "status"] == "EXITED"
    assert changes.loc["OLD", "name_of_issuer"] == "OLD CO"
    assert changes.loc["OLD", "value"] == 0


def test_no_prior_quarter():
    cur = _holdings([("A", "A", "COM", 1, 10)])
    assert h13f.compare_quarters(cur, None)["status"].tolist() == ["NO_PRIOR"]


def test_select_positions_caps_top_and_movers(monkeypatch):
    monkeypatch.setattr(h13f, "TOP_N_BY_VALUE", 2)
    monkeypatch.setattr(h13f, "TOP_N_MOVERS", 1)
    prev = _holdings([(f"C{i}", f"C{i}", "COM", 100, 1000 - i) for i in range(10)])
    cur_rows = [(f"C{i}", f"C{i}", "COM", 100, 1000 - i) for i in range(10)]
    cur_rows[5] = ("C5", "C5", "COM", 200, 1990)  # big add, now #1 by value
    cur_rows[8] = ("C8", "C8", "COM", 10, 99)  # big cut
    cur = _holdings(cur_rows)
    changes = h13f.compare_quarters(cur, prev)
    selected = h13f.select_positions(changes, prev)
    assert set(selected["cusip"]) == {"C5", "C0", "C8"}  # top-2 (C5, C0) + 1 add (C5, deduped) + 1 cut (C8)


# ---------------------------------------------------------------- end to end


class FakeSEC:
    """Serves submissions JSON, filing indexes, and info tables for one filer."""

    def __init__(self, cik, quarters):
        # quarters: list of (accession, filing_date, report_date, xml_bytes), newest first
        self.cik = cik
        self.quarters = quarters
        self.calls = []

    def __call__(self, url):
        self.calls.append(url)
        if "submissions" in url:
            recent = {
                "form": ["SC 13G"] + ["13F-HR"] * len(self.quarters) + ["13F-HR/A"],
                "accessionNumber": ["x"] + [q[0] for q in self.quarters] + ["amend"],
                "filingDate": ["2026-09-01"] + [q[1] for q in self.quarters] + ["2026-09-02"],
                "reportDate": [""] + [q[2] for q in self.quarters] + [self.quarters[0][2]],
            }
            return json.dumps({"filings": {"recent": recent, "files": []}}).encode()
        for acc, _, _, xml in self.quarters:
            nodash = acc.replace("-", "")
            if url.endswith(f"{nodash}/index.json"):
                return json.dumps({"directory": {"item": [
                    {"name": "primary_doc.xml"}, {"name": "infotable.xml"}, {"name": f"{nodash}-index.html"},
                ]}}).encode()
            if url.endswith(f"{nodash}/infotable.xml"):
                return xml
        raise AssertionError(f"unexpected URL {url}")


@pytest.fixture
def isolated_data(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_DATA_DIR", tmp_path / "raw")
    monkeypatch.setattr(config, "INSTITUTIONAL_MENTIONS_PATH", tmp_path / "processed" / "mentions.parquet")
    return tmp_path


def _fake_resolver(cusips):
    table = {
        "037833100": {"ticker": "AAPL", "security_type": "Common Stock"},
        "594918104": {"ticker": "MSFT", "security_type": "Common Stock"},
        "78462F103": {"ticker": "SPY", "security_type": "ETP"},
    }
    return {c: table.get(c) for c in cusips}


def test_refresh_end_to_end_and_cache(isolated_data):
    q1 = _table([_info_row("APPLE INC", "037833100", 100, 20_000), _info_row("MICROSOFT CORP", "594918104", 100, 40_000)])
    q2 = _table([
        _info_row("APPLE INC", "037833100", 200, 44_000),  # doubled
        _info_row("MICROSOFT CORP", "594918104", 100, 42_000),
        _info_row("SPDR S&amp;P 500", "78462F103", 10, 5_000),  # ETF, excluded
        _info_row("UNKNOWN CO", "999999999", 10, 100),  # unresolved ticker
    ])
    sec = FakeSEC(1067983, [("0001-26-000002", "2026-08-14", "2026-06-30", q2), ("0001-26-000001", "2026-05-15", "2026-03-31", q1)])

    mentions, summaries = h13f.refresh_13f_mentions({"Berkshire": 1067983}, fetch=sec, resolve=_fake_resolver)
    assert list(mentions.columns) == OUTPUT_COLUMNS
    assert summaries[0]["status"] == "ok" and summaries[0]["quarter"] == "2026-06-30"
    by_ticker = mentions.set_index("company_name")
    assert "SPDR S&P 500" not in by_ticker.index
    assert by_ticker.loc["APPLE INC", "ticker"] == "AAPL"
    assert by_ticker.loc["APPLE INC", "view_direction"] == "POSITIVE"
    assert by_ticker.loc["MICROSOFT CORP", "view_direction"] == "NEGATIVE"  # flat while the filer's median doubled
    assert pd.isna(by_ticker.loc["UNKNOWN CO", "ticker"])
    assert (mentions["report_type"] == "13F").all()
    assert "BUY" not in " ".join(mentions["institutional_view"]).upper()

    # Second run: filings already cached on disk -> only the submissions JSON is fetched again.
    sec.calls.clear()
    h13f.refresh_13f_mentions({"Berkshire": 1067983}, fetch=sec, resolve=_fake_resolver)
    assert all("submissions" in url for url in sec.calls)

    # The universe builds from these rows unchanged (ticker-less rows excluded).
    uni = universe.build_universe(mentions, enrich_market_data=False)
    assert set(uni["ticker"]) == {"AAPL", "MSFT"}


def test_one_failing_filer_does_not_stop_others(isolated_data):
    q = _table([_info_row("APPLE INC", "037833100", 100, 20_000)])
    good = FakeSEC(1, [("0001-26-000001", "2026-08-14", "2026-06-30", q)])

    def fetch(url):
        if "CIK0000000002" in url:
            raise ConnectionError("SEC down")
        return good(url)

    mentions, summaries = h13f.refresh_13f_mentions({"Good": 1, "Bad": 2}, fetch=fetch, resolve=_fake_resolver)
    assert set(mentions["institution"]) == {"Good"}
    assert summaries[1]["status"].startswith("error")


def test_merge_replaces_old_13f_rows_but_keeps_report_mentions(isolated_data):
    report_row = {c: None for c in OUTPUT_COLUMNS} | {
        "institution": "Berkshire", "report_type": "outlook", "ticker": "KO", "chunk_id": "r1",
        "institutional_view": "x", "view_direction": "POSITIVE", "confidence": "Low", "evidence": "e",
    }
    old_13f = report_row | {"report_type": "13F", "ticker": "OLD", "chunk_id": "13F:old"}
    new_13f = report_row | {"report_type": "13F", "ticker": "AAPL", "chunk_id": "13F:new"}
    pd.DataFrame([report_row, old_13f]).to_parquet(config.INSTITUTIONAL_MENTIONS_PATH.parent.mkdir(parents=True) or config.INSTITUTIONAL_MENTIONS_PATH)

    combined = h13f.merge_13f_mentions(pd.DataFrame([new_13f]))
    assert set(combined["ticker"]) == {"KO", "AAPL"}


def test_load_filers_defaults_and_csv(isolated_data):
    assert h13f.load_filers() == h13f.DEFAULT_FILERS
    path = config.RAW_DATA_DIR / "13f_filers.csv"
    path.parent.mkdir(parents=True)
    path.write_text("institution,cik\nBerkshire Hathaway,1067983\n")
    assert h13f.load_filers() == {"Berkshire Hathaway": 1067983}


def test_needs_refresh(isolated_data):
    assert h13f.needs_refresh()
    h13f.mark_refreshed()
    assert not h13f.needs_refresh()
    assert h13f.needs_refresh(max_age_days=0)


def test_needs_refresh_after_code_version_bump(isolated_data, monkeypatch):
    h13f.mark_refreshed()
    assert not h13f.needs_refresh()
    monkeypatch.setattr(h13f, "REFRESH_VERSION", "999")
    assert h13f.needs_refresh()
