"""Section-heading detection -- no network/files required, synthetic filing text.

Regression coverage for two real bugs found against live filings:
1. The MD&A pattern required the line to end right after "analysis", but the
   real heading continues with "of Financial Condition and Results of
   Operations" -- so MD&A, the most signal-dense section in a 10-K, was never
   actually detected in any filing processed before the fix.
2. Filings use a curly apostrophe (U+2019) from HTML-entity decoding, not the
   ASCII "'" the regexes were written with.
"""

import process_documents as pd_module


def test_mda_full_heading_with_curly_apostrophe_is_detected():
    heading = "ITEM 7.MANAGEMENT’S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS"
    assert pd_module.classify_section(heading) == "Management's Discussion and Analysis"


def test_mda_bare_heading_still_detected():
    assert pd_module.classify_section("Management's Discussion and Analysis") == "Management's Discussion and Analysis"
    assert pd_module.classify_section("Management’s Discussion and Analysis") == "Management's Discussion and Analysis"


def test_market_for_registrants_common_equity_with_curly_apostrophe():
    heading = "Item 5. Market for Registrant’s Common Equity"
    assert pd_module.classify_section(heading) == "Market for Registrant's Common Equity"


def test_detect_sections_extracts_mda_body_not_just_toc():
    text = "\n".join(
        [
            "TABLE OF CONTENTS",
            "Item 7.",
            "",
            "Management’s Discussion and Analysis",
            "",
            "27",  # page number -- this is the ToC row, should be skipped
            "",
            "ITEM 7.MANAGEMENT’S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS",
            "",
            "Revenue increased 12% year over year driven by strong demand.",
            "",
            "Item 7A. Quantitative and Qualitative Disclosures About Market Risk",
            "",
            "We are exposed to market risk from changes in interest rates.",
        ]
    )
    sections = pd_module.detect_sections(text)
    mda_sections = [s for s in sections if s["section"] == "Management's Discussion and Analysis"]
    assert len(mda_sections) == 1
    assert "Revenue increased 12%" in mda_sections[0]["text"]
