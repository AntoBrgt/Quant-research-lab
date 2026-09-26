"""Local institutional report loading -- no network, temp directories only."""

from institutional_research import documents


def test_list_report_files_finds_supported_extensions_only(tmp_path):
    inst_dir = tmp_path / "BlackRock"
    inst_dir.mkdir()
    (inst_dir / "outlook.txt").write_text("2026 outlook text", encoding="utf-8")
    (inst_dir / "outlook.meta.json").write_text("{}", encoding="utf-8")  # must be excluded
    (inst_dir / "notes.xlsx").write_text("not supported", encoding="utf-8")

    files = documents.list_report_files(root=tmp_path)
    assert len(files) == 1
    assert files[0].name == "outlook.txt"


def test_institution_is_derived_from_parent_folder(tmp_path):
    inst_dir = tmp_path / "J.P. Morgan Asset Management"
    inst_dir.mkdir()
    report_path = inst_dir / "2026_market_outlook.txt"
    report_path.write_text("We favor European financials given improving margins.", encoding="utf-8")

    loaded = documents.load_report(report_path, root=tmp_path)
    assert loaded["report"].institution == "J.P. Morgan Asset Management"
    assert loaded["report"].report_title == "2026 Market Outlook"
    assert loaded["text"].startswith("We favor European financials")


def test_meta_json_sidecar_overrides_guessed_metadata(tmp_path):
    inst_dir = tmp_path / "BlackRock"
    inst_dir.mkdir()
    report_path = inst_dir / "notes.txt"
    report_path.write_text("Some thematic commentary.", encoding="utf-8")
    (report_path.with_suffix(".txt.meta.json")).write_text(
        '{"report_title": "Global Investment Outlook 2026", "publication_date": "2026-01-15", '
        '"report_url": "https://example.com/outlook.pdf", "report_type": "outlook"}',
        encoding="utf-8",
    )

    loaded = documents.load_report(report_path, root=tmp_path)
    report = loaded["report"]
    assert report.report_title == "Global Investment Outlook 2026"
    assert report.publication_date == "2026-01-15"
    assert report.report_url == "https://example.com/outlook.pdf"
    assert report.report_type == "outlook"


def test_empty_file_is_skipped_not_raised(tmp_path):
    inst_dir = tmp_path / "BlackRock"
    inst_dir.mkdir()
    empty_path = inst_dir / "empty.txt"
    empty_path.write_text("   ", encoding="utf-8")

    assert documents.load_report(empty_path, root=tmp_path) is None


def test_load_all_reports_skips_bad_files_and_loads_good_ones(tmp_path):
    inst_dir = tmp_path / "BlackRock"
    inst_dir.mkdir()
    (inst_dir / "good.txt").write_text("Real content about AI infrastructure demand.", encoding="utf-8")
    (inst_dir / "empty.txt").write_text("", encoding="utf-8")

    loaded = documents.load_all_reports(root=tmp_path)
    assert len(loaded) == 1
    assert "AI infrastructure" in loaded[0]["text"]


def test_publication_date_guessed_from_filename_when_no_sidecar(tmp_path):
    inst_dir = tmp_path / "BlackRock"
    inst_dir.mkdir()
    report_path = inst_dir / "2026-03-01_outlook.txt"
    report_path.write_text("Some commentary with no obvious date in the body.", encoding="utf-8")

    loaded = documents.load_report(report_path, root=tmp_path)
    assert loaded["report"].publication_date == "2026-03-01"
