"""
Tests for ResultCompactor and CompactReceipt.
"""

import json

import pytest

from app.agents.result_compactor import (
    MIN_COMPACT_SIZE,
    PASSTHROUGH_TOOLS,
    CompactReceipt,
    ResultCompactor,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def compactor():
    return ResultCompactor()


# ---------------------------------------------------------------------------
# CompactReceipt.format
# ---------------------------------------------------------------------------


class TestCompactReceipt:
    def test_format_includes_all_sections(self):
        r = CompactReceipt(
            entry_id="search/001",
            size_bytes=2048,
            preview="Line one\nLine two",
            gap="5 more fields not shown.",
        )
        text = r.format()
        assert "[session_fs: search/001 | 2.0 KB]" in text
        assert "Preview:" in text
        assert "Line one" in text
        assert "Gap: 5 more fields not shown." in text
        assert 'read_session_file("search/001")' in text
        assert 'session_grep("<pattern>")' in text

    def test_human_size_bytes(self):
        r = CompactReceipt(entry_id="x", size_bytes=512, preview="", gap="")
        assert "512 B" in r.format()

    def test_human_size_kilobytes(self):
        r = CompactReceipt(entry_id="x", size_bytes=5120, preview="", gap="")
        assert "5.0 KB" in r.format()

    def test_human_size_megabytes(self):
        r = CompactReceipt(entry_id="x", size_bytes=2 * 1024 * 1024, preview="", gap="")
        assert "2.0 MB" in r.format()


# ---------------------------------------------------------------------------
# ResultCompactor.should_compact
# ---------------------------------------------------------------------------


class TestShouldCompact:
    def test_small_result_not_compacted(self, compactor):
        assert compactor.should_compact("search", "short") is False

    def test_large_result_compacted(self, compactor):
        assert compactor.should_compact("search", "x" * (MIN_COMPACT_SIZE + 1)) is True

    def test_passthrough_tools_never_compacted(self, compactor):
        big = "x" * (MIN_COMPACT_SIZE + 100)
        for tool_name in PASSTHROUGH_TOOLS:
            assert compactor.should_compact(tool_name, big) is False

    def test_boundary_not_compacted(self, compactor):
        assert compactor.should_compact("tool", "x" * MIN_COMPACT_SIZE) is False

    def test_boundary_plus_one_compacted(self, compactor):
        assert compactor.should_compact("tool", "x" * (MIN_COMPACT_SIZE + 1)) is True


# ---------------------------------------------------------------------------
# ResultCompactor.compact – preview extraction
# ---------------------------------------------------------------------------


class TestPreviewExtraction:
    def test_takes_first_meaningful_lines(self, compactor):
        text = "\n\nLine 1\n\nLine 2\n\nLine 3\n\nLine 4\n" + "x" * 600
        receipt = compactor.compact("t/001", text)
        assert "Line 1" in receipt.preview
        assert "Line 2" in receipt.preview
        assert "Line 3" in receipt.preview
        assert "Line 4" not in receipt.preview  # only 3 lines

    def test_skips_blank_lines(self, compactor):
        text = "\n\n\nOnly line\n" + "x" * 600
        receipt = compactor.compact("t/001", text)
        assert "Only line" in receipt.preview

    def test_truncates_long_lines(self, compactor):
        text = "a" * 500 + "\n" + "b" * 500
        receipt = compactor.compact("t/001", text)
        assert receipt.preview.endswith("...")
        assert len(receipt.preview) <= 304  # 300 + "..."


# ---------------------------------------------------------------------------
# ResultCompactor.compact – gap analysis (JSON)
# ---------------------------------------------------------------------------


class TestJsonGapAnalysis:
    def test_json_object_lists_hidden_keys(self, compactor):
        data = {
            "temperature": 20,
            "humidity": 80,
            "forecast_5day": [1, 2, 3, 4, 5],
            "alerts": ["storm warning"],
        }
        text = json.dumps(data, indent=2)
        receipt = compactor.compact("weather/001", text)

        # The preview is short, so most keys should be flagged as hidden
        assert "keys" in receipt.gap.lower()

    def test_json_array_reports_length(self, compactor):
        data = [{"id": i, "title": f"Item {i}"} for i in range(50)]
        text = json.dumps(data)
        receipt = compactor.compact("search/001", text)
        assert "50 items" in receipt.gap

    def test_json_nested_list_flagged(self, compactor):
        data = {"results": [{"id": i} for i in range(10)], "meta": "ok"}
        text = json.dumps(data, indent=2)
        receipt = compactor.compact("s/001", text)
        assert "results" in receipt.gap
        assert "10 items" in receipt.gap


# ---------------------------------------------------------------------------
# ResultCompactor.compact – gap analysis (plain text)
# ---------------------------------------------------------------------------


class TestTextGapAnalysis:
    def test_reports_total_line_count(self, compactor):
        lines = [f"Line {i}: some content here" for i in range(50)]
        text = "\n".join(lines)
        receipt = compactor.compact("t/001", text)
        assert "50 total lines" in receipt.gap

    def test_surfaces_hidden_topics(self, compactor):
        # First 3 lines (the preview) mention London only.
        # Lines 4+ introduce Berlin and Tokyo which should appear in the gap.
        preview_lines = "The weather in London is sunny today.\n" * 4
        hidden_lines = (
            "Additional data about Berlin and Tokyo shows different patterns.\n"
            * 20
        )
        text = preview_lines + hidden_lines
        receipt = compactor.compact("t/001", text)
        gap_lower = receipt.gap.lower()
        assert "berlin" in gap_lower or "tokyo" in gap_lower or "topics" in gap_lower

    def test_percentage_shown(self, compactor):
        text = "x" * 1000
        receipt = compactor.compact("t/001", text)
        assert "%" in receipt.gap
        assert "1000 chars" in receipt.gap


# ---------------------------------------------------------------------------
# ResultCompactor.compact – complete result
# ---------------------------------------------------------------------------


class TestCompleteResult:
    def test_short_result_gap_says_complete(self, compactor):
        # Manually test _analyze_gap for a result shorter than the preview
        # (should_compact would normally prevent this, but test the logic)
        text = "Short result"
        receipt = compactor.compact("t/001", text)
        assert "complete" in receipt.gap.lower()

    def test_receipt_entry_id_preserved(self, compactor):
        text = "x" * 600
        receipt = compactor.compact("search/042", text)
        assert receipt.entry_id == "search/042"

    def test_receipt_size_bytes_utf8(self, compactor):
        # Multibyte characters
        text = "\u00e9" * 600  # é is 2 bytes in UTF-8
        receipt = compactor.compact("t/001", text)
        assert receipt.size_bytes == len(text.encode("utf-8"))
