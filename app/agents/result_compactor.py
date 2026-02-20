"""
Compact receipt generator for tool results.

When a tool returns a large result, the full payload is stored on the session
filesystem and the conversation receives a compact *receipt* instead.  The
receipt contains:

1. **Preview** – the first 2-3 meaningful lines so the model has immediate
   signal.
2. **Gap analysis** – a short description of what the preview does NOT cover,
   so the model can decide whether to drill deeper via ``session_grep`` or
   ``read_session_file``.

Design Principles:
- Zero LLM calls – purely heuristic, no latency cost.
- Structural awareness – understands JSON objects/arrays to give richer gaps.
- Conservative – small results pass through unmodified
  (below MIN_COMPACT_SIZE).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Compact receipt
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompactReceipt:
    """Immutable compact representation of a stored tool result."""

    entry_id: str
    size_bytes: int
    preview: str
    gap: str

    def format(self) -> str:
        return (
            f"[session_fs: {self.entry_id} | {self._human_size()}]\n"
            f"Preview:\n{self.preview}\n"
            f"Gap: {self.gap}\n"
            f'Retrieve: session_grep("<pattern>") or '
            f'read_session_file("{self.entry_id}")'
        )

    def _human_size(self) -> str:
        if self.size_bytes < 1024:
            return f"{self.size_bytes} B"
        if self.size_bytes < 1024 * 1024:
            return f"{self.size_bytes / 1024:.1f} KB"
        return f"{self.size_bytes / (1024 * 1024):.1f} MB"


# ---------------------------------------------------------------------------
# Compactor
# ---------------------------------------------------------------------------

# Tools whose results should never be compacted (they are already selective).
PASSTHROUGH_TOOLS: frozenset[str] = frozenset(
    {
        "session_grep",
        "session_summary",
        "read_session_file",
        "get_current_time",
    }
)

# Results shorter than this are not worth compacting.
MIN_COMPACT_SIZE = 500

# Preview tuning
_PREVIEW_MAX_LINES = 3
_PREVIEW_MAX_CHARS = 300


class ResultCompactor:
    """Produces compact receipts from large tool results."""

    def should_compact(self, tool_name: str, text: str) -> bool:
        """Return True if result is large enough for compaction."""
        if tool_name in PASSTHROUGH_TOOLS:
            return False
        return len(text) > MIN_COMPACT_SIZE

    def compact(self, entry_id: str, text: str) -> CompactReceipt:
        """Build a compact receipt with preview and gap analysis."""
        preview = self._extract_preview(text)
        gap = self._analyze_gap(text, preview)
        return CompactReceipt(
            entry_id=entry_id,
            size_bytes=len(text.encode("utf-8")),
            preview=preview,
            gap=gap,
        )

    # -- preview extraction --------------------------------------------------

    @staticmethod
    def _extract_preview(text: str) -> str:
        """First 2-3 meaningful lines, capped at _PREVIEW_MAX_CHARS."""
        lines = text.split("\n")
        meaningful = [ln for ln in lines if ln.strip()][:_PREVIEW_MAX_LINES]
        preview = "\n".join(meaningful)
        if len(preview) > _PREVIEW_MAX_CHARS:
            preview = preview[:_PREVIEW_MAX_CHARS] + "..."
        return preview

    # -- gap analysis --------------------------------------------------------

    def _analyze_gap(self, text: str, preview: str) -> str:
        total = len(text)
        # Strip only the truncation ellipsis, not arbitrary trailing dots
        shown = len(preview[:-3]) if preview.endswith("...") else len(preview)

        if total <= shown:
            return "Preview contains the complete result."

        # Attempt structured analysis first, fall back to plain-text.
        try:
            data = json.loads(text)
            hints = self._json_gap(data, preview)
        except (json.JSONDecodeError, TypeError):
            hints = self._text_gap(text, preview)

        pct = min(shown / total * 100, 100)
        hints.append(f"showing ~{pct:.0f}% of {total} chars total")
        return ". ".join(h for h in hints if h) + "."

    # -- JSON gap analysis ---------------------------------------------------

    @staticmethod
    def _json_gap(data: object, preview: str) -> list[str]:
        hints: list[str] = []

        if isinstance(data, dict):
            keys = list(data.keys())
            hidden = [k for k in keys if k not in preview]
            if hidden:
                sample = hidden[:8]
                label = ", ".join(sample)
                extra = (
                    f" (+{len(hidden) - 8} more)" if len(hidden) > 8 else ""
                )
                hints.append(
                    f"JSON object with {len(keys)} keys; "
                    f"not in preview: {label}{extra}"
                )
            # Flag large nested arrays — often the most valuable data.
            for k, v in data.items():
                if isinstance(v, list) and len(v) > 2:
                    hints.append(f"key '{k}' is a list with {len(v)} items")

        elif isinstance(data, list):
            hints.append(f"JSON array with {len(data)} items")

        return hints

    # -- plain-text gap analysis ---------------------------------------------

    @staticmethod
    def _text_gap(text: str, preview: str) -> list[str]:
        hints: list[str] = []

        total_lines = text.count("\n") + 1
        preview_lines = preview.count("\n") + 1
        if total_lines > preview_lines:
            hints.append(
                f"{total_lines} total lines; "
                f"preview shows first {preview_lines}"
            )

        # Surface unique capitalised terms from the hidden portion.
        hidden = text[len(preview) :]
        hidden_terms = set(
            re.findall(r"\b[A-Z][a-z]{2,}(?:\s[A-Z][a-z]+)*\b", hidden)
        )
        preview_terms = set(
            re.findall(r"\b[A-Z][a-z]{2,}(?:\s[A-Z][a-z]+)*\b", preview)
        )
        unique = hidden_terms - preview_terms
        if unique:
            sample = sorted(unique)[:6]
            hints.append(f"topics not in preview: {', '.join(sample)}")

        return hints
