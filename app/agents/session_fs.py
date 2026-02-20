"""
Session filesystem with structured storage and in-memory indexing.

Design Patterns:
- Repository Pattern: SessionStore encapsulates all storage access behind
  a clean interface, decoupling the agent from filesystem details.
- Strategy Pattern: CleanupPolicy determines cleanup behavior per mode.
- Composite Pattern: Hierarchical directory structure (tools/{name}/) lets
  the agent navigate results by tool category or scan everything.

Directory Structure:
    sessions/{session_id}/
    ├── _index.json                      # Persisted master index
    └── tools/
        └── {tool_name}/
            └── {sequence}_{hash8}.json  # Individual tool results
"""

import hashlib
import json
import shutil
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Strategy: cleanup policy
# ---------------------------------------------------------------------------

class CleanupPolicy(str, Enum):
    """Determines when session files are removed.

    ALWAYS  – remove on success, error, and cancel (default for FAST).
    ON_ERROR – remove only when the run fails or is cancelled.
    NEVER   – keep files indefinitely (useful for debugging).
    """

    ALWAYS = "always"
    ON_ERROR = "on_error"
    NEVER = "never"


# ---------------------------------------------------------------------------
# Index entry
# ---------------------------------------------------------------------------

@dataclass
class IndexEntry:
    """Single entry in the session index."""

    entry_id: str
    tool_name: str
    args_hash: str
    relative_path: str
    created_at: float
    size_bytes: int
    preview: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "tool_name": self.tool_name,
            "args_hash": self.args_hash,
            "relative_path": self.relative_path,
            "created_at": self.created_at,
            "size_bytes": self.size_bytes,
            "preview": self.preview,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IndexEntry":
        return cls(
            entry_id=data["entry_id"],
            tool_name=data["tool_name"],
            args_hash=data["args_hash"],
            relative_path=data["relative_path"],
            created_at=data["created_at"],
            size_bytes=data["size_bytes"],
            preview=data.get("preview", ""),
            metadata=data.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Repository: SessionStore
# ---------------------------------------------------------------------------

class SessionStore:
    """Repository that manages structured session storage with an in-memory index.

    Public interface
    ----------------
    store_tool_result  – persist a tool result and index it
    get_by_tool        – list entries for a specific tool (O(1) dict lookup)
    get_all            – list every entry
    read               – read a stored file by entry id
    grep               – regex search across previews, then optionally full files
    summary            – aggregate stats the agent can use for orientation
    cleanup            – remove the entire session directory
    """

    _INDEX_FILE = "_index.json"
    _TOOLS_DIR = "tools"
    _PREVIEW_LENGTH = 300

    def __init__(self, session_id: str, base_path: Path | None = None):
        self.session_id = session_id
        self._base = (base_path or Path("sessions")) / session_id
        # In-memory index keyed by tool name for O(1) category lookups
        self._by_tool: dict[str, list[IndexEntry]] = {}
        # All entries in insertion order
        self._entries: list[IndexEntry] = []
        # Monotonic sequence counter per tool for readable filenames
        self._sequence: dict[str, int] = {}
        self._dirty = False

    # -- lifecycle -----------------------------------------------------------

    def create(self) -> None:
        """Create session directory structure and load any existing index."""
        (self._base / self._TOOLS_DIR).mkdir(parents=True, exist_ok=True)
        self._load_index()

    def cleanup(self) -> None:
        """Remove the entire session directory tree."""
        if self._base.exists():
            shutil.rmtree(self._base)
        self._by_tool.clear()
        self._entries.clear()
        self._sequence.clear()
        self._dirty = False

    # -- write ---------------------------------------------------------------

    async def store_tool_result(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        result: Any,
    ) -> IndexEntry:
        """Store a tool result in the structured filesystem and update the index.

        Returns the IndexEntry for the newly stored file.
        """
        safe_name = self._safe_tool_dir(tool_name)
        tool_dir = self._base / self._TOOLS_DIR / safe_name
        tool_dir.mkdir(parents=True, exist_ok=True)

        # Deterministic sequence + args hash for readable, dedup-friendly names
        seq = self._next_sequence(safe_name)
        args_hash = self._hash_args(tool_args)
        filename = f"{seq:03d}_{args_hash}.json"
        rel_path = f"{self._TOOLS_DIR}/{safe_name}/{filename}"
        abs_path = self._base / rel_path

        # Serialise with provenance metadata
        payload = {
            "tool": tool_name,
            "args": tool_args,
            "result": result,
            "timestamp": time.time(),
        }
        content = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        abs_path.write_text(content, encoding="utf-8")

        # Build index entry with a preview for fast grep
        preview = self._extract_preview(result)
        entry = IndexEntry(
            entry_id=f"{safe_name}/{seq:03d}",
            tool_name=tool_name,
            args_hash=args_hash,
            relative_path=rel_path,
            created_at=time.time(),
            size_bytes=len(content.encode()),
            preview=preview,
            metadata={"args": tool_args},
        )

        self._entries.append(entry)
        self._by_tool.setdefault(tool_name, []).append(entry)
        self._dirty = True
        self._flush_index()

        return entry

    # -- read ----------------------------------------------------------------

    def get_by_tool(self, tool_name: str) -> list[IndexEntry]:
        """O(1) lookup of all entries produced by a given tool."""
        return list(self._by_tool.get(tool_name, []))

    def get_all(self) -> list[IndexEntry]:
        """Return all index entries in insertion order."""
        return list(self._entries)

    async def read(self, entry_id: str) -> dict[str, Any]:
        """Read a stored file by its entry id.

        Raises FileNotFoundError if the entry or file is missing.
        """
        entry = self._find_entry(entry_id)
        if entry is None:
            raise FileNotFoundError(f"No index entry: {entry_id}")

        abs_path = self._base / entry.relative_path
        if not abs_path.exists():
            raise FileNotFoundError(f"File missing: {entry.relative_path}")

        return json.loads(abs_path.read_text(encoding="utf-8"))

    async def read_by_path(self, relative_path: str) -> dict[str, Any]:
        """Read a stored file by its relative path."""
        abs_path = self._base / relative_path
        if not abs_path.exists():
            raise FileNotFoundError(f"File not found: {relative_path}")
        return json.loads(abs_path.read_text(encoding="utf-8"))

    # -- search --------------------------------------------------------------

    async def grep(
        self,
        pattern: str,
        case_sensitive: bool = False,
        tool_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Two-phase search: scan previews first, then full files for context.

        This avoids reading every file from disk when the preview already
        contains enough signal.
        """
        import re

        flags = 0 if case_sensitive else re.IGNORECASE
        regex = re.compile(pattern, flags)

        entries = self._by_tool.get(tool_filter, []) if tool_filter else self._entries
        matches: list[dict[str, Any]] = []

        for entry in entries:
            # Phase 1: check preview (fast, in-memory)
            preview_match = regex.search(entry.preview)
            if preview_match:
                matches.append({
                    "entry_id": entry.entry_id,
                    "tool": entry.tool_name,
                    "match": preview_match.group(0),
                    "context": entry.preview,
                    "source": "preview",
                })
                continue

            # Phase 2: fall back to full file read
            try:
                data = await self.read(entry.entry_id)
                content_str = json.dumps(data.get("result", ""), default=str)
                full_match = regex.search(content_str)
                if full_match:
                    start = max(0, full_match.start() - 100)
                    end = min(len(content_str), full_match.end() + 100)
                    matches.append({
                        "entry_id": entry.entry_id,
                        "tool": entry.tool_name,
                        "match": full_match.group(0),
                        "context": content_str[start:end],
                        "source": "full",
                    })
            except Exception:
                continue

        return matches

    # -- summary -------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Aggregate stats the agent can use for quick orientation."""
        tools_used = list(self._by_tool.keys())
        total_size = sum(e.size_bytes for e in self._entries)
        per_tool = {
            name: {
                "count": len(entries),
                "total_bytes": sum(e.size_bytes for e in entries),
                "latest": entries[-1].entry_id if entries else None,
            }
            for name, entries in self._by_tool.items()
        }
        return {
            "session_id": self.session_id,
            "total_entries": len(self._entries),
            "total_size_bytes": total_size,
            "tools_used": tools_used,
            "per_tool": per_tool,
        }

    # -- internal helpers ----------------------------------------------------

    def _load_index(self) -> None:
        """Load persisted index into memory."""
        index_path = self._base / self._INDEX_FILE
        if not index_path.exists():
            return

        try:
            raw = json.loads(index_path.read_text(encoding="utf-8"))
            for item in raw.get("entries", []):
                entry = IndexEntry.from_dict(item)
                self._entries.append(entry)
                self._by_tool.setdefault(entry.tool_name, []).append(entry)

                # Restore sequence counters
                safe = self._safe_tool_dir(entry.tool_name)
                parts = entry.entry_id.rsplit("/", 1)
                if len(parts) == 2:
                    try:
                        seq = int(parts[1])
                        self._sequence[safe] = max(self._sequence.get(safe, 0), seq)
                    except ValueError:
                        pass
        except (json.JSONDecodeError, KeyError):
            pass

    def _flush_index(self) -> None:
        """Persist the in-memory index to disk."""
        if not self._dirty:
            return
        index_path = self._base / self._INDEX_FILE
        payload = {
            "session_id": self.session_id,
            "entry_count": len(self._entries),
            "entries": [e.to_dict() for e in self._entries],
        }
        index_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self._dirty = False

    def _find_entry(self, entry_id: str) -> IndexEntry | None:
        for e in self._entries:
            if e.entry_id == entry_id:
                return e
        return None

    def _next_sequence(self, safe_name: str) -> int:
        current = self._sequence.get(safe_name, 0)
        nxt = current + 1
        self._sequence[safe_name] = nxt
        return nxt

    @staticmethod
    def _hash_args(args: dict[str, Any]) -> str:
        raw = json.dumps(args, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:8]

    @staticmethod
    def _safe_tool_dir(tool_name: str) -> str:
        return tool_name.replace("/", "_").replace("\\", "_").replace(" ", "_")

    @staticmethod
    def _extract_preview(result: Any) -> str:
        """Extract a text preview from a result for the in-memory index."""
        if isinstance(result, str):
            text = result
        else:
            text = json.dumps(result, default=str, ensure_ascii=False)
        return text[:SessionStore._PREVIEW_LENGTH]


# ---------------------------------------------------------------------------
# Lifecycle: context manager with cleanup-policy awareness
# ---------------------------------------------------------------------------

class SessionLifecycle:
    """Manages the full create → use → cleanup lifecycle of a session store.

    Usage::

        async with SessionLifecycle("sid-123", CleanupPolicy.ALWAYS) as store:
            await store.store_tool_result(...)

    On exit the lifecycle checks the policy and whether an error occurred,
    then cleans up accordingly.
    """

    def __init__(
        self,
        session_id: str,
        policy: CleanupPolicy = CleanupPolicy.ALWAYS,
        base_path: Path | None = None,
    ):
        self._store = SessionStore(session_id, base_path)
        self._policy = policy
        self._error_occurred = False

    @property
    def store(self) -> SessionStore:
        return self._store

    def mark_error(self) -> None:
        """Signal that the run hit an error (used by the agent)."""
        self._error_occurred = True

    async def __aenter__(self) -> SessionStore:
        self._store.create()
        return self._store

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            self._error_occurred = True

        should_cleanup = (
            self._policy == CleanupPolicy.ALWAYS
            or (self._policy == CleanupPolicy.ON_ERROR and self._error_occurred)
        )

        if should_cleanup:
            self._store.cleanup()

        return None  # do not suppress exceptions


# ---------------------------------------------------------------------------
# Backwards-compatible models (kept for imports)
# ---------------------------------------------------------------------------

class AgentMode(BaseModel):
    """Agent execution mode."""

    mode: str = Field(description="Mode: fast or agentic")
    max_tools_fast: int = Field(default=1, description="Max tool calls in fast mode")
    allow_parallel_fast: bool = Field(
        default=False, description="Allow parallel tools in fast mode"
    )
    filesystem_enabled: bool = Field(
        default=True, description="Enable filesystem in agentic mode"
    )

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_session_store(
    session_id: str,
    base_path: Path | None = None,
) -> SessionStore:
    """Factory to create a SessionStore.

    Args:
        session_id: Unique session identifier
        base_path: Optional base directory (defaults to ``sessions/``)

    Returns:
        A SessionStore instance (call ``.create()`` before use).
    """
    return SessionStore(session_id, base_path)
