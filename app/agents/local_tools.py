"""Local tool definitions for chat agents.

Each tool is bound to a SessionStore instance so it can read/search
the session filesystem. Pass ``store=None`` for tools that only
need ``get_current_time``.
"""

import time

from strands import tool

from app.agents.session_fs import SessionStore


def create_session_tools(store: SessionStore | None) -> list:
    """Build session-aware tools bound to *store*."""

    @tool
    async def get_current_time() -> str:
        """Get the current time."""
        return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    @tool
    async def session_grep(
        pattern: str, case_sensitive: bool = False
    ) -> str:
        """Search session filesystem for a regex pattern.

        Args:
            pattern: Regex pattern to search.
            case_sensitive: Whether the search is case sensitive.
        """
        if not store:
            return "No session filesystem available"

        matches = await store.grep(pattern, case_sensitive)
        if not matches:
            return f"No matches for: {pattern}"

        lines = [f"Found {len(matches)} matches:"]
        for m in matches[:20]:
            lines.append(f"\n- Entry: {m['entry_id']}")
            lines.append(f"  Tool: {m['tool']}")
            lines.append(f"  Match: {m['match']}")
        return "\n".join(lines)

    @tool
    async def session_summary() -> str:
        """Get summary of session filesystem data."""
        if not store:
            return "No session filesystem available"

        s = store.summary()
        per_tool = []
        for name, info in s.get("per_tool", {}).items():
            per_tool.append(
                f"  - {name}: "
                f"{info['count']} results, "
                f"{info['total_bytes']} bytes"
            )

        return (
            f"Session: {s['session_id']}\n"
            f"Total entries: {s['total_entries']}\n"
            f"Tools used:\n" + "\n".join(per_tool)
        )

    @tool
    async def read_session_file(entry_id: str) -> str:
        """Read a file from the session filesystem.

        Args:
            entry_id: Entry identifier (e.g. 'search/001').
        """
        if not store:
            return "No session filesystem available"

        try:
            data = await store.read(entry_id)
            return f"Contents of {entry_id}:\n\n{data}"
        except FileNotFoundError:
            return f"Entry not found: {entry_id}"

    return [
        get_current_time,
        session_grep,
        session_summary,
        read_session_file,
    ]
