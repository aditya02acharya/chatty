"""
Tests for session filesystem: SessionStore, SessionLifecycle, and CleanupPolicy.
"""

import json

import pytest

# Import directly from session_fs to avoid pulling in strands via __init__.py
from app.agents.session_fs import (  # noqa: E402
    CleanupPolicy,
    IndexEntry,
    SessionLifecycle,
    SessionStore,
    create_session_store,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_sessions(tmp_path):
    """Provide a temp directory as the sessions base path."""
    return tmp_path


@pytest.fixture
def store(tmp_sessions):
    """Create and initialise a SessionStore backed by tmp_path."""
    s = SessionStore("test-session-001", base_path=tmp_sessions)
    s.create()
    return s


# ---------------------------------------------------------------------------
# SessionStore: directory structure
# ---------------------------------------------------------------------------


class TestSessionStoreStructure:
    def test_create_builds_directory_tree(self, tmp_sessions):
        store = SessionStore("sid-1", base_path=tmp_sessions)
        store.create()

        session_dir = tmp_sessions / "sid-1"
        assert session_dir.is_dir()
        assert (session_dir / "tools").is_dir()

    def test_cleanup_removes_directory(self, store, tmp_sessions):
        session_dir = tmp_sessions / "test-session-001"
        assert session_dir.exists()

        store.cleanup()
        assert not session_dir.exists()

    def test_cleanup_is_idempotent(self, store):
        store.cleanup()
        store.cleanup()  # should not raise


# ---------------------------------------------------------------------------
# SessionStore: store & read
# ---------------------------------------------------------------------------


class TestSessionStoreReadWrite:
    @pytest.mark.asyncio
    async def test_store_creates_tool_subdirectory(self, store, tmp_sessions):
        await store.store_tool_result("get_weather", {"city": "London"}, {"temp": 20})

        tool_dir = tmp_sessions / "test-session-001" / "tools" / "get_weather"
        assert tool_dir.is_dir()
        files = list(tool_dir.glob("*.json"))
        assert len(files) == 1

    @pytest.mark.asyncio
    async def test_stored_file_contains_provenance(self, store, tmp_sessions):
        await store.store_tool_result("search", {"q": "python"}, "some results")

        tool_dir = tmp_sessions / "test-session-001" / "tools" / "search"
        files = list(tool_dir.glob("*.json"))
        data = json.loads(files[0].read_text())

        assert data["tool"] == "search"
        assert data["args"] == {"q": "python"}
        assert data["result"] == "some results"
        assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_store_returns_index_entry(self, store):
        entry = await store.store_tool_result("my_tool", {"a": 1}, "result")

        assert isinstance(entry, IndexEntry)
        assert entry.tool_name == "my_tool"
        assert entry.entry_id == "my_tool/001"
        assert entry.preview.startswith("result")

    @pytest.mark.asyncio
    async def test_read_by_entry_id(self, store):
        entry = await store.store_tool_result("tool_a", {}, {"key": "value"})
        data = await store.read(entry.entry_id)

        assert data["tool"] == "tool_a"
        assert data["result"] == {"key": "value"}

    @pytest.mark.asyncio
    async def test_read_missing_entry_raises(self, store):
        with pytest.raises(FileNotFoundError, match="No index entry"):
            await store.read("nonexistent/999")

    @pytest.mark.asyncio
    async def test_multiple_results_same_tool(self, store):
        e1 = await store.store_tool_result("search", {"q": "a"}, "result_a")
        e2 = await store.store_tool_result("search", {"q": "b"}, "result_b")

        assert e1.entry_id == "search/001"
        assert e2.entry_id == "search/002"

        # Both readable
        d1 = await store.read(e1.entry_id)
        d2 = await store.read(e2.entry_id)
        assert d1["result"] == "result_a"
        assert d2["result"] == "result_b"


# ---------------------------------------------------------------------------
# SessionStore: in-memory index
# ---------------------------------------------------------------------------


class TestSessionStoreIndex:
    @pytest.mark.asyncio
    async def test_get_by_tool_returns_correct_entries(self, store):
        await store.store_tool_result("tool_a", {}, "r1")
        await store.store_tool_result("tool_b", {}, "r2")
        await store.store_tool_result("tool_a", {}, "r3")

        a_entries = store.get_by_tool("tool_a")
        assert len(a_entries) == 2
        b_entries = store.get_by_tool("tool_b")
        assert len(b_entries) == 1

    @pytest.mark.asyncio
    async def test_get_by_tool_unknown_returns_empty(self, store):
        assert store.get_by_tool("no_such_tool") == []

    @pytest.mark.asyncio
    async def test_get_all_preserves_insertion_order(self, store):
        await store.store_tool_result("first", {}, "1")
        await store.store_tool_result("second", {}, "2")
        await store.store_tool_result("third", {}, "3")

        all_entries = store.get_all()
        assert [e.tool_name for e in all_entries] == ["first", "second", "third"]

    @pytest.mark.asyncio
    async def test_index_persisted_and_reloaded(self, tmp_sessions):
        """Index survives store recreation (simulates process restart)."""
        store1 = SessionStore("sid-persist", base_path=tmp_sessions)
        store1.create()
        await store1.store_tool_result("search", {"q": "hello"}, "world")

        # New store instance for same session
        store2 = SessionStore("sid-persist", base_path=tmp_sessions)
        store2.create()  # loads existing index

        assert len(store2.get_all()) == 1
        assert store2.get_all()[0].tool_name == "search"


# ---------------------------------------------------------------------------
# SessionStore: grep (two-phase search)
# ---------------------------------------------------------------------------


class TestSessionStoreGrep:
    @pytest.mark.asyncio
    async def test_grep_finds_in_preview(self, store):
        await store.store_tool_result("search", {}, "The weather in London is sunny")

        matches = await store.grep("London")
        assert len(matches) == 1
        assert matches[0]["source"] == "preview"
        assert matches[0]["tool"] == "search"

    @pytest.mark.asyncio
    async def test_grep_case_insensitive_by_default(self, store):
        await store.store_tool_result("search", {}, "Python is great")

        matches = await store.grep("python")
        assert len(matches) == 1

    @pytest.mark.asyncio
    async def test_grep_case_sensitive(self, store):
        await store.store_tool_result("search", {}, "Python is great")

        matches = await store.grep("python", case_sensitive=True)
        assert len(matches) == 0

    @pytest.mark.asyncio
    async def test_grep_with_tool_filter(self, store):
        await store.store_tool_result("tool_a", {}, "target data here")
        await store.store_tool_result("tool_b", {}, "target data here")

        matches = await store.grep("target", tool_filter="tool_a")
        assert len(matches) == 1
        assert matches[0]["tool"] == "tool_a"

    @pytest.mark.asyncio
    async def test_grep_no_matches(self, store):
        await store.store_tool_result("search", {}, "some data")
        matches = await store.grep("nonexistent_xyz_pattern")
        assert matches == []

    @pytest.mark.asyncio
    async def test_grep_falls_back_to_full_file(self, store):
        """When preview is truncated, grep reads the full file."""
        # Create a result whose match content is beyond the preview length
        long_prefix = "x" * 400  # exceeds the 300-char preview
        result = long_prefix + "FINDME"
        await store.store_tool_result("tool", {}, result)

        matches = await store.grep("FINDME")
        assert len(matches) == 1
        assert matches[0]["source"] == "full"


# ---------------------------------------------------------------------------
# SessionStore: summary
# ---------------------------------------------------------------------------


class TestSessionStoreSummary:
    @pytest.mark.asyncio
    async def test_summary_empty(self, store):
        s = store.summary()
        assert s["total_entries"] == 0
        assert s["tools_used"] == []

    @pytest.mark.asyncio
    async def test_summary_with_data(self, store):
        await store.store_tool_result("search", {}, "r1")
        await store.store_tool_result("search", {}, "r2")
        await store.store_tool_result("weather", {}, "r3")

        s = store.summary()
        assert s["total_entries"] == 3
        assert set(s["tools_used"]) == {"search", "weather"}
        assert s["per_tool"]["search"]["count"] == 2
        assert s["per_tool"]["weather"]["count"] == 1


# ---------------------------------------------------------------------------
# SessionStore: safe tool names
# ---------------------------------------------------------------------------


class TestSessionStoreSafeNames:
    @pytest.mark.asyncio
    async def test_tool_name_with_slashes(self, store):
        entry = await store.store_tool_result(
            "mcp/remote/search", {"q": "test"}, "result"
        )
        assert "mcp_remote_search" in entry.relative_path

    @pytest.mark.asyncio
    async def test_tool_name_with_spaces(self, store):
        entry = await store.store_tool_result(
            "my tool", {"q": "test"}, "result"
        )
        assert "my_tool" in entry.relative_path


# ---------------------------------------------------------------------------
# SessionLifecycle
# ---------------------------------------------------------------------------


class TestSessionLifecycle:
    @pytest.mark.asyncio
    async def test_lifecycle_always_cleans_up_on_success(self, tmp_sessions):
        async with SessionLifecycle(
            "lc-1", CleanupPolicy.ALWAYS, base_path=tmp_sessions
        ) as store:
            await store.store_tool_result("search", {}, "data")
            assert (tmp_sessions / "lc-1").exists()

        # After exit: cleaned up
        assert not (tmp_sessions / "lc-1").exists()

    @pytest.mark.asyncio
    async def test_lifecycle_always_cleans_up_on_error(self, tmp_sessions):
        with pytest.raises(ValueError):
            async with SessionLifecycle(
                "lc-2", CleanupPolicy.ALWAYS, base_path=tmp_sessions
            ) as store:
                await store.store_tool_result("search", {}, "data")
                raise ValueError("boom")

        assert not (tmp_sessions / "lc-2").exists()

    @pytest.mark.asyncio
    async def test_lifecycle_on_error_keeps_files_on_success(self, tmp_sessions):
        async with SessionLifecycle(
            "lc-3", CleanupPolicy.ON_ERROR, base_path=tmp_sessions
        ) as store:
            await store.store_tool_result("search", {}, "data")

        # No error → files kept
        assert (tmp_sessions / "lc-3").exists()

    @pytest.mark.asyncio
    async def test_lifecycle_on_error_cleans_up_on_error(self, tmp_sessions):
        with pytest.raises(RuntimeError):
            async with SessionLifecycle(
                "lc-4", CleanupPolicy.ON_ERROR, base_path=tmp_sessions
            ) as store:
                await store.store_tool_result("search", {}, "data")
                raise RuntimeError("fail")

        assert not (tmp_sessions / "lc-4").exists()

    @pytest.mark.asyncio
    async def test_lifecycle_never_keeps_files_always(self, tmp_sessions):
        with pytest.raises(RuntimeError):
            async with SessionLifecycle(
                "lc-5", CleanupPolicy.NEVER, base_path=tmp_sessions
            ) as store:
                await store.store_tool_result("search", {}, "data")
                raise RuntimeError("fail")

        # NEVER policy: files survive even on error
        assert (tmp_sessions / "lc-5").exists()

    @pytest.mark.asyncio
    async def test_lifecycle_mark_error(self, tmp_sessions):
        lifecycle = SessionLifecycle(
            "lc-6", CleanupPolicy.ON_ERROR, base_path=tmp_sessions
        )
        store = await lifecycle.__aenter__()
        await store.store_tool_result("search", {}, "data")

        # Manually mark error (as the agent would on CancelledError)
        lifecycle.mark_error()
        await lifecycle.__aexit__(None, None, None)

        # ON_ERROR + error marked → cleaned up
        assert not (tmp_sessions / "lc-6").exists()


# ---------------------------------------------------------------------------
# IndexEntry serialisation
# ---------------------------------------------------------------------------


class TestIndexEntry:
    def test_round_trip(self):
        entry = IndexEntry(
            entry_id="search/001",
            tool_name="search",
            args_hash="abc12345",
            relative_path="tools/search/001_abc12345.json",
            created_at=1234567890.0,
            size_bytes=256,
            preview="some preview",
            metadata={"args": {"q": "test"}},
        )

        d = entry.to_dict()
        restored = IndexEntry.from_dict(d)

        assert restored.entry_id == entry.entry_id
        assert restored.tool_name == entry.tool_name
        assert restored.args_hash == entry.args_hash
        assert restored.preview == entry.preview
        assert restored.metadata == entry.metadata


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


class TestFactory:
    def test_create_session_store(self, tmp_sessions):
        store = create_session_store("fac-1", base_path=tmp_sessions)
        assert isinstance(store, SessionStore)
        assert store.session_id == "fac-1"
