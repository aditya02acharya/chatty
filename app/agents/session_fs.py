"""
Session filesystem manager for agentic mode.

Acts as working memory/scratchpad for the agent:
- Stores tool results in session folders
- Maintains metadata for each file
- Enables grep/search across stored data
- Preserves large context without token limits
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


@dataclass
class StoredFile:
    """Metadata for a file stored in session."""

    filename: str
    source_tool: str
    source_args: dict[str, Any]
    created_at: float
    size_bytes: int
    content_type: str  # json, text, binary, etc.
    metadata: dict[str, Any] = field(default_factory=dict)


class SessionFileSystem(BaseModel):
    """Manages filesystem storage for a session."""

    session_id: str = Field(description="Unique session identifier")
    base_path: Path = Field(
        default_factory=lambda: Path("sessions"),
        description="Base directory for sessions",
    )
    metadata_file: str = Field(default=".metadata.json", description="Metadata filename")

    model_config = {"populate_by_name": True}

    @property
    def session_path(self) -> Path:
        """Get the filesystem path for this session."""
        return self.base_path / self.session_id

    def create_session(self) -> None:
        """Create the session directory."""
        self.session_path.mkdir(parents=True, exist_ok=True)

    async def store_tool_result(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        result: Any,
        filename: str | None = None,
    ) -> str:
        """Store a tool result in the session filesystem.

        Args:
            tool_name: Name of the tool
            tool_args: Arguments passed to tool
            result: Tool result
            filename: Optional custom filename

        Returns:
            Path to stored file (relative to session directory)
        """
        self.create_session()

        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = tool_name.replace("/", "_").replace("\\", "_")
            filename = f"{timestamp}_{safe_name}.json"

        filepath = self.session_path / filename

        # Prepare data with metadata
        data = {
            "tool": tool_name,
            "args": tool_args,
            "result": result,
            "timestamp": time.time(),
            "iso_timestamp": datetime.now().isoformat(),
        }

        # Write to file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Update metadata index
        await self._update_metadata(
            StoredFile(
                filename=filename,
                source_tool=tool_name,
                source_args=tool_args,
                created_at=time.time(),
                size_bytes=filepath.stat().st_size,
                content_type="json",
                metadata={"args": tool_args},
            )
        )

        return filename

    async def _update_metadata(self, file_info: StoredFile) -> None:
        """Update the session metadata index.

        Args:
            file_info: File metadata to record
        """
        metadata_path = self.session_path / self.metadata_file

        # Load existing metadata
        files = []
        if metadata_path.exists():
            with open(metadata_path) as f:
                data = json.load(f)
                files = data.get("files", [])

        # Add new file
        files.append({
            "filename": file_info.filename,
            "source_tool": file_info.source_tool,
            "created_at": file_info.created_at,
            "size_bytes": file_info.size_bytes,
            "metadata": file_info.metadata,
        })

        # Save metadata
        with open(metadata_path, "w") as f:
            json.dump({"files": files}, f, indent=2)

    async def list_files(self) -> list[dict[str, Any]]:
        """List all files in the session.

        Returns:
            List of file metadata
        """
        metadata_path = self.session_path / self.metadata_file

        if not metadata_path.exists():
            return []

        with open(metadata_path) as f:
            data = json.load(f)
            return data.get("files", [])

    async def read_file(self, filename: str) -> dict[str, Any]:
        """Read a stored file.

        Args:
            filename: Name of the file to read

        Returns:
            File contents (parsed JSON)
        """
        filepath = self.session_path / filename

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filename}")

        with open(filepath, encoding="utf-8") as f:
            return json.load(f)

    async def grep(
        self,
        pattern: str,
        case_sensitive: bool = False,
        file_filter: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for pattern across all files in session.

        Args:
            pattern: Search pattern (regex)
            case_sensitive: Whether search is case sensitive
            file_filter: Optional list of filenames to search

        Returns:
            List of matches with context
        """
        import re

        flags = 0 if case_sensitive else re.IGNORECASE
        regex = re.compile(pattern, flags)

        matches = []
        files = await self.list_files()

        for file_info in files:
            if file_filter and file_info["filename"] not in file_filter:
                continue

            try:
                content = await self.read_file(file_info["filename"])
                content_str = json.dumps(content, indent=2)

                for match in regex.finditer(content_str):
                    # Get context around match
                    start = max(0, match.start() - 100)
                    end = min(len(content_str), match.end() + 100)
                    context = content_str[start:end]

                    matches.append({
                        "filename": file_info["filename"],
                        "tool": file_info["source_tool"],
                        "match": match.group(0),
                        "context": context,
                        "position": match.start(),
                    })
            except Exception:
                continue

        return matches

    async def get_summary(self) -> dict[str, Any]:
        """Get a summary of the session filesystem.

        Returns:
            Summary statistics and information
        """
        files = await self.list_files()

        total_size = sum(f.get("size_bytes", 0) for f in files)
        tools_used = list(set(f.get("source_tool", "unknown") for f in files))

        return {
            "session_id": self.session_id,
            "total_files": len(files),
            "total_size_bytes": total_size,
            "tools_used": tools_used,
            "files": files,
        }

    def cleanup(self) -> None:
        """Clean up the session filesystem."""
        import shutil

        if self.session_path.exists():
            shutil.rmtree(self.session_path)


class AgentMode(BaseModel):
    """Agent execution mode."""

    mode: str = Field(description="Mode: fast or agentic")
    max_tools_fast: int = Field(default=1, description="Max tool calls in fast mode")
    allow_parallel_fast: bool = Field(default=False, description="Allow parallel tools in fast mode")
    filesystem_enabled: bool = Field(default=True, description="Enable filesystem in agentic mode")

    model_config = {"populate_by_name": True}


def create_session_filesystem(session_id: str) -> SessionFileSystem:
    """Factory to create a session filesystem.

    Args:
        session_id: Unique session identifier

    Returns:
        SessionFileSystem instance
    """
    return SessionFileSystem(session_id=session_id)
