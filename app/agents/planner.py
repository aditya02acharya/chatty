"""
Tool-call planning with dependency analysis.

Plans which tools to invoke and in what order, supporting both
FAST (minimal tools) and AGENTIC (thorough, multi-step) strategies.
"""

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field
from strands.models import BedrockModel

from app.agents.execution_mode import ExecutionMode


@dataclass
class ToolCall:
    """A tool call with metadata."""

    name: str
    args: dict[str, Any]
    dependencies: list[str]  # Names of tools this depends on
    independent: bool = True
    estimated_duration: float = 1.0  # seconds


class ToolCallPlanner(BaseModel):
    """Plans tool execution with dependency analysis."""

    model: BedrockModel = Field(description="Model for planning")

    model_config = {"arbitrary_types_allowed": True}

    async def plan_execution(
        self,
        query: str,
        available_tools: list[str],
        mode: ExecutionMode,
    ) -> list[ToolCall]:
        """Plan tool execution with dependency analysis."""
        if mode == ExecutionMode.FAST:
            return await self._plan_fast(query, available_tools)
        return await self._plan_agentic(query, available_tools)

    async def _plan_fast(
        self, query: str, available_tools: list[str]
    ) -> list[ToolCall]:
        prompt = f"""Does this query require multiple tools?

Query: "{query}"
Available tools: {", ".join(available_tools[:10])}

Return JSON:
{{
    "tools_needed": [],
    "reasoning": "..."
}}

If only 1 tool or no tools needed, return that.
If multiple explicitly requested
(e.g. "compare X and Y"), return all.
"""

        try:
            response = await self.model.structured_output_async(
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                system_prompt="Identify required tools.",
            )
            result = response.content
            tools_requested = result.get("tools_needed", [])
            return [
                ToolCall(name=t, args={}, dependencies=[], independent=True)
                for t in tools_requested[:3]
            ]
        except Exception:
            return []

    async def _plan_agentic(
        self, query: str, available_tools: list[str]
    ) -> list[ToolCall]:
        tools_str = "\n".join(f"- {t}" for t in available_tools)

        prompt = f"""Plan the execution for this query:

Query: "{query}"

Available tools:
{tools_str}

Create a plan. Return JSON:
{{
    "steps": [
        {{"tool": "tool_name", "args": {{...}},
          "depends_on": [], "reasoning": "..."}},
        ...
    ]
}}

Rules:
- Tools with no dependencies can run in parallel
- Tools that depend on other tool results must run after
- Be thorough - use multiple sources to verify information
"""

        try:
            response = await self.model.structured_output_async(
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                system_prompt=(
                    "You are a planning assistant."
                    " Create detailed plans."
                ),
            )
            result = response.content
            steps = result.get("steps", [])

            return [
                ToolCall(
                    name=s["tool"],
                    args=s.get("args", {}),
                    dependencies=s.get("depends_on", []),
                    independent=len(s.get("depends_on", [])) == 0,
                )
                for s in steps
            ]
        except Exception:
            return []

    def group_for_execution(
        self, calls: list[ToolCall]
    ) -> list[list[ToolCall]]:
        """Group tool calls into waves for optimal execution.

        Each wave contains calls whose dependencies are already satisfied.
        Calls within a wave can run in parallel.
        """
        groups: list[list[ToolCall]] = []
        completed: set[str] = set()
        remaining = list(calls)

        while remaining:
            ready = [
                c
                for c in remaining
                if all(dep in completed for dep in c.dependencies)
            ]

            if not ready:
                # Circular dependency or unresolvable
                # — run remaining sequentially
                for c in remaining:
                    groups.append([c])
                break

            groups.append(ready)
            completed.update(c.name for c in ready)
            remaining = [c for c in remaining if c not in ready]

        return groups
