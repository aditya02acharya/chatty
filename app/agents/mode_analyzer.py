"""
Mode analysis for automatic execution mode selection.

Analyzes user queries to determine whether FAST or AGENTIC mode
is more appropriate, using an LLM call with a heuristic fallback.
"""

from typing import Any

from pydantic import BaseModel, Field
from strands.models import BedrockModel

from app.agents.execution_mode import ExecutionMode


class ModeAnalyzer(BaseModel):
    """Analyzes query to determine appropriate mode."""

    model: BedrockModel = Field(description="Model for analysis")

    model_config = {"arbitrary_types_allowed": True}

    async def analyze_query(
        self, query: str
    ) -> tuple[ExecutionMode, dict[str, Any]]:
        """Analyze query to determine mode and strategy."""
        prompt = f"""Analyze this query and determine the best mode:

Query: "{query}"

Consider:
- FAST mode: Direct questions, simple lookups, quick answers
- AGENTIC mode: Research, multi-step investigation, data synthesis

Return JSON:
{{
    "mode": "fast|agentic",
    "reasoning": "...",
    "estimated_tools": 0-10,
    "requires_research": true/false,
    "complexity": "low|medium|high"
}}
"""

        try:
            response = await self.model.structured_output_async(
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                system_prompt=(
                    "You analyze queries to"
                    " determine processing strategy."
                ),
            )
            result = response.content

            mode_str = result.get("mode", "fast").lower()
            mode = (
                ExecutionMode.AGENTIC
                if mode_str == "agentic"
                else ExecutionMode.FAST
            )

            return mode, result
        except Exception:
            # Fallback: simple heuristic
            query_lower = query.lower()
            keywords = [
                "research",
                "investigate",
                "analyze",
                "compare",
                "detailed",
                "comprehensive",
            ]
            if any(kw in query_lower for kw in keywords):
                return ExecutionMode.AGENTIC, {
                    "reasoning": "keyword_match",
                    "complexity": "high",
                }
            return ExecutionMode.FAST, {
                "reasoning": "default",
                "complexity": "low",
            }
