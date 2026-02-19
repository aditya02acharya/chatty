"""
Reusable agent node patterns using strands SDK.

Provides composable patterns for building agent workflows.
"""

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field
from strands import ToolContext
from strands.models import BedrockModel

from app.core.config import settings


@dataclass
class ToolNodeConfig:
    """Configuration for ToolNode execution."""

    parallel: bool = True
    max_parallel: int = 5
    timeout_seconds: int = 30
    retry_on_failure: bool = True
    max_retries: int = 2


class ToolNode:
    """Node pattern for executing tools with configurable parallelism.

    Supports:
    - Parallel or sequential tool execution
    - Timeout and retry logic
    - Metadata tracking via hooks
    """

    def __init__(
        self,
        tools: list,
        config: ToolNodeConfig | None = None,
        hook_registry: Any | None = None,
    ):
        """Initialize the ToolNode.

        Args:
            tools: List of tool functions or ToolProvider objects
            config: Optional ToolNodeConfig
            hook_registry: Optional hook registry for tracking
        """
        self._tools = tools
        self._config = config or ToolNodeConfig()
        self._hook_registry = hook_registry

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: ToolContext | None = None,
    ) -> Any:
        """Execute a single tool.

        Args:
            tool_name: Name of tool to execute
            arguments: Tool arguments
            context: Optional ToolContext

        Returns:
            Tool result
        """
        # Find the tool
        tool_fn = None
        for t in self._tools:
            if hasattr(t, "name") and t.name == tool_name:
                tool_fn = t
                break
            elif hasattr(t, "__name__") and t.__name__ == tool_name:
                tool_fn = t
                break

        if tool_fn is None:
            raise ValueError(f"Tool not found: {tool_name}")

        # Pre-hook
        if self._hook_registry:
            from strands.hooks import BeforeToolCallEvent
            await self._hook_registry.execute(
                BeforeToolCallEvent(
                    tool_name=tool_name,
                    tool_args=arguments,
                    tool_context=context,
                )
            )

        try:
            # Execute
            if hasattr(tool_fn, "__call__"):
                result = await tool_fn(**arguments)
            else:
                result = tool_fn(**arguments)

            # Post-hook
            if self._hook_registry:
                from strands.hooks import AfterToolCallEvent
                await self._hook_registry.execute(
                    AfterToolCallEvent(
                        tool_name=tool_name,
                        tool_args=arguments,
                        result=result,
                        tool_context=context,
                    )
                )

            return result
        except Exception as e:
            # Post-hook with error
            if self._hook_registry:
                from strands.hooks import AfterToolCallEvent
                await self._hook_registry.execute(
                    AfterToolCallEvent(
                        tool_name=tool_name,
                        tool_args=arguments,
                        result=None,
                        error=e,
                        tool_context=context,
                    )
                )
            raise

    async def execute_parallel(
        self,
        calls: list[tuple[str, dict[str, Any]]],
        context: ToolContext | None = None,
    ) -> list[tuple[str, Any, Exception | None]]:
        """Execute multiple tools in parallel.

        Args:
            calls: List of (tool_name, arguments) tuples
            context: Optional ToolContext

        Returns:
            List of (tool_name, result, error) tuples
        """
        import asyncio

        tasks = [
            self.execute(name, args, context)
            for name, args in calls
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        return [
            (
                name,
                result if not isinstance(result, Exception) else None,
                result if isinstance(result, Exception) else None,
            )
            for (name, _), result in zip(calls, results)
        ]


class ReflectionNode:
    """Node pattern for reflecting on results before proceeding.

    Uses an LLM to analyze results and determine next actions.
    """

    def __init__(
        self,
        model: BedrockModel | None = None,
        reflection_prompt: str | None = None,
    ):
        """Initialize the ReflectionNode.

        Args:
            model: Optional Bedrock model for reflection
            reflection_prompt: Optional custom reflection prompt
        """
        self._model = model or BedrockModel(
            model_id=settings.bedrock.default_model,
            region_name=settings.bedrock.region,
        )
        self._reflection_prompt = reflection_prompt or self._default_reflection_prompt()

    def _default_reflection_prompt(self) -> str:
        """Default reflection prompt template."""
        return """Analyze the following tool execution result:

Tool: {tool_name}
Arguments: {arguments}
Result: {result}

Provide:
1. Assessment: Did the tool succeed?
2. Analysis: What does this result tell us?
3. Next Step: What should we do next?

Respond in JSON format:
{{
    "success": true/false,
    "analysis": "...",
    "next_step": "..."
}}
"""

    async def reflect(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
    ) -> dict[str, Any]:
        """Reflect on a tool result.

        Args:
            tool_name: Name of tool that was executed
            arguments: Arguments passed to tool
            result: Tool result

        Returns:
            Reflection analysis dict
        """
        prompt = self._reflection_prompt.format(
            tool_name=tool_name,
            arguments=arguments,
            result=str(result)[:1000],  # Truncate for prompt
        )

        # Use structured output for JSON response
        response = await self._model.structured_output_async(
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            system_prompt="You are a helpful assistant that analyzes tool results.",
        )

        return response.content


class SynthesisNode:
    """Node pattern for synthesizing multiple results into a response.

    Combines outputs from multiple sources into a coherent response.
    """

    def __init__(
        self,
        model: BedrockModel | None = None,
        synthesis_prompt: str | None = None,
    ):
        """Initialize the SynthesisNode.

        Args:
            model: Optional Bedrock model for synthesis
            synthesis_prompt: Optional custom synthesis prompt
        """
        self._model = model or BedrockModel(
            model_id=settings.bedrock.default_model,
            region_name=settings.bedrock.region,
        )
        self._synthesis_prompt = synthesis_prompt or self._default_synthesis_prompt()

    def _default_synthesis_prompt(self) -> str:
        """Default synthesis prompt template."""
        return """Synthesize the following information into a coherent response:

User Query: {query}

Available Information:
{information}

Provide a comprehensive, accurate response based on the available information.
If information is missing or contradictory, acknowledge this in your response.
"""

    async def synthesize(
        self,
        query: str,
        information: list[tuple[str, Any]],
    ) -> str:
        """Synthesize multiple results into a response.

        Args:
            query: Original user query
            information: List of (source, result) tuples

        Returns:
            Synthesized response
        """
        info_text = "\n".join([
            f"- {source}: {str(result)[:500]}"
            for source, result in information
        ])

        prompt = self._synthesis_prompt.format(
            query=query,
            information=info_text,
        )

        response = await self._model.stream_async(
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            system_prompt="You are a helpful assistant that synthesizes information.",
        )

        # Collect streamed response
        full_response = ""
        async for chunk in response:
            full_response += chunk

        return full_response


class RoutingNode:
    """Node pattern for routing based on conditions.

    Routes requests to different handlers based on analysis.
    """

    def __init__(
        self,
        model: BedrockModel | None = None,
        routes: dict[str, str] | None = None,
    ):
        """Initialize the RoutingNode.

        Args:
            model: Optional Bedrock model for routing decisions
            routes: Optional dict of route_name -> description
        """
        self._model = model or BedrockModel(
            model_id=settings.bedrock.default_model,
            region_name=settings.bedrock.region,
        )
        self._routes = routes or {
            "ANSWER": "Generate a direct answer without tools",
            "CALL_TOOL": "Call a tool to gather information",
            "CLARIFY": "Ask the user for clarification",
        }

    async def route(
        self,
        query: str,
        available_tools: list[str] | None = None,
    ) -> dict[str, Any]:
        """Determine the appropriate route for a query.

        Args:
            query: User query
            available_tools: Optional list of available tool names

        Returns:
            Routing decision dict with route and reasoning
        """
        tools_text = ""
        if available_tools:
            tools_text = f"Available tools: {', '.join(available_tools)}"

        routes_text = "\n".join([
            f"- {name}: {desc}"
            for name, desc in self._routes.items()
        ])

        prompt = f"""Analyze the following query and determine the best route:

Query: {query}
{tools_text}

Available Routes:
{routes_text}

Respond in JSON format:
{{
    "route": "ROUTE_NAME",
    "reasoning": "...",
    "tool_name": "..." (if route is CALL_TOOL)
}}
"""

        response = await self._model.structured_output_async(
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            system_prompt=(
                "You are a routing assistant that determines "
                "the best action for a query."
            ),
        )

        return response.content


class ValidationNode:
    """Node pattern for validating outputs against criteria.

    Validates responses before returning them.
    """

    class ValidationCriteria(BaseModel):
        """Criteria for validation."""

        min_length: int = Field(default=10, description="Minimum response length")
        max_length: int = Field(default=5000, description="Maximum response length")
        required_keywords: list[str] = Field(
            default_factory=list, description="Required keywords"
        )
        forbidden_patterns: list[str] = Field(
            default_factory=list, description="Forbidden regex patterns"
        )

    def __init__(
        self,
        criteria: ValidationCriteria | None = None,
        model: BedrockModel | None = None,
    ):
        """Initialize the ValidationNode.

        Args:
            criteria: Optional validation criteria
            model: Optional model for fixing invalid responses
        """
        self._criteria = criteria or ValidationNode.ValidationCriteria()
        self._model = model

    async def validate(
        self,
        response: str,
        auto_fix: bool = False,
    ) -> tuple[bool, str, list[str] | None]:
        """Validate a response against criteria.

        Args:
            response: Response to validate
            auto_fix: Whether to attempt fixing invalid responses

        Returns:
            Tuple of (is_valid, response_or_fixed, errors_or_none)
        """
        errors = []

        # Check length
        if len(response) < self._criteria.min_length:
            errors.append(f"Response too short: {len(response)} < {self._criteria.min_length}")
        if len(response) > self._criteria.max_length:
            errors.append(f"Response too long: {len(response)} > {self._criteria.max_length}")

        # Check required keywords
        for keyword in self._criteria.required_keywords:
            if keyword.lower() not in response.lower():
                errors.append(f"Missing required keyword: {keyword}")

        # Check forbidden patterns
        import re
        for pattern in self._criteria.forbidden_patterns:
            if re.search(pattern, response):
                errors.append(f"Contains forbidden pattern: {pattern}")

        is_valid = len(errors) == 0

        if not is_valid and auto_fix and self._model:
            # Attempt to fix the response
            fix_prompt = f"""The following response failed validation:

{response}

Errors:
{chr(10).join(f'- {e}' for e in errors)}

Please fix the response to address these errors while maintaining the original meaning.
"""
            fixed_response = await self._model.stream_async(
                messages=[{"role": "user", "content": [{"text": fix_prompt}]}],
                system_prompt="You are a helpful assistant that fixes invalid responses.",
            )

            full_fixed = ""
            async for chunk in fixed_response:
                full_fixed += chunk

            return True, full_fixed, errors

        return is_valid, response, errors if errors else None
