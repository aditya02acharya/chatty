"""
Tool wrapper with pre/post hooks for metadata tracking.

Provides a wrapper for local tools that integrates with strands hooks system
for tracking token cost, latency, and other metadata.
"""

import asyncio
import time
from collections.abc import Callable

from strands import ToolContext
from strands.hooks import (
    AfterToolCallEvent,
    BeforeToolCallEvent,
    HookRegistry,
)


class ToolCallMetadata:
    """Metadata tracked for tool calls via pre/post hooks."""

    tool_name: str
    start_time: float
    end_time: float | None
    latency_ms: float | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    error: str | None
    is_remote: bool
    server_name: str | None

    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        self.start_time = time.time()
        self.end_time = None
        self.latency_ms = None
        self.input_tokens = None
        self.output_tokens = None
        self.total_tokens = None
        self.error = None
        self.is_remote = False
        self.server_name = None

    def finish(self, error: str | None = None) -> None:
        """Mark the tool call as finished."""
        self.end_time = time.time()
        self.latency_ms = (self.end_time - self.start_time) * 1000
        self.error = error


class ToolWrapper:
    """Wrapper for tools with pre/post hooks.

    Automatically tracks metadata and executes registered hooks.
    """

    _func: Callable
    _name: str
    _description: str | None
    _hook_registry: HookRegistry | None
    _metadata: ToolCallMetadata | None

    def __init__(
        self,
        func: Callable,
        name: str | None = None,
        description: str | None = None,
        hook_registry: HookRegistry | None = None,
    ):
        """Initialize tool wrapper.

        Args:
            func: Tool function to wrap
            name: Optional tool name (defaults to function name)
            description: Optional tool description
            hook_registry: Optional hook registry for pre/post hooks
        """
        self._func = func
        self._name = name or func.__name__
        self._description = description or func.__doc__
        self._hook_registry = hook_registry
        self._metadata = None

    @property
    def name(self) -> str:
        """Tool name."""
        return self._name

    @property
    def description(self) -> str | None:
        """Tool description."""
        return self._description

    @property
    def metadata(self) -> ToolCallMetadata | None:
        """Get metadata from the last call."""
        return self._metadata

    async def __call__(self, *args, **kwargs):
        """Call the wrapped tool with hooks."""
        # Create metadata tracker
        self._metadata = ToolCallMetadata(self._name)

        # Extract ToolContext if present
        tool_context = None
        if args and isinstance(args[0], ToolContext):
            tool_context = args[0]

        # Pre-hook
        if self._hook_registry:
            await self._hook_registry.execute(
                BeforeToolCallEvent(
                    tool_name=self._name,
                    tool_args=kwargs,
                    tool_context=tool_context,
                )
            )

        try:
            # Call the actual function
            if asyncio.iscoroutinefunction(self._func):
                result = await self._func(*args, **kwargs)
            else:
                result = self._func(*args, **kwargs)

            self._metadata.finish()

            # Post-hook (success)
            if self._hook_registry:
                await self._hook_registry.execute(
                    AfterToolCallEvent(
                        tool_name=self._name,
                        tool_args=kwargs,
                        result=result,
                        tool_context=tool_context,
                    )
                )

            return result
        except Exception as e:
            self._metadata.finish(error=str(e))

            # Post-hook (error)
            if self._hook_registry:
                await self._hook_registry.execute(
                    AfterToolCallEvent(
                        tool_name=self._name,
                        tool_args=kwargs,
                        result=None,
                        error=e,
                        tool_context=tool_context,
                    )
                )
            raise


def with_hooks(
    name: str | None = None,
    description: str | None = None,
    hook_registry: HookRegistry | None = None,
) -> Callable:
    """Decorator to add hooks and metadata tracking to a tool function.

    Args:
        name: Optional tool name (defaults to function name)
        description: Optional tool description
        hook_registry: Hook registry for pre/post hooks

    Returns:
        Decorator function

    Example:
        from strands.hooks import HookRegistry

        hooks = HookRegistry()

        @with_hooks(name="get_time", hook_registry=hooks)
        async def get_current_time():
            return time.time()
    """
    def decorator(func):
        return ToolWrapper(
            func,
            name=name,
            description=description,
            hook_registry=hook_registry,
        )
    return decorator
