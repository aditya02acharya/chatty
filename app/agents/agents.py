"""Agent factory functions.

Each function creates a focused Strands Agent with a single
responsibility. These agents are used as nodes in the chat graph.
"""

from strands import Agent
from strands.agent.conversation_manager import (
    NullConversationManager,
    SlidingWindowConversationManager,
)
from strands.agent.conversation_manager.conversation_manager import (
    ConversationManager,
)
from strands.hooks import HookProvider
from strands.models import BedrockModel

from app.agents.prompts import (
    researcher_prompt,
    responder_prompt,
    supervisor_prompt,
)


def create_supervisor(model: BedrockModel) -> Agent:
    """Lightweight agent that classifies queries.

    Returns exactly one word: ANSWER, CALL_TOOL, or CLARIFY.
    """
    return Agent(
        model=model,
        system_prompt=supervisor_prompt(),
        conversation_manager=NullConversationManager(),
    )


def create_responder(
    model: BedrockModel,
    tools: list | None = None,
) -> Agent:
    """Direct-response agent for the ANSWER / CLARIFY paths.

    Handles simple questions, greetings, and clarification requests.
    Optionally receives basic tools (e.g. get_current_time).
    """
    return Agent(
        model=model,
        tools=tools or [],
        system_prompt=responder_prompt(),
        conversation_manager=NullConversationManager(),
    )


def create_researcher(
    model: BedrockModel,
    tools: list | None = None,
    hooks: list[HookProvider] | None = None,
    conversation_manager: ConversationManager | None = None,
) -> Agent:
    """Research agent for the CALL_TOOL path.

    Uses full tool access, session filesystem, and the
    SmartAgentHook for result compaction.  Internally handles
    the tool-execute / reflect / need-more loop via the
    Strands Agent's native tool loop.

    When ``conversation_manager`` is provided (e.g. a
    PostgresConversationManager), it is used instead of the
    default SlidingWindowConversationManager.
    """
    if conversation_manager is None:
        conversation_manager = SlidingWindowConversationManager(
            window_size=100
        )
    return Agent(
        model=model,
        tools=tools or [],
        system_prompt=researcher_prompt(),
        conversation_manager=conversation_manager,
        hooks=hooks or [],
    )
