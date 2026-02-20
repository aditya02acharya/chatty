"""
Modular Pydantic settings configuration.

Settings are loaded from:
1. Default values in the Pydantic model
2. YAML config files (config/settings.yaml, config/settings.{env}.yaml)
3. Environment variables (highest priority)

Usage:
    from app.core.config import settings
    print(settings.bedrock.region)
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import ClassVar, Literal

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _load_yaml_config() -> dict:
    """Load and merge YAML configuration files.

    Returns:
        Merged configuration dictionary from all YAML files.
    """
    config_dir = Path(__file__).parent.parent.parent / "config"
    merged_config: dict = {}

    env = os.getenv("ENVIRONMENT", "dev").lower()
    if env not in ("dev", "prod", "test"):
        env = "dev"

    base_path = config_dir / "settings.yaml"
    if base_path.exists():
        with open(base_path) as f:
            base_config = yaml.safe_load(f) or {}
            merged_config.update(base_config)

    env_path = config_dir / f"settings.{env}.yaml"
    if env_path.exists():
        with open(env_path) as f:
            env_config = yaml.safe_load(f) or {}
            for key, value in env_config.items():
                if key in merged_config and isinstance(
                    merged_config[key], dict
                ):
                    merged_config[key].update(value)
                else:
                    merged_config[key] = value

    return merged_config


# Load YAML config once at module import
_yaml_config = _load_yaml_config()


class YamlSettings(BaseSettings):
    """Base settings class that merges YAML config values.

    Subclasses set ``_yaml_section`` to the top-level YAML key
    whose values should seed the Pydantic fields.  This avoids
    duplicating the same ``__init__`` override in every settings
    class.
    """

    _yaml_section: ClassVar[str] = ""

    def __init__(self, **kwargs):
        yaml_values = _yaml_config.get(self._yaml_section, {})
        for key, value in yaml_values.items():
            if key not in kwargs:
                kwargs[key] = value
        super().__init__(**kwargs)


class ServerSettings(YamlSettings):
    """Server configuration settings."""

    _yaml_section: ClassVar[str] = "server"

    host: str = Field(
        default="0.0.0.0",
        description="Server host address",
    )
    port: int = Field(
        default=8000,
        description="Server port",
        ge=1,
        le=65535,
    )
    workers: int = Field(
        default=4,
        description="Number of worker processes",
        ge=1,
    )
    max_concurrent_requests: int = Field(
        default=2000,
        description="Max concurrent requests",
        ge=1,
    )
    keepalive_timeout: int = Field(
        default=75,
        description="Keep-alive timeout",
        ge=1,
    )
    graceful_shutdown_timeout: int = Field(
        default=30,
        description="Graceful shutdown timeout",
        ge=1,
    )

    model_config = SettingsConfigDict(
        env_prefix="SERVER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class BedrockSettings(YamlSettings):
    """AWS Bedrock configuration settings."""

    _yaml_section: ClassVar[str] = "bedrock"

    region: str = Field(
        default="us-east-1",
        description="AWS region",
    )
    profile: str | None = Field(
        default=None,
        description="AWS profile name",
    )
    default_model: str = Field(
        default="anthropic.claude-sonnet-4-20250514",
        description="Default model ID",
    )
    models: dict[str, str] = Field(
        default_factory=lambda: {
            "claude": "anthropic.claude-sonnet-4-20250514",
            "llama": "meta.llama-3-1-405b-instruct",
            "mistral": "mistral.mistral-large-2407",
        },
        description="Model name to ID mapping",
    )
    auth_method: Literal["iam", "api_key"] = Field(
        default="iam",
        description="Authentication method",
    )
    timeout: int = Field(
        default=60,
        description="Request timeout in seconds",
        ge=1,
    )
    max_retries: int = Field(
        default=3,
        description="Maximum retry attempts",
        ge=0,
    )
    inference_config: dict = Field(
        default_factory=lambda: {
            "max_tokens": 4096,
            "temperature": 0.7,
            "top_p": 0.9,
            "stop_sequences": [],
        },
        description="Default inference parameters",
    )

    model_config = SettingsConfigDict(
        env_prefix="BEDROCK_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class MCPSettings(YamlSettings):
    """MCP (Model Context Protocol) configuration settings."""

    _yaml_section: ClassVar[str] = "mcp"

    timeout: int = Field(
        default=30,
        description="MCP request timeout",
        ge=1,
    )
    max_retries: int = Field(
        default=2,
        description="Maximum retry attempts",
        ge=0,
    )
    endpoints: list[dict] = Field(
        default_factory=list,
        description="Remote MCP endpoints configuration",
    )

    model_config = SettingsConfigDict(
        env_prefix="MCP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class ToolDiscoverySettings(BaseSettings):
    """Configuration for MCP-based tool discovery."""

    enabled: bool = Field(
        default=False,
        description="Enable dynamic tool discovery via MCP tool",
    )
    tool_name: str = Field(
        default="find_tools",
        description="Name of the MCP tool that performs discovery",
    )
    server_name: str | None = Field(
        default=None,
        description="MCP server hosting the discovery tool (auto-detect if None)",
    )
    top_k: int = Field(
        default=10,
        description="Maximum number of tools to return",
        ge=1,
        le=50,
    )

    model_config = SettingsConfigDict(
        env_prefix="TOOL_DISCOVERY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class AgentSettings(YamlSettings):
    """Agent behavior configuration settings."""

    _yaml_section: ClassVar[str] = "agent"

    max_iterations: int = Field(
        default=10,
        description="Maximum agent loop iterations",
        ge=1,
    )
    tool_timeout: int = Field(
        default=30,
        description="Tool execution timeout",
        ge=1,
    )
    reflection_enabled: bool = Field(
        default=True,
        description="Enable reflection after tool use",
    )
    system_prompt: str = Field(
        default=(
            "You are a helpful AI assistant with access to "
            "various tools.\n"
            "Use tools when needed to provide accurate and "
            "helpful responses.\n"
            "Always explain your reasoning when using tools."
        ),
        description="System prompt for the agent",
    )
    tool_discovery: ToolDiscoverySettings = Field(
        default_factory=ToolDiscoverySettings,
        description="Tool discovery configuration",
    )

    model_config = SettingsConfigDict(
        env_prefix="AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class AGUISettings(YamlSettings):
    """ag-ui protocol configuration settings."""

    _yaml_section: ClassVar[str] = "agui"

    buffer_size: int = Field(
        default=8192,
        description="Event buffer size for SSE",
        ge=1024,
    )
    heartbeat_interval: int = Field(
        default=15,
        description="Heartbeat interval for SSE connections",
        ge=1,
    )
    event_types: list[str] = Field(
        default_factory=lambda: [
            "thinking",
            "decided",
            "tool_call",
            "tool_result",
            "content",
            "done",
            "error",
        ],
        description="Supported event types",
    )

    model_config = SettingsConfigDict(
        env_prefix="AGUI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class LogSettings(YamlSettings):
    """Logging configuration settings."""

    _yaml_section: ClassVar[str] = "logging"

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Log level"
    )
    format: Literal["json", "text"] = Field(
        default="json",
        description="Log format",
    )
    include_timestamp: bool = Field(
        default=True,
        description="Include timestamp",
    )
    include_caller: bool = Field(
        default=True,
        description="Include caller info",
    )
    telemetry_enabled: bool = Field(
        default=False,
        description="Enable telemetry",
    )

    model_config = SettingsConfigDict(
        env_prefix="LOG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class Settings:
    """Central settings container aggregating all modules.

    Usage:
        from app.core.config import settings

        print(settings.server.host)
        print(settings.bedrock.region)
    """

    def __init__(self):
        self.server = ServerSettings()
        self.bedrock = BedrockSettings()
        self.mcp = MCPSettings()
        self.agent = AgentSettings()
        self.agui = AGUISettings()
        self.log = LogSettings()

    def reload(self) -> None:
        """Reload all settings from config files."""
        global _yaml_config
        _yaml_config = _load_yaml_config()
        self.__init__()


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Returns:
        Settings: Cached settings instance
    """
    return Settings()


# Global settings instance
settings = get_settings()
