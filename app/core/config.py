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
from typing import Literal

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _load_yaml_config() -> dict:
    """Load and merge YAML configuration files.

    Returns:
        Merged configuration dictionary from all YAML files.
    """
    config_dir = Path(__file__).parent.parent.parent / "config"
    base_config = {}
    merged_config = {}

    # Determine environment
    env = os.getenv("ENVIRONMENT", "dev").lower()
    if env not in ("dev", "prod", "test"):
        env = "dev"

    # Load base config first
    base_path = config_dir / "settings.yaml"
    if base_path.exists():
        with open(base_path) as f:
            base_config = yaml.safe_load(f) or {}
            merged_config.update(base_config)

    # Override with environment-specific config
    env_path = config_dir / f"settings.{env}.yaml"
    if env_path.exists():
        with open(env_path) as f:
            env_config = yaml.safe_load(f) or {}
            # Deep merge environment config
            for key, value in env_config.items():
                if key in merged_config and isinstance(merged_config[key], dict):
                    merged_config[key].update(value)
                else:
                    merged_config[key] = value

    return merged_config


# Load YAML config once at module import
_yaml_config = _load_yaml_config()


def _get_nested_value(config: dict, path: str, default=None):
    """Get a nested value from config dict using dot notation.

    Args:
        config: Configuration dictionary
        path: Dot-separated path (e.g., 'bedrock.region')
        default: Default value if path not found

    Returns:
        Value at path or default
    """
    keys = path.split(".")
    value = config
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return default
    return value if value is not None else default


class ServerSettings(BaseSettings):
    """Server configuration settings."""

    host: str = Field(default="0.0.0.0", description="Server host address")
    port: int = Field(default=8000, description="Server port", ge=1, le=65535)
    workers: int = Field(default=4, description="Number of worker processes", ge=1)
    max_concurrent_requests: int = Field(
        default=2000, description="Max concurrent requests", ge=1
    )
    keepalive_timeout: int = Field(default=75, description="Keep-alive timeout", ge=1)
    graceful_shutdown_timeout: int = Field(
        default=30, description="Graceful shutdown timeout", ge=1
    )

    model_config = SettingsConfigDict(
        env_prefix="SERVER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def __init__(self, **kwargs):
        # Initialize with YAML defaults if not provided
        yaml_values = _yaml_config.get("server", {})
        for key, value in yaml_values.items():
            if key not in kwargs:
                kwargs[key] = value
        super().__init__(**kwargs)


class BedrockSettings(BaseSettings):
    """AWS Bedrock configuration settings."""

    region: str = Field(default="us-east-1", description="AWS region")
    profile: str | None = Field(default=None, description="AWS profile name")
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
        default="iam", description="Authentication method"
    )
    timeout: int = Field(default=60, description="Request timeout in seconds", ge=1)
    max_retries: int = Field(default=3, description="Maximum retry attempts", ge=0)
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

    def __init__(self, **kwargs):
        yaml_values = _yaml_config.get("bedrock", {})
        for key, value in yaml_values.items():
            if key not in kwargs:
                kwargs[key] = value
        super().__init__(**kwargs)


class MCPSettings(BaseSettings):
    """MCP (Model Context Protocol) configuration settings."""

    timeout: int = Field(default=30, description="MCP request timeout", ge=1)
    max_retries: int = Field(default=2, description="Maximum retry attempts", ge=0)
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

    def __init__(self, **kwargs):
        yaml_values = _yaml_config.get("mcp", {})
        for key, value in yaml_values.items():
            if key not in kwargs:
                kwargs[key] = value
        super().__init__(**kwargs)


class AgentSettings(BaseSettings):
    """Agent behavior configuration settings."""

    max_iterations: int = Field(
        default=10, description="Maximum agent loop iterations", ge=1
    )
    tool_timeout: int = Field(
        default=30, description="Tool execution timeout", ge=1
    )
    reflection_enabled: bool = Field(
        default=True, description="Enable reflection after tool use"
    )
    system_prompt: str = Field(
        default="""You are a helpful AI assistant with access to various tools.
Use tools when needed to provide accurate and helpful responses.
Always explain your reasoning when using tools.""",
        description="System prompt for the agent",
    )

    model_config = SettingsConfigDict(
        env_prefix="AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def __init__(self, **kwargs):
        yaml_values = _yaml_config.get("agent", {})
        for key, value in yaml_values.items():
            if key not in kwargs:
                kwargs[key] = value
        super().__init__(**kwargs)


class AGUISettings(BaseSettings):
    """ag-ui protocol configuration settings."""

    buffer_size: int = Field(
        default=8192, description="Event buffer size for SSE", ge=1024
    )
    heartbeat_interval: int = Field(
        default=15, description="Heartbeat interval for SSE connections", ge=1
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

    def __init__(self, **kwargs):
        yaml_values = _yaml_config.get("agui", {})
        for key, value in yaml_values.items():
            if key not in kwargs:
                kwargs[key] = value
        super().__init__(**kwargs)


class LogSettings(BaseSettings):
    """Logging configuration settings."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Log level"
    )
    format: Literal["json", "text"] = Field(
        default="json", description="Log format"
    )
    include_timestamp: bool = Field(default=True, description="Include timestamp")
    include_caller: bool = Field(default=True, description="Include caller info")
    telemetry_enabled: bool = Field(default=False, description="Enable telemetry")

    model_config = SettingsConfigDict(
        env_prefix="LOG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def __init__(self, **kwargs):
        yaml_values = _yaml_config.get("logging", {})
        for key, value in yaml_values.items():
            if key not in kwargs:
                kwargs[key] = value
        super().__init__(**kwargs)


class Settings:
    """Central settings container aggregating all modular settings.

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
        """Reload all settings from config files and environment."""
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
