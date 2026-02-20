"""
FastAPI application entry point.

Production-ready chatbot backend with AWS Strands, Bedrock, and ag-ui protocol.
"""

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import chat, tools
from app.core.config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log.level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.

    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting chatbot API")
    logger.info("Bedrock region: %s", settings.bedrock.region)
    logger.info("Default model: %s", settings.bedrock.default_model)
    logger.info(
        "MCP enabled: %d endpoints configured",
        len(settings.mcp.endpoints),
    )

    yield

    # Shutdown
    logger.info("Shutting down chatbot API")


# Create FastAPI application
app = FastAPI(
    title="Agentic Chatbot API",
    description=(
        "Production-ready chatbot backend with "
        "AWS Strands, Bedrock, and ag-ui protocol"
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS
# Note: allow_credentials=True requires explicit origins, not wildcards.
# Using allow_credentials=False with wildcard origins for broad access.
# For production, set specific origins and enable credentials as needed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix="/api/v1")
app.include_router(tools.router, prefix="/api/v1")


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {
        "service": "agentic-chatbot",
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "chatbot-api",
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=True,  # Enable for development
    )
