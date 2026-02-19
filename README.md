# Agentic Chatbot Backend

Production-ready chatbot backend with intelligent mode selection and hybrid execution.

## Goals

Build a high-performance (500+ req/s) chatbot API that:

- **Adapts to query complexity**: Fast mode for simple questions, agentic mode for detailed comprehensive answer
- **Executes optimally**: Parallel independent tools, sequential tools on incomplete or irrelevant context
- **Maintains context**: Filesystem-based working memory for agentic mode
- **Uses official libraries**: AWS Strands SDK, ag-ui protocol, MCP - no custom implementations

## Design Ideas

### Modes

| Mode | Use Case | Tool Limit | Context |
|------|----------|------------|---------|
| **FAST** | Direct questions, lookups | 1-2 tools | Null (stateless) |
| **AGENTIC** | Research, multi-source data | 10 calls | Session filesystem |
| **AUTO** | Let LLM decide | Varies | Varies |

### Architecture

```
FastAPI → Agent Router → Mode Analyzer → Tool Planner → Strands Agent
                              ↓                    ↓
                         (FAST/AGENTIC)      (Parallel/Sequential)
                                                   ↓
                                              Session Filesystem
```

### Key Decisions

- **No partial streaming**: High-risk environment requires complete, curated responses only
- **No embeddings**: Fast, simple context management via filesystem + grep
- **Official libraries only**: strands.agents, ag-ui-protocol, MCP SDK
- **Hybrid execution**: Parallel when independent, sequential when dependent
- **Session isolation**: Each agentic session gets its own filesystem workspace

### Filesystem Working Memory

Agentic mode stores tool results to `sessions/{session_id}/`:
- Enables LLM to grep across collected data
- Preserves large context without token limits
- Maintains provenance (which tool, when, what args)

### Transport

- **API**: REST over HTTP with SSE streaming
- **Tools**: MCP Streamable HTTP for remote tools
- **Events**: ag-ui protocol for real-time UI updates
