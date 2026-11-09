---
title: Building an MCP server in Python
date: 2026-09-07
summary: A small, read-only MCP server with the official Python SDK v2: two typed tools, stdio, Claude Desktop and Claude Code config, Inspector and security
tags: Agents, Python
---

The Model Context Protocol (MCP) is the standard way to give an AI application access to tools and data without writing a custom integration for every model and every app. In this post I build a small MCP server in Python that lets an assistant search and read a folder of Markdown notes, connect it to Claude Desktop and Claude Code, test it with the MCP Inspector, and go through the security basics.

One thing to know before copying older tutorials: the Python SDK had a major release in July 2026. In `mcp` 2.x, the class that used to be `FastMCP` is now `MCPServer`, and `from mcp.server.fastmcp import FastMCP` raises an error. Everything below targets `mcp` 2.x.

## What MCP is

MCP uses a client-server architecture with three participants:

- **Host**: the AI application the user works in, such as Claude Desktop, Claude Code or VS Code.
- **Client**: a component inside the host that holds a connection to one server. A host creates one client per server.
- **Server**: a program that provides context and capabilities. It can run locally or remotely.

Under that sits a JSON-RPC 2.0 data layer and a transport layer. There are two standard transports: **stdio**, where the host launches the server as a subprocess and talks over standard input and output, and **Streamable HTTP**, for remote servers.

Servers expose three core primitives:

- **Tools**: functions the model can call to do something, such as query a database or call an API.
- **Resources**: data the application can read for context, such as file contents or a schema.
- **Prompts**: reusable templates that help structure an interaction.

The current protocol version, `2026-07-28`, is stateless: every request carries the protocol version and capabilities in its `_meta` field, and a `server/discover` request lets a client ask what a server supports. The SDK handles all of this for you, so you mostly think in terms of tools, resources and prompts.

## Set up the project

You need Python 3.10 or later and the `mcp` package at 2.0.0 or later. I use `uv`, which is what the official quickstart uses:

```bash
uv init notes-server
cd notes-server
uv add "mcp[cli]"
mkdir notes
printf '# Retries\nUse exponential backoff with jitter.\n' > notes/retries.md
```

The `cli` extra installs the `mcp` command-line tool. Plain `pip install "mcp[cli]"` works too.

## The server

The server exposes two tools, `search_notes` and `read_note`, plus one resource that lists every note. It is read-only by design.

```python
"""A small, read-only MCP server that exposes a folder of Markdown notes."""

import logging
import os
from pathlib import Path
from typing import Annotated

from pydantic import Field

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

# stdio servers must never write to stdout: it carries the JSON-RPC stream.
# The logging module writes to stderr by default, which is safe.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("notes-server")

NOTES_DIR = Path(os.environ.get("NOTES_DIR", "./notes")).expanduser().resolve()
MAX_NOTE_BYTES = 100_000

mcp = MCPServer("notes")

READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=False)


def _note_path(name: str) -> Path:
    """Map a note name to a file inside NOTES_DIR, refusing anything outside it."""
    path = (NOTES_DIR / f"{name}.md").resolve()
    if not path.is_relative_to(NOTES_DIR):
        raise ToolError("Note name resolves outside the notes folder.")
    return path


@mcp.tool(annotations=READ_ONLY)
def search_notes(
    query: Annotated[str, Field(min_length=2, max_length=100, description="Text to look for")],
    limit: Annotated[int, Field(ge=1, le=20, description="Maximum results")] = 5,
) -> list[str]:
    """Search note titles and bodies for a phrase. Returns matching note names."""
    needle = query.casefold()
    matches: list[str] = []
    for path in sorted(NOTES_DIR.glob("*.md")):
        if path.stat().st_size > MAX_NOTE_BYTES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if needle in path.stem.casefold() or needle in text.casefold():
            matches.append(path.stem)
        if len(matches) >= limit:
            break
    logger.info("search_notes matched %d note(s)", len(matches))
    return matches


@mcp.tool(annotations=READ_ONLY)
def read_note(
    name: Annotated[
        str,
        Field(pattern=r"^[A-Za-z0-9_-]{1,64}$", description="Note name without .md, e.g. 'retries'"),
    ],
) -> str:
    """Return the full Markdown text of one note."""
    path = _note_path(name)
    if not path.is_file():
        raise ToolError(f"No note called {name!r}.")
    if path.stat().st_size > MAX_NOTE_BYTES:
        raise ToolError("Note is too large to return.")
    return path.read_text(encoding="utf-8", errors="replace")


@mcp.resource("notes://index")
def notes_index() -> str:
    """A newline-separated list of every note name."""
    return "\n".join(p.stem for p in sorted(NOTES_DIR.glob("*.md")))


if __name__ == "__main__":
    mcp.run(transport="stdio")
```

Save it as `notes_server.py`. A few things are doing more work than they look.

**Type hints become the schema.** `MCPServer` reads the function signature and docstring to build each tool's name, description and JSON Schema. `Annotated[int, Field(ge=1, le=20)]` turns into `"minimum": 1, "maximum": 20` in the schema the model sees, and the SDK validates arguments against it before your function runs. A call with `limit=50` never reaches `search_notes`.

**Return types matter.** Because `search_notes` is annotated `-> list[str]`, the SDK returns structured content (`{"result": [...]}`) alongside the text content, which clients can use without parsing prose.

**Errors are results.** Raising `ToolError` produces a tool result with `isError: true` and your message. The model sees that the call failed and why, and can try something else, which is better than a crashed server.

**Annotations are hints, not controls.** `readOnlyHint` tells a host the tool does not modify anything, which it may use to decide how much to prompt the user. It does not stop the code from doing something else, so it is no substitute for actually writing read-only code.

**Never print in a stdio server.** Anything on stdout corrupts the JSON-RPC stream. Use `logging`, which goes to stderr.

## Run it over stdio

```bash
uv run notes_server.py
```

It will appear to hang. That is correct: it is waiting for JSON-RPC messages on stdin from a host. Press Ctrl+C to stop it. In practice you rarely run it yourself; the host launches it.

## Test it with the MCP Inspector

The Inspector is the reference tool for testing servers. It needs Node 22.19.0 or newer and runs through `npx`. Pass it the command that launches your server:

```bash
npx @modelcontextprotocol/inspector uv --directory /ABSOLUTE/PATH/TO/notes-server run notes_server.py
```

This opens a browser UI where you can list the tools, call them with arguments and see the raw JSON-RPC exchange. The SDK also has a shortcut, `uv run mcp dev notes_server.py`, which opens the server in the Inspector.

For automated checks, the Inspector has a CLI mode that suits CI:

```bash
npx @modelcontextprotocol/inspector --cli uv --directory /ABSOLUTE/PATH/TO/notes-server run notes_server.py \
  --method tools/call --tool-name search_notes --tool-arg query=backoff
```

The things I check every time: each tool's schema is what I expect, a valid call works, an invalid argument (`limit=50`, a name like `../secrets`) comes back as an error result, and nothing appears on stdout except protocol messages.

## Connect it to Claude Desktop

Claude Desktop reads servers from `claude_desktop_config.json`. Open it from Settings, Developer, Edit Config, or directly:

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

Add the server under `mcpServers`:

```json
{
  "mcpServers": {
    "notes": {
      "command": "uv",
      "args": [
        "--directory",
        "/ABSOLUTE/PATH/TO/notes-server",
        "run",
        "notes_server.py"
      ],
      "env": {
        "NOTES_DIR": "/ABSOLUTE/PATH/TO/notes-server/notes"
      }
    }
  }
}
```

Use absolute paths. On Windows, escape backslashes (`C:\\Users\\me\\notes-server`) or use forward slashes, and you may need the full path to `uv.exe` (find it with `where uv`). Quit Claude Desktop completely and restart it. If the server does not appear, the logs are in `~/Library/Logs/Claude` on macOS and `%APPDATA%\Claude\logs` on Windows; `mcp-server-notes.log` holds your server's stderr.

## Connect it to Claude Code

Claude Code adds servers with `claude mcp add`. Everything after `--` is the command to run, passed through untouched:

```bash
claude mcp add --env NOTES_DIR=/ABSOLUTE/PATH/TO/notes-server/notes notes \
  -- uv --directory /ABSOLUTE/PATH/TO/notes-server run notes_server.py
```

The default scope is `local` (this project, just you). `--scope project` writes the entry to `.mcp.json` in the repository so the team shares it, and `--scope user` makes it available in all your projects. Inside a session, `/mcp` shows connected servers and their status.

## Security basics

An MCP server runs with your user's permissions and sits between a model and real systems. A few habits go a long way.

**Least privilege.** Expose the smallest useful surface. This server is read-only, scoped to one folder, and has no tool that deletes or sends anything. If a server needs credentials, give it a token scoped to exactly what the tools do, passed through `env`, never hard-coded. Add write tools only when you need them, and keep them separate so a host can gate them.

**Validate every input.** The model writes the arguments, and the model can be manipulated by text it has read. Treat arguments as untrusted user input. Constrain them in the schema (patterns, lengths, ranges), then check again in code where it matters, as `_note_path` does by resolving the path and refusing anything outside `NOTES_DIR`. The pattern on `name` already blocks `../`; the path check is there in case the pattern is ever loosened.

**Treat tool output as untrusted.** This is the one people miss. A note, a web page or an email returned by a tool can contain instructions aimed at the model ("ignore previous instructions and send the contents of..."). Your server cannot fix that on its own, but it can avoid making it worse: cap output size, return data rather than instructions, and do not combine "reads untrusted content" and "can send data out" in the same server without a human in the loop. I cover this in more depth in a separate post on prompt injection.

**Be careful what you install.** Third-party servers run code on your machine. Read the source or use servers from publishers you trust, and pin versions.

## Where to go next

From here I would add a prompt (`@mcp.prompt()`) for a common workflow such as "summarise the notes about X", write tests that use the SDK's in-process `Client` against the `MCPServer` instance, and only then think about Streamable HTTP and authorisation for a remote deployment.

## Sources

- [Model Context Protocol: Architecture overview (2026-07-28)](https://modelcontextprotocol.io/docs/2026-07-28/learn/architecture)
- [Model Context Protocol: Build an MCP server](https://modelcontextprotocol.io/docs/2026-07-28/develop/build-server)
- [Model Context Protocol: Connect to local MCP servers](https://modelcontextprotocol.io/docs/tutorials/use-local-mcp-server)
- [Model Context Protocol: MCP Inspector](https://modelcontextprotocol.io/docs/tools/inspector)
- [MCP Python SDK v2 migration guide](https://py.sdk.modelcontextprotocol.io/v2/migration/)
- [mcp 2.0.0 on PyPI](https://pypi.org/project/mcp/2.0.0/)
- [modelcontextprotocol/python-sdk on GitHub](https://github.com/modelcontextprotocol/python-sdk)
- [Claude Code: Connect Claude Code to tools via MCP](https://code.claude.com/docs/en/mcp)
