# rarecell-mcp

Exposed FastMCP workflow server for rarecell. Lets any MCP client
(Claude Desktop, Claude Code, Cursor) drive a rarecell isolation
end-to-end with four high-level tools:

- `draft_profile(prompt, output_path)` — NL → draft profile YAML
- `validate_input(adata_path)` — verify counts + gene IDs
- `run_isolation(input_path, profile_path, out_dir, auto_policy?)` — drive IsolateRunner
- `inspect_report(report_path, question?)` — summarize a past run

## Install

```bash
pip install rarecell-mcp
```

## Run

```bash
rarecell-mcp serve
```

## Wire into Claude Desktop / Claude Code

```json
{
  "mcpServers": {
    "rarecell": {
      "command": "rarecell-mcp",
      "args": ["serve"]
    }
  }
}
```

This is pre-release v0.x.
