#!/usr/bin/env python3
"""Export a Claude Code session transcript to readable markdown for learning-material mining.

Why this exists (not `/export`): `/export` is an in-session INTERACTIVE slash command — a skill (e.g. `/refresh`)
cannot self-invoke it. But Claude Code already persists every session as JSONL under
`~/.claude/projects/<cwd-with-slashes-as-dashes>/<session-id>.jsonl`. This reads that source of truth and renders
markdown that keeps what makes the transcript useful for learning the MATH + METHODOLOGY (not just vibes):
  * user prompts (system-reminder / command-context noise stripped),
  * assistant prose IN FULL (the explanations + insight boxes),
  * assistant THINKING (the reasoning; toggle with --no-thinking),
  * every tool_use (the exact command / edit that was run — the methodology),
  * tool_result output, TRUNCATED (the empirical numbers — loss curves, probe outputs).

NOTE on thinking: Claude Code does NOT persist assistant reasoning in plaintext — a thinking block on disk is an
empty string plus an opaque encrypted `signature`. So an export carries user prompts, assistant PROSE (the
explanations / insight boxes — where the pedagogy actually lives), tool calls, and results, but not the private
chain-of-thought. `--no-thinking` is kept for compatibility but is effectively a no-op (empty blocks are skipped).

The `refresh` skill calls this at the end of its cycle to drop the current session into a gitignored `transcripts/`
dir. Default target = the most-recently-modified (i.e. current/live) session for this project.

Usage:
  python tools/export_transcript.py                      # latest session of the CWD's project -> transcripts/
  python tools/export_transcript.py --session <uuid|all> # a specific session, or every session
  python tools/export_transcript.py --result-lines 20
"""
from __future__ import annotations
import argparse, json, re, sys
from datetime import datetime
from pathlib import Path

SYSTEM_REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)
COMMAND_WRAP = re.compile(r"<(command-name|command-message|command-args|local-command-[a-z]+)>.*?</\1>", re.DOTALL)
CMD_NAME = re.compile(r"<command-name>\s*(.*?)\s*</command-name>", re.DOTALL)


def project_dir_for(cwd: Path) -> Path:
    """Claude stores a project's sessions under ~/.claude/projects/<abs-cwd with '/' -> '-'>."""
    return Path.home() / ".claude" / "projects" / str(cwd).replace("/", "-")


def text_of(content) -> str:
    """tool_result / message content -> plain text (content may be a str or a list of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for b in content:
            if isinstance(b, dict):
                if b.get("type") == "text":
                    out.append(b.get("text", ""))
                elif "text" in b:
                    out.append(b["text"])
                elif b.get("type") == "image":
                    out.append("[image]")
            elif isinstance(b, str):
                out.append(b)
        return "\n".join(out)
    return "" if content is None else str(content)


def truncate(s: str, max_lines: int) -> str:
    lines = s.rstrip("\n").split("\n")
    if len(lines) <= max_lines:
        return "\n".join(lines)
    hidden = len(lines) - max_lines
    return "\n".join(lines[:max_lines]) + f"\n… [+{hidden} more lines truncated]"


def render_session(jsonl: Path, include_thinking: bool, result_lines: int) -> tuple[str, datetime | None]:
    recs = []
    with jsonl.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    parts: list[str] = []
    last_ts: datetime | None = None
    for o in recs:
        if o.get("type") not in ("user", "assistant"):
            continue
        if o.get("isSidechain") or o.get("isMeta"):
            continue  # skip subagent sidechains + meta records
        ts = o.get("timestamp")
        if ts:
            try:
                last_ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                pass
        content = o.get("message", {}).get("content")
        blocks = [content] if isinstance(content, str) else (content or [])

        if o["type"] == "user":
            # a "user" record is either a real prompt OR a carrier for a tool_result
            tool_results = [b for b in blocks if isinstance(b, dict) and b.get("type") == "tool_result"]
            if tool_results:
                for b in tool_results:
                    body = truncate(text_of(b.get("content")), result_lines)
                    parts.append(f"<details><summary>↳ tool result</summary>\n\n```\n{body}\n```\n</details>")
                continue
            raw = text_of(content)
            cmd = CMD_NAME.search(raw)
            cleaned = COMMAND_WRAP.sub("", SYSTEM_REMINDER.sub("", raw)).strip()
            if cmd and not cleaned:
                parts.append(f"### 🧑 User — `{cmd.group(1)}`")
            elif cleaned:
                prefix = f"`{cmd.group(1)}` " if cmd else ""
                parts.append(f"### 🧑 User\n\n{prefix}{cleaned}")
            continue

        # assistant
        for b in blocks:
            if not isinstance(b, dict):
                if isinstance(b, str) and b.strip():
                    parts.append(f"**Claude:**\n\n{b.strip()}")
                continue
            t = b.get("type")
            if t == "text" and b.get("text", "").strip():
                parts.append(f"**Claude:**\n\n{b['text'].strip()}")
            elif t == "thinking" and include_thinking:
                think = (b.get("thinking") or b.get("text") or "").strip()
                if think:
                    parts.append(f"<details><summary>💭 thinking</summary>\n\n{think}\n</details>")
            elif t == "tool_use":
                name = b.get("name", "?")
                inp = b.get("input", {}) or {}
                key = (inp.get("command") or inp.get("file_path") or inp.get("skill")
                       or inp.get("pattern") or inp.get("description") or "")
                key = str(key)
                if len(key) > 400:
                    key = key[:400] + " …"
                parts.append(f"> 🔧 **{name}** — `{key}`" if key else f"> 🔧 **{name}**")

    header = (f"# Transcript — {jsonl.stem}\n\n"
              f"_Session {jsonl.stem}; last activity {last_ts.date() if last_ts else 'unknown'}; "
              f"{len(recs)} records._\n\n"
              f"> Auto-exported from the on-disk JSONL by `tools/export_transcript.py` (wired into `/refresh`). "
              f"For learning-material mining — prose, reasoning, commands, and measured results.\n\n---\n")
    return header + "\n\n".join(parts) + "\n", last_ts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default="latest", help="session UUID, 'latest' (default), or 'all'")
    ap.add_argument("--out-dir", default="transcripts", help="output dir (gitignored); default transcripts/")
    ap.add_argument("--project-dir", default=None, help="override the ~/.claude/projects/<...> source dir")
    ap.add_argument("--cwd", default=None, help="project cwd used to locate transcripts (default: this repo root)")
    ap.add_argument("--no-thinking", dest="thinking", action="store_false",
                    help="(compat no-op: thinking isn't persisted in the JSONL — only an encrypted signature)")
    ap.add_argument("--result-lines", type=int, default=40, help="max lines kept per tool result (default 40)")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    cwd = Path(args.cwd) if args.cwd else repo
    src = Path(args.project_dir) if args.project_dir else project_dir_for(cwd)
    if not src.is_dir():
        sys.exit(f"no transcript dir at {src} — is this the right project cwd?")

    files = sorted(src.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        sys.exit(f"no .jsonl transcripts under {src}")
    if args.session == "latest":
        targets = files[:1]
    elif args.session == "all":
        targets = files
    else:
        targets = [p for p in files if p.stem == args.session]
        if not targets:
            sys.exit(f"session {args.session} not found under {src}")

    out_dir = (repo / args.out_dir) if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for jsonl in targets:
        md, ts = render_session(jsonl, args.thinking, args.result_lines)
        date = (ts.date().isoformat() if ts else "undated")
        out = out_dir / f"{date}_{jsonl.stem[:8]}.md"
        out.write_text(md)
        written.append(out)
    for w in written:
        print(f"wrote {w}  ({w.stat().st_size // 1024} KB)")
    print(f"\n{len(written)} transcript(s) -> {out_dir}")


if __name__ == "__main__":
    main()
