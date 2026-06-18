#!/usr/bin/env bash
# UserPromptSubmit hook for the ml-gloss skill.
# Emits (as added context) the list of ML terms already explained in
# notes/ml_glossary.md plus the dual-explanation behavior rule, so jargon gets a
# plain-English gloss on FIRST use only and is never re-explained.

set -euo pipefail

ROOT="${CLAUDE_PROJECT_DIR:-.}"
GLOSSARY="$ROOT/notes/ml_glossary.md"

if [[ -f "$GLOSSARY" ]]; then
  # Term entries are markdown bullets like "- **term** — ...". Pull the bold term(s).
  terms="$(grep -oE '^- \*\*[^*]+\*\*' "$GLOSSARY" 2>/dev/null \
            | sed -E 's/^- \*\*//; s/\*\*$//' \
            | awk 'NR>1{printf "; "} {printf "%s", $0}' || true)"
else
  terms=""
fi

echo "[ml-gloss] Behavior: when you use ML jargon, keep the precise term AND add a brief plain-English gloss in parentheses on its FIRST use in the conversation; then append the term to notes/ml_glossary.md. Do NOT re-gloss terms already listed below."
echo "[ml-gloss] Already explained (reference notes/ml_glossary.md): ${terms:-none yet}"
