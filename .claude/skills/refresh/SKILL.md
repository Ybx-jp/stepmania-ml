---
name: refresh
description: >
  Run the knowledge-base REFRESH CYCLE — propagate what was just learned into the DURABLE layer so the next cold
  session doesn't re-derive it: update touched memories (+ MEMORY.md), notes/INDEX.md, any SKILL whose subject
  changed (new lever/default/corrected mechanism, pending human notes), and the experiment-lineage file(s) + index
  for the active/pivoted thread, then commit the docs. Use when the user says "refresh / sync / checkpoint the
  notes/memory", and proactively at the END of a work thread or session, after a PIVOT, or once a thread overturns
  a prior conclusion. Pairs with experiment-design (the lineage directive) and the file-memory system.
---

# refresh — the knowledge-base refresh cycle

Codifies the end-of-thread curation pass: take everything learned since the last refresh and write it into the
**durable knowledge layer** so a future cold session inherits it instead of re-deriving (or repeating a ruled-out
setup). **Run it at a checkpoint** (thread done / session ending / a pivot or overturned conclusion), not after
every tiny step. The goal is *accuracy + non-duplication*, not blanket rewriting — update what changed, fix what
went stale, link what's now related.

## 0. Scope the delta (what actually changed since last refresh)
- `git status --short` + `git log` since the last `docs(...)`/refresh commit → list modified code, new
  `notes/*_findings.md`, touched skills.
- Name the **thread(s)** this work belongs to (a coherent line of inquiry, not one probe) and whether any thread
  **pivoted / overturned a prior conclusion** (those are the highest-value things to capture while fresh).
- If nothing durable changed (pure exploration, no finding), say so and skip — don't manufacture churn.

## 1. Memories (`~/.claude/projects/<project>/memory/`)
- For each memory the work TOUCHED: fix stale claims (a "fix pending" that's now APPLIED, a superseded "ACTIVE
  thread" pointer, a number that moved), and add `[[links]]` to newly-related nodes. **Match the verb to the
  evidence** (vouched ≠ measured ≠ mapped).
- New durable facts (user/feedback/project/reference, non-obvious, not derivable from the repo) → a new memory
  file + a one-line `MEMORY.md` index entry. Check for an existing file first (update, don't duplicate); delete
  ones proven wrong.
- Reconcile the **active-thread pointer** across nodes so exactly one chain reads as current.

## 2. `notes/INDEX.md`
- Add each new `*_findings.md` under the right section + a dated **UPDATE** blurb tracing the arc (one line each:
  what → what learned). Keep it the accurate map; fix any pointer the work invalidated.

## 3. Skills whose SUBJECT changed (reconcile pending)
- If the work added a lever / changed a default / CORRECTED a mechanism, update the owning skill
  (`conditioning-mechanics` = the deployed math, `generation-defaults` = the canonical config, etc.) so it matches
  the code — a stale skill silently mis-guides the next probe.
- Clean up **pending human notes / uncommitted skill edits** into proper form (e.g. an inline "I don't trust this"
  → a labeled ⚠️ DISPUTED/unverified flag); preserve the intent, don't silently delete.

## 4. Experiment lineage (per the experiment-design DIRECTIVE)
- For each active/pivoted thread: create or update its
  `.claude/skills/experiment-design/experiment_lineage/<thread>-arc.md` — hypothesis CHAIN (believed → learned),
  each probe + verdict, the **attribution corrections** (what would have made each conclusion wrong), current
  state, open fork. Thread the `notes/*_findings.md` together; don't duplicate them.
- Add **reciprocal cross-arc links** (corroborates / depends-on) — when arc A cites arc B, back-link in B.
- Update `experiment_lineage/INDEX.md`: flip statuses (✅/🟡/⬜), add rows for new threads.

## 5. Integrity pass
- Referenced filenames resolve (grep the new refs); reciprocal links present; no dangling pointer to a renamed
  branch/file; the active-thread pointer is singular.

## 6. Commit
- Stage the docs/notes/skills (NOT bulky `outputs/` artifacts); one coherent `docs(...)` commit with a message
  that lists what was refreshed. Memories live OUTSIDE the repo (persisted, not committed) — note that.
- Report: what was refreshed, what was committed, and any lineage **stubs** left to backfill.

## Guardrails
- Don't blanket-rewrite — refresh the delta; leave correct content alone (churn hides real changes).
- Don't commit `outputs/` artifacts, secrets, or unrelated pre-existing modifications you didn't make.
- Don't invent findings to fill a section; "no durable change here" is a valid outcome.
- This skill WRITES the knowledge layer; `experiment-design`/`conditioning-mechanics` define WHAT goes in it.
