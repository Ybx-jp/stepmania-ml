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

## 5. `notes/HANDOFF.md` — the next-Claude pointer (REWRITE to current state)
This is the first thing the next cold session reads — it MUST describe where we are NOW, not a past session.
Rewrite (don't append-only) so it stands alone:
- **WHERE WE ARE** — deployed model/state, what this work changed (or "diagnostic, no model change").
- **THE ACTIVE THREAD + its lineage file** — the current line of inquiry, its state, and the **open fork /
  binding question** (what decision is pending, e.g. a by-ear gate).
- **AWAITING USER** — any installed playtest set / pending verdict, with the exact question + where to log it.
- **CANONICAL EXPORT DEFAULTS** — a PERMANENT section (keep it in every rewrite) holding the deployed
  export config inside the `<!-- CANONICAL-EXPORT-DEFAULTS:START ... :END -->` markers as `key = value` lines;
  it MUST mirror `export_typed_samples.py`'s argparse defaults (validated in step 6).
- **BRANCH / PR STATE** — current branch (verify the name), pushed?, open PRs.
- **READ-FIRST pointers** — the 2–3 notes + the skills + the lineage file the next Claude should open, in order.
- **DISCIPLINE reminders** — the load-bearing rules (Rule 0, by-ear gate, one-change).
Keep it tight and current; stale handoffs mislead worse than no handoff. Date it.

## 6. Integrity pass
- Referenced filenames resolve (grep the new refs); reciprocal links present; no dangling pointer to a renamed
  branch/file; the active-thread pointer is singular.
- **VALIDATE CANONICAL EXPORT DEFAULTS:** run `python tools/check_export_defaults.py`. It parses the HANDOFF
  canonical block and FAILS (exit 1, lists the diffs) if any value drifted from `export_typed_samples.py`'s live
  argparse defaults. On failure, reconcile — update the HANDOFF block AND the `generation-defaults` skill §1 to
  match the code (or fix the code if the default is the bug) — and re-run until it passes. A stale "canonical
  defaults" description silently mis-guides the next export/probe.

## 7. Commit
- **Branch first if on the default branch** (`main`) — refresh docs land on a `docs/<thread>-<state>` branch, never
  directly on `main` (it's protected). Create it before committing.
- Stage the docs/notes/skills (incl. `HANDOFF.md`; NOT bulky `outputs/` artifacts; DO include the session's
  work-product probe/tooling scripts the lineage references, so the names resolve); one coherent `docs(...)`
  commit with a message that lists what was refreshed. Memories live OUTSIDE the repo (persisted, not committed).

## 8. Open the PR (final step)
- `git push -u origin <branch>` then `gh pr create --base main` — open a PR for the refresh branch so it lands via
  the protected-main flow.
- Write the PR body like any other PR: describe the changes made, the experiments conducted, and the results —
  same as the repo's existing PRs. End with the `🤖 Generated with [Claude Code]` line per the repo convention.
- Report back: what was refreshed, the **PR number/URL**, and any lineage **stubs** left to backfill. Reference the
  PR by number; don't assert its merge state (CLAUDE.md Documentation Discipline).

## Guardrails
- Don't blanket-rewrite — refresh the delta; leave correct content alone (churn hides real changes).
- Don't commit `outputs/` artifacts, secrets, or unrelated pre-existing modifications you didn't make.
- Don't invent findings to fill a section; "no durable change here" is a valid outcome.
- This skill WRITES the knowledge layer; `experiment-design`/`conditioning-mechanics` define WHAT goes in it.
