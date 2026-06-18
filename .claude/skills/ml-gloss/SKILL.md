---
name: ml-gloss
description: >
  Dual-language ML explanations: keep precise jargon but gloss each term in plain
  English on first use, tracked in a glossary so nothing is re-explained. Use when
  the user asks to view/search/edit the ML glossary, reset or re-explain a term, or
  asks how the gloss-on-first-use behavior works. The behavior itself runs every turn
  via the UserPromptSubmit hook in .claude/settings.json; this skill is the manual
  and the on-demand manager.
---

# ml-gloss

Keeps ML discussion both **precise and readable**: use the real jargon (so the user
learns the standard terms) and pair it with a plain-English gloss the first time each
term appears — once only, with a persistent glossary as the reference.

## The behavior (runs automatically every turn)

A `UserPromptSubmit` hook (`.claude/hooks/ml-gloss.sh`) injects, before each reply:
1. the rule below, and
2. the list of terms already in `notes/ml_glossary.md`.

Follow it in every response, not just when this skill is invoked:

- **First use of a term** (not already in the glossary): write the precise term, then a
  brief plain-English gloss in parentheses. Example:
  *"...this is exposure bias (the model only trained on real history, so its own mistakes
  at generation time push it into states it never saw)."*
  Keep the gloss to one clause — enough to land the idea, not a lecture.
- **Then record it**: append the term to `notes/ml_glossary.md` under the current date
  heading, as `- **term** — plain meaning *(how it shows up here)*`.
- **Already-explained terms** (listed by the hook / present in the glossary): just use the
  jargon plainly. Do **not** re-gloss — the user has the glossary for reference.
- Group obvious aliases in one entry (e.g. `teacher forcing / teacher-forced`).
- Only gloss genuine ML/domain jargon. Skip everyday words and basic programming terms.

## Glossary file

`notes/ml_glossary.md` — one bullet per term, grouped under dated `## Seeded/Added <date>`
headings. This is the user's reference; keep entries tight and plain.

## On-demand actions (when this skill is invoked)

- **view / list** — show the current glossary (or just the term list).
- **search <term>** — return that entry, or note it's undefined.
- **re-explain <term>** — give the plain gloss again on request (a user ask overrides the
  no-repeat rule).
- **add <term>: <gloss>** — append a new entry.
- **prune / edit** — tidy duplicates or sharpen a gloss.

## Components

- `notes/ml_glossary.md` — the dictionary (committed; the user's reference).
- `.claude/hooks/ml-gloss.sh` — UserPromptSubmit hook emitting the rule + known terms.
- `.claude/settings.json` — registers the hook.
