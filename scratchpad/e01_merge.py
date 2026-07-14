#!/usr/bin/env python
"""E0.1 best-of-N spread: merge the 5 guidance variants of each song into ONE
StepMania folder as distinct difficulty SLOTS, so the user selects a song once
and cycles the candidates for comparative (ear) ranking.

Why a merge (not 5 installed folders): every variant run writes a folder with the
SAME name (`0_<title>`), so installing them side-by-side collides. Stacking as
slots also makes A/B-at-fixed-song ergonomic (the whole point of E0.1).

Layout in:  outputs/e01_spread/<variant>/<i>_<title>/chart.sm  (+ audio)
Layout out: outputs/e01_merged/<i>_<title>/chart.sm            (+ audio)

Each generated variant -> one slot; the human ORIGINAL (present in every variant's
.sm) -> the Edit slot as the "reliably good" reference bar. The guidance label
lives in the author field so it's visible in-game.
"""
import os, glob, shutil, sys

SPREAD = "outputs/e01_spread"
OUT = "outputs/e01_merged"

# (variant dir, StepMania slot name, visible author label)  -- order = plain -> dissolve
VARIANTS = [
    ("00_plain",      "Beginner",  "plain-no-cond-g1.0"),
    ("01_match_g1.0", "Easy",      "match-g1.0"),
    ("02_match_g1.5", "Medium",    "match-g1.5"),
    ("03_match_g2.0", "Hard",      "match-g2.0"),
    ("04_match_g3.0", "Challenge", "match-g3.0"),
]
HUMAN_SLOT, HUMAN_LABEL = "Edit", "human-valset-NOTa-bar"


def split_sm(text):
    """Return (header_text, [block_text, ...]) where each block starts at '#NOTES:'."""
    idx = text.find("#NOTES:")
    if idx < 0:
        return text, []
    header = text[:idx]
    rest = text[idx:]
    # split keeping the delimiter
    parts = rest.split("#NOTES:")
    blocks = ["#NOTES:" + p for p in parts if p.strip()]
    return header, blocks


def block_author(block):
    lines = block.splitlines()
    # lines[0]='#NOTES:', [1]='  dance-single:', [2]='  author:', [3]='  difficulty:'
    return lines[2].strip().rstrip(":") if len(lines) > 3 else ""


def relabel(block, slot, author):
    lines = block.splitlines()
    lines[2] = f"     {author}:"   # author/description field (visible)
    lines[3] = f"     {slot}:"     # StepMania difficulty slot
    return "\n".join(lines) + ("\n" if not block.endswith("\n") else "")


def title_of(header):
    for ln in header.splitlines():
        if ln.startswith("#TITLE:"):
            return ln[len("#TITLE:"):].rstrip(";").strip()
    return "?"


LABELS = [lbl for _, _, lbl in VARIANTS] + [HUMAN_LABEL]


def write_ranking_sheet(songs):
    """songs: list of (folder, title). Emit a fill-in sheet that answers the E0.1
    kill-switch AND parses into preference pairs (the 'Rank' line) for E2."""
    lines = [
        "# E0.1 best-of-N spread — ranking sheet",
        "",
        "Each song has 6 slots (StepMania difficulty -> guidance variant):",
        "",
        "| slot | variant | meaning |",
        "|---|---|---|",
        "| Beginner  | plain  | canonical, NO groove conditioning, g=1.0 (just another candidate — NOT a yardstick) |",
        "| Easy      | g1.0   | --match_radar, guidance 1.0 (conditioned, un-amplified) |",
        "| Medium    | g1.5   | --match_radar, guidance 1.5 (musical steer) |",
        "| Hard      | g2.0   | --match_radar, guidance 2.0 (strong, near the knee) |",
        "| Challenge | g3.0   | --match_radar, guidance 3.0 (past the knee) |",
        "| Edit      | HUMAN  | val-set human chart — REFERENCE/CURIOSITY ONLY, NOT a quality bar (some are note-walls) |",
        "",
        "**THE ORACLE IS YOUR TASTE.** Rank purely by how good each feels to YOU — not vs plain, not vs the human.",
        "",
        "**E0.1's two gates:** (1) SPREAD — do the candidates genuinely differ in quality, or feel same-y? "
        "(if same-y, there's nothing to select and the selection arc dies here). (2) TOP-IS-GOOD — is your #1 pick "
        "a genuine banger *by your taste*? (that's the oracle ceiling selection could reach).",
        "",
        f"Fill `Rank` (best->worst, labels: {', '.join(LABELS)}), `Spread_real` (y/n), `Top_is_banger` (y/n). "
        "The Rank line seeds the preference pairs for the reward model.",
        "",
    ]
    for folder, title in songs:
        lines += [
            f"## {title}",
            "```",
            f"Rank:          >  >  >  >  > ",      # e.g. g1.5 > g2.0 > plain > g1.0 > HUMAN > g3.0
            "Spread_real:   ",                     # y / n  (gate 1: do they differ by taste?)
            "Top_is_banger: ",                     # y / n  (gate 2: is your #1 good BY YOUR TASTE?)
            "Notes:         ",
            "```",
            "",
        ]
    path = os.path.join(OUT, "E01_RANKING.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nranking sheet -> {path}")


def main():
    plain_songs = sorted(glob.glob(f"{SPREAD}/00_plain/*/"))
    if not plain_songs:
        sys.exit(f"no songs under {SPREAD}/00_plain/ -- did the spread run finish?")
    os.makedirs(OUT, exist_ok=True)
    made = 0
    songs = []
    for song_dir in plain_songs:
        folder = os.path.basename(song_dir.rstrip("/"))   # e.g. 0_<title>
        base_sm = os.path.join(song_dir, "chart.sm")
        if not os.path.exists(base_sm):
            print(f"  SKIP {folder}: no chart.sm"); continue
        header, _ = split_sm(open(base_sm, encoding="utf-8", errors="replace").read())
        songs.append((folder, title_of(header)))

        merged_blocks = []
        # 5 generated variants
        for vdir, slot, label in VARIANTS:
            vsm = os.path.join(SPREAD, vdir, folder, "chart.sm")
            if not os.path.exists(vsm):
                print(f"  WARN {folder}: missing {vdir}"); continue
            _, blocks = split_sm(open(vsm, encoding="utf-8", errors="replace").read())
            gen = [b for b in blocks if block_author(b) == "generated"]
            if not gen:
                print(f"  WARN {folder}/{vdir}: no generated block"); continue
            merged_blocks.append(relabel(gen[0], slot, label))
        # human reference (from plain)
        _, pblocks = split_sm(open(base_sm, encoding="utf-8", errors="replace").read())
        orig = [b for b in pblocks if block_author(b) == "original"]
        if orig:
            merged_blocks.append(relabel(orig[0], HUMAN_SLOT, HUMAN_LABEL))

        out_dir = os.path.join(OUT, folder)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "chart.sm"), "w", encoding="utf-8") as f:
            f.write(header + "\n".join(merged_blocks))
        # copy audio (any non-.sm file in the plain folder)
        for a in os.listdir(song_dir):
            if not a.endswith(".sm"):
                shutil.copy2(os.path.join(song_dir, a), os.path.join(out_dir, a))
        print(f"  OK {folder}: {len(merged_blocks)} slots")
        made += 1
    print(f"\nmerged {made} songs -> {OUT}/  (Beginner=plain .. Challenge=g3.0, Edit=HUMAN)")
    write_ranking_sheet(songs)


if __name__ == "__main__":
    main()
