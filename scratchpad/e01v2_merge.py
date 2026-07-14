#!/usr/bin/env python
"""E0.1-v2 merge: stack each song's 5 guidance-sweep variants into ONE StepMania folder as
difficulty SLOTS (no human-ref this time -- the user found val human charts to be maxed gibberish,
and the yardstick is the user's TASTE, not the human chart).

Guidance sweep over a FIXED MODERATE style (NOT --match_radar): plain + modstyle at g=1.0/1.5/2.0/3.0.
27 songs (BPM x length factorial). Layout in: outputs/e01v2_spread/<variant>/<i>_<title>/chart.sm.
"""
import os, glob, shutil, sys

SPREAD = "outputs/e01v2_spread"
OUT = "outputs/e01v2_merged"
VARIANTS = [
    ("00_plain",   "Beginner",  "plain-no-cond"),
    ("style_g1.0", "Easy",      "modstyle-g1.0"),
    ("style_g1.5", "Medium",    "modstyle-g1.5"),
    ("style_g2.0", "Hard",      "modstyle-g2.0"),
    ("style_g3.0", "Challenge", "modstyle-g3.0"),
]
LABELS = [lbl for _, _, lbl in VARIANTS]


def split_sm(text):
    idx = text.find("#NOTES:")
    if idx < 0:
        return text, []
    header, rest = text[:idx], text[idx:]
    return header, ["#NOTES:" + p for p in rest.split("#NOTES:") if p.strip()]


def block_author(block):
    lines = block.splitlines()
    return lines[2].strip().rstrip(":") if len(lines) > 3 else ""


def relabel(block, slot, author):
    lines = block.splitlines()
    lines[2] = f"     {author}:"
    lines[3] = f"     {slot}:"
    return "\n".join(lines) + ("\n" if not block.endswith("\n") else "")


def title_of(header):
    for ln in header.splitlines():
        if ln.startswith("#TITLE:"):
            return ln[len("#TITLE:"):].rstrip(";").strip()
    return "?"


def write_sheet(songs):
    L = [
        "# E0.1-v2 ranking sheet (guidance sweep over a fixed MODERATE style; NO match_radar)",
        "",
        "5 slots/song (StepMania difficulty -> guidance variant):",
        "",
        "| slot | variant | meaning |",
        "|---|---|---|",
        "| Beginner  | plain | canonical, NO groove conditioning (just another candidate) |",
        "| Easy      | g1.0  | fixed moderate --style, guidance 1.0 (conditioned, unamplified) |",
        "| Medium    | g1.5  | moderate style, guidance 1.5 (musical) |",
        "| Hard      | g2.0  | moderate style, guidance 2.0 (strong, near the knee) |",
        "| Challenge | g3.0  | moderate style, guidance 3.0 (past the knee) |",
        "",
        "**THE ORACLE IS YOUR TASTE.** Rank by how good each feels to YOU. Two gates: (1) SPREAD -- do they "
        "differ in quality, or feel same-y? (2) TOP-IS-GOOD -- is your #1 a genuine banger by your taste?",
        "",
        f"Fill `Rank` (best->worst, labels: {', '.join(LABELS)}), `Spread_real` (y/n), `Top_is_banger` (y/n).",
        "Songs are grouped by BPM x length bin (in the folder name prefix) so you can see if the knee moves with tempo/length.",
        "",
    ]
    for folder, title in songs:
        L += [f"## {title}", "```",
              "Rank:          >  >  > ", "Spread_real:   ", "Top_is_banger: ", "Notes:         ", "```", ""]
    open(os.path.join(OUT, "E01v2_RANKING.md"), "w", encoding="utf-8").write("\n".join(L))


def main():
    plain = sorted(glob.glob(f"{SPREAD}/00_plain/*/"))
    if not plain:
        sys.exit(f"no songs under {SPREAD}/00_plain/ -- run finished?")
    os.makedirs(OUT, exist_ok=True)
    songs, made = [], 0
    for sd in plain:
        folder = os.path.basename(sd.rstrip("/"))
        base = os.path.join(sd, "chart.sm")
        if not os.path.exists(base):
            continue
        header, _ = split_sm(open(base, encoding="utf-8", errors="replace").read())
        songs.append((folder, title_of(header)))
        blocks = []
        for vdir, slot, label in VARIANTS:
            vsm = os.path.join(SPREAD, vdir, folder, "chart.sm")
            if not os.path.exists(vsm):
                print(f"  WARN {folder}: missing {vdir}"); continue
            _, bl = split_sm(open(vsm, encoding="utf-8", errors="replace").read())
            gen = [b for b in bl if block_author(b) == "generated"]
            if gen:
                blocks.append(relabel(gen[0], slot, label))
        od = os.path.join(OUT, folder); os.makedirs(od, exist_ok=True)
        open(os.path.join(od, "chart.sm"), "w", encoding="utf-8").write(header + "\n".join(blocks))
        for a in os.listdir(sd):
            if not a.endswith(".sm"):
                shutil.copy2(os.path.join(sd, a), os.path.join(od, a))
        print(f"  OK {folder}: {len(blocks)} slots"); made += 1
    write_sheet(songs)
    print(f"\nmerged {made} songs -> {OUT}/ + E01v2_RANKING.md")


if __name__ == "__main__":
    main()
