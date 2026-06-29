#!/usr/bin/env python3
"""A shitty little UI for driving export_typed_samples.py.

Run it with the project's conda python (so the subprocess inherits the env):

    conda activate stepmania-chart-gen
    python tools/chart_ui.py

It maps the exporter's ~20 flags onto a few friendly controls (conditioning MODE
instead of remembering `--style "chaos=q0.99" --guidance 3.0`, a governor PRESET
instead of three stamina knobs), shows the exact command it will run, and streams
generation output live. Nothing here is ML logic — it just builds an argv and
shells out to experiments/generation_typed/export_typed_samples.py.
"""
import os
import sys
import shlex
import queue
import signal
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, scrolledtext

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPORTER = os.path.join("experiments", "generation_typed", "export_typed_samples.py")

# The deployed highres model (gen_motif_full_fixed) vs the older style/base checkpoint.
MODELS = {
    "deployed (highres, gen_motif_full_fixed)": (
        "checkpoints/gen_motif_full_fixed/best_val.pt", "highres"),
    "gen_style (base, 23-dim)": ("checkpoints/gen_style/best_val.pt", "base"),
}

# The 4 Hard playtest songs, as path substrings that uniquely select their source charts.
PLAYTEST_FILTER = "deja loin (164),#infinity/oh world,sabo/high school love,kneeso hime"

# Governor presets -> (fatigue_penalty, stamina_ceiling, stamina_breathe). None = flag omitted.
# stamina_ceiling: LOWER thins harder & pulls density toward natural baseline. 50 = gentlest "tasteful
# edit" that respects a CONDITIONED (cranked) density; 25 = relief toward natural (fights density-cranking
# conditioning like chaos — the 2026-06-28 confound). Default Full -> 50.
# NOTE: off-states use 0.0 (not None) for ceiling so the flag is passed EXPLICITLY — the exporter now
# defaults stamina_ceiling to 50, so omitting it would silently re-enable stamina (the <=0 guard = off).
GOV_PRESETS = {
    "Off (no governor)":            (0.0, 0.0, 0.0),
    "Stage-1 (fatigue/jacks only)": (2.0, 0.0, 0.0),
    "Full (fatigue + stamina@50 + breathing arc)": (2.0, 50.0, 1.2),
    "Custom":                       (2.0, 50.0, 1.2),
}
COND_MODES = ["OFF (plain)", "ON (match own radar)", "Style (manifold, e.g. chaos)"]


class ChartUI:
    def __init__(self, root):
        self.root = root
        root.title("🎛  shitty little chart studio")
        self.proc = None
        self.q = queue.Queue()

        pad = dict(padx=6, pady=3, sticky="w")
        frm = ttk.Frame(root, padding=10)
        frm.grid(row=0, column=0, sticky="nsew")
        root.columnconfigure(0, weight=1); root.rowconfigure(0, weight=1)

        r = 0
        # --- model ---
        ttk.Label(frm, text="Model").grid(row=r, column=0, **pad)
        self.model = tk.StringVar(value=list(MODELS)[0])
        ttk.OptionMenu(frm, self.model, self.model.get(), *MODELS, command=self._touch
                       ).grid(row=r, column=1, columnspan=3, sticky="ew"); r += 1

        # --- conditioning ---
        ttk.Label(frm, text="Conditioning").grid(row=r, column=0, **pad)
        self.cond = tk.StringVar(value=COND_MODES[0])
        ttk.OptionMenu(frm, self.cond, self.cond.get(), *COND_MODES, command=self._touch
                       ).grid(row=r, column=1, sticky="ew")
        ttk.Label(frm, text="style spec").grid(row=r, column=2, **pad)
        self.style = tk.StringVar(value="chaos=q0.99")
        e = ttk.Entry(frm, textvariable=self.style, width=16); e.grid(row=r, column=3, sticky="ew")
        e.bind("<KeyRelease>", self._touch); r += 1

        ttk.Label(frm, text="Guidance (CFG)").grid(row=r, column=0, **pad)
        self.guidance = tk.DoubleVar(value=3.0)
        self._slider(frm, r, self.guidance, 1.0, 4.0)
        ttk.Label(frm, text="1=off · 1.4 musical · 2-3 strong/forced", foreground="#888"
                  ).grid(row=r, column=3, **pad); r += 1

        # --- pattern temperature ---
        ttk.Label(frm, text="Pattern temp").grid(row=r, column=0, **pad)
        self.temp = tk.DoubleVar(value=1.0)
        self._slider(frm, r, self.temp, 0.5, 1.5)
        ttk.Label(frm, text="0.7 coherent · ~1.0 jacks→jumps · >1.2 scramble", foreground="#888"
                  ).grid(row=r, column=3, **pad); r += 1

        # --- 16th unlock (onset_phase_calib b16) -- un-buries off-beat 16ths the global tau hides ---
        ttk.Label(frm, text="16th unlock").grid(row=r, column=0, **pad)
        self.b16 = tk.DoubleVar(value=0.0)
        self._slider(frm, r, self.b16, 0.0, 2.0)
        ttk.Label(frm, text="0=off (buried) · 0.5–1.0 unlock · 2.0 over-syncopated", foreground="#888"
                  ).grid(row=r, column=3, **pad); r += 1

        # --- governor ---
        ttk.Label(frm, text="Governor").grid(row=r, column=0, **pad)
        self.gov = tk.StringVar(value="Full (fatigue + stamina@50 + breathing arc)")
        ttk.OptionMenu(frm, self.gov, self.gov.get(), *GOV_PRESETS, command=self._on_gov
                       ).grid(row=r, column=1, columnspan=3, sticky="ew"); r += 1

        self.fat = tk.StringVar(); self.cei = tk.StringVar(); self.bre = tk.StringVar()
        gf = ttk.Frame(frm); gf.grid(row=r, column=1, columnspan=3, sticky="w")
        for i, (lab, var) in enumerate([("fatigue_penalty", self.fat),
                                        ("stamina_ceiling", self.cei),
                                        ("stamina_breathe", self.bre)]):
            ttk.Label(gf, text=lab).grid(row=0, column=2 * i, padx=(0, 3))
            en = ttk.Entry(gf, textvariable=var, width=6); en.grid(row=0, column=2 * i + 1, padx=(0, 10))
            en.bind("<KeyRelease>", self._touch)
        r += 1

        # --- songs / difficulty ---
        ttk.Label(frm, text="Song filter").grid(row=r, column=0, **pad)
        self.filter = tk.StringVar(value=PLAYTEST_FILTER)
        e = ttk.Entry(frm, textvariable=self.filter); e.grid(row=r, column=1, columnspan=2, sticky="ew")
        e.bind("<KeyRelease>", self._touch)
        ttk.Button(frm, text="4 Hard playtest", command=self._preset_songs
                   ).grid(row=r, column=3, sticky="ew"); r += 1

        ttk.Label(frm, text="Difficulty").grid(row=r, column=0, **pad)
        self.diff = tk.StringVar(value="Hard")
        ttk.OptionMenu(frm, self.diff, "Hard", "Any", "Beginner", "Easy", "Medium", "Hard",
                       command=self._touch).grid(row=r, column=1, sticky="ew")
        ttk.Label(frm, text="# songs").grid(row=r, column=2, **pad)
        self.num = tk.IntVar(value=4)
        ttk.Spinbox(frm, from_=1, to=16, textvariable=self.num, width=5, command=self._touch
                    ).grid(row=r, column=3, sticky="w"); r += 1

        # --- output ---
        ttk.Label(frm, text="Output name").grid(row=r, column=0, **pad)
        self.out = tk.StringVar(value="ui_set")
        e = ttk.Entry(frm, textvariable=self.out); e.grid(row=r, column=1, sticky="ew")
        e.bind("<KeyRelease>", self._touch)
        self.install = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm, text="install to ~/sm-generated", variable=self.install,
                        command=self._touch).grid(row=r, column=2, columnspan=2, **pad); r += 1

        # --- command preview ---
        ttk.Label(frm, text="Command").grid(row=r, column=0, sticky="nw", padx=6, pady=3)
        self.cmd = scrolledtext.ScrolledText(frm, height=4, wrap="word", font=("monospace", 9),
                                             background="#f4f4f4")
        self.cmd.grid(row=r, column=1, columnspan=3, sticky="ew", pady=3); r += 1

        # --- buttons ---
        bf = ttk.Frame(frm); bf.grid(row=r, column=0, columnspan=4, sticky="ew", pady=(6, 3))
        self.run_btn = ttk.Button(bf, text="▶ Generate", command=self.run)
        self.run_btn.pack(side="left")
        self.stop_btn = ttk.Button(bf, text="■ Stop", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=6)
        self.status = ttk.Label(bf, text="idle", foreground="#888"); self.status.pack(side="left", padx=10)
        r += 1

        # --- output log ---
        self.log = scrolledtext.ScrolledText(frm, height=16, wrap="none", font=("monospace", 9),
                                             background="#111", foreground="#ddd")
        self.log.grid(row=r, column=0, columnspan=4, sticky="nsew", pady=(6, 0))
        frm.rowconfigure(r, weight=1)
        for c in (1, 2): frm.columnconfigure(c, weight=1)

        self._on_gov(); self._refresh()
        self.root.after(100, self._drain)

    # ---- widget helpers ----
    def _slider(self, frm, r, var, lo, hi):
        f = ttk.Frame(frm); f.grid(row=r, column=1, columnspan=2, sticky="ew")
        ttk.Scale(f, from_=lo, to=hi, variable=var, orient="horizontal",
                  command=lambda *_: self._refresh()).pack(side="left", fill="x", expand=True)
        ttk.Label(f, textvariable=var, width=5).pack(side="left")
        var.trace_add("write", lambda *_: self._round(var))

    def _round(self, var):
        try: var.set(round(float(var.get()), 2))
        except (tk.TclError, ValueError): pass

    def _touch(self, *_): self._refresh()

    def _on_gov(self, *_):
        fat, cei, bre = GOV_PRESETS[self.gov.get()]
        self.fat.set("" if fat is None else fat)
        self.cei.set("" if cei is None else cei)
        self.bre.set("" if bre is None else bre)
        self._refresh()

    def _preset_songs(self):
        self.filter.set(PLAYTEST_FILTER); self.diff.set("Hard"); self.num.set(4); self._refresh()

    # ---- command building ----
    def build_argv(self):
        ckpt, feats = MODELS[self.model.get()]
        a = [sys.executable, EXPORTER,
             "--data_dir", "data", "--audio_dir", "data",
             "--checkpoint", ckpt, "--features", feats,
             "--pattern_temperature", str(self.temp.get()),
             "--num_songs", str(self.num.get()), "--seed", "42",
             "--out_dir", os.path.join("outputs", self.out.get() or "ui_set")]
        if self.filter.get().strip():
            a += ["--song_filter", self.filter.get().strip()]
        if self.diff.get() != "Any":
            a += ["--difficulty_select", self.diff.get()]
        # 16th unlock: per-phase calib offset "b8,b16" (leave 8ths alone, un-bury 16ths only)
        if self.b16.get() > 0:
            a += ["--onset_phase_calib", f"0,{self.b16.get()}"]
        # conditioning
        mode = self.cond.get()
        if mode.startswith("ON"):
            a += ["--match_radar", "--guidance", str(self.guidance.get())]
        elif mode.startswith("Style"):
            a += ["--style", self.style.get().strip(), "--guidance", str(self.guidance.get())]
        # governor
        for flag, var in [("--fatigue_penalty", self.fat),
                          ("--stamina_ceiling", self.cei),
                          ("--stamina_breathe", self.bre)]:
            v = var.get().strip()
            if v != "":
                a += [flag, v]
        if self.install.get():
            a.append("--install")
        return a

    def _refresh(self):
        argv = self.build_argv()
        # render python path short, quote args with spaces
        shown = ["python", EXPORTER] + [shlex.quote(x) for x in argv[2:]]
        self.cmd.delete("1.0", "end"); self.cmd.insert("1.0", " ".join(shown))

    # ---- run / stream ----
    def run(self):
        if self.proc: return
        argv = self.build_argv()
        self.log.delete("1.0", "end")
        self._append(f"$ {' '.join(shlex.quote(x) for x in argv[1:])}\n\n")
        self.run_btn.config(state="disabled"); self.stop_btn.config(state="normal")
        self.status.config(text="running…", foreground="#0a0")
        self.proc = subprocess.Popen(argv, cwd=REPO, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1,
                                     start_new_session=True)
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()

    def _reader(self, proc):
        for line in proc.stdout:
            self.q.put(line)
        proc.wait()
        self.q.put(("__done__", proc.returncode))

    def _drain(self):
        try:
            while True:
                item = self.q.get_nowait()
                if isinstance(item, tuple):
                    code = item[1]
                    self._append(f"\n[finished, exit {code}]\n")
                    self.proc = None
                    self.run_btn.config(state="normal"); self.stop_btn.config(state="disabled")
                    self.status.config(text=f"done (exit {code})",
                                       foreground="#0a0" if code == 0 else "#c00")
                else:
                    self._append(item)
        except queue.Empty:
            pass
        self.root.after(100, self._drain)

    def _append(self, text):
        self.log.insert("end", text); self.log.see("end")

    def stop(self):
        if self.proc:
            try: os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError): pass
            self.status.config(text="stopped", foreground="#c00")


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("860x760")
    ChartUI(root)
    root.mainloop()
