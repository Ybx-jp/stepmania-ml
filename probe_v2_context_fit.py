"""data-layer-v2 fit probe: does the DEPLOYED LayeredTypedChartGenerator train at the 48th-grid
3x context (~4320 frames / 2 min) on the RTX 3060, and at what batch size / precision?

Directly measures the model that v2 would retrain (NOT train_factorized's different arch). Runs a real
forward+backward at each (T, batch, precision), reports peak CUDA memory + step time. Cheap, decisive.
"""
import time, torch, torch.nn.functional as F
from src.generation.typed_model import (LayeredTypedChartGenerator, NUM_PANELS, NUM_SYMBOLS,
                                         NUM_PATTERNS, NUM_TYPES)

dev = torch.device("cuda")
TOTAL_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
AUDIO_DIM = 42  # highres

# T targets: today's 16th cap (1440), the model pos cap (2048), and the 48th-grid 2-min song (4320)
#            + the raised max_len that must hold it (4608 = 4320 rounded up).
CONTEXTS = {"16th_1440(today)": 1440, "48th_4320(2min)": 4320}
MAX_LEN = 4608
BATCHES = [1, 2, 4, 8, 16]

def build():
    m = LayeredTypedChartGenerator(audio_dim=AUDIO_DIM, max_len=MAX_LEN).to(dev)
    m.train()
    return m

def one(m, T, B, amp):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    audio = torch.randn(B, T, AUDIO_DIM, device=dev)
    states = torch.randint(0, NUM_SYMBOLS, (B, T, NUM_PANELS), device=dev)
    diff = torch.randint(0, 4, (B,), device=dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    dt = []
    for it in range(3):  # 1 warmup + 2 timed
        torch.cuda.synchronize(); t0 = time.time()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            ol, pat, typ = m(audio, states, diff)
            loss = ol.float().mean() + pat.float().mean() + typ.float().mean()
        loss.backward(); opt.step()
        torch.cuda.synchronize()
        if it > 0: dt.append(time.time() - t0)
    peak = torch.cuda.max_memory_allocated() / 1e9
    return peak, sum(dt) / len(dt)

print(f"GPU {TOTAL_GB:.1f} GB | model d128/4dec/2enc | audio_dim {AUDIO_DIM} | max_len {MAX_LEN}")
print(f"{'context':>18} {'prec':>5} {'B':>3} {'peak_GB':>8} {'step_s':>8} {'fit?':>5}")
for name, T in CONTEXTS.items():
    for amp in (False, True):
        for B in BATCHES:
            m = build()
            try:
                peak, step = one(m, T, B, amp)
                fit = "OK" if peak < TOTAL_GB * 0.92 else "TIGHT"
                print(f"{name:>18} {'bf16' if amp else 'fp32':>5} {B:>3} {peak:>8.2f} {step:>8.3f} {fit:>5}")
                oom = peak >= TOTAL_GB * 0.92
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"{name:>18} {'bf16' if amp else 'fp32':>5} {B:>3} {'OOM':>8} {'-':>8} {'OOM':>5}")
                    oom = True
                    torch.cuda.empty_cache()
                else:
                    raise
            finally:
                del m; torch.cuda.empty_cache()
            if oom:
                break  # larger batches will also OOM
