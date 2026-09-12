# Moving this to the RTX 5050 laptop

## 0. The one thing that will bite you

The RTX 5050 is a **Blackwell** card (compute capability sm_120). PyTorch
builds for **cu124 and older do not contain kernels for it**. If you copy the
`.venv` from the other laptop, or install the default `pip install torch`, you
will get:

```
CUDA error: no kernel image is available for execution on the device
```

Install the **cu128** wheels instead. That is the only difference between the
two machines.

**Never copy `.venv/` between machines.** It hardcodes paths and the wrong CUDA
build. Always rebuild it. `.gitignore` already excludes it.

---

## 1. What to copy

Copy these two things to the new laptop, keeping the layout:

```
PKL25 M17 SCREENSHOTS\          <- the 21 *.png screenshots
PKL25 M17 SCREENSHOTS\pkl_autopin\   <- this folder (code)
```

`run.py` defaults to "the folder my parent directory is", so keeping
`pkl_autopin` inside the screenshots folder means no arguments are needed.

**Option A — git (recommended, you already use lalit-lab):**

```powershell
# on THIS laptop, inside pkl_autopin
git init
git add .
git commit -m "PKL autopin: synthetic-trained court registration"
gh repo create lalit-lab/pkl-autopin --private --source=. --push
```

```powershell
# on the RTX 5050 laptop
git clone https://github.com/lalit-lab/pkl-autopin.git
```

Then copy the 21 `.png` screenshots across separately (they are not in git —
and shouldn't be, they're ~18 MB of binaries).

**Option B — just zip it.** Right-click `pkl_autopin` → Send to → Compressed
folder, but **delete `.venv` first** (it is 4.6 GB and useless there).

---

## 2. Install on the RTX 5050 laptop

Needs Python 3.11 or 3.12 and a recent NVIDIA driver (570+).

```powershell
cd <wherever>\PKL25 M17 SCREENSHOTS\pkl_autopin

py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip

# Blackwell / RTX 50xx -> cu128. This is the critical line.
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Verify before training — this must print `True` and name the card:

```powershell
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0)); print(torch.zeros(8,device='cuda').sum())"
```

If `torch.cuda.is_available()` is True but the last line throws "no kernel
image", you are still on a pre-cu128 build — uninstall and redo the torch line:

```powershell
.venv\Scripts\python.exe -m pip uninstall -y torch torchvision
```

---

## 3. Train

The 5050 has more VRAM and far more throughput than the 1650, so raise the
batch size and the number of renderer workers. Data generation was the
bottleneck on the old machine at batch 10; give it more workers so the GPU
stays fed.

```powershell
.venv\Scripts\python.exe train.py --steps 26000 --batch 32 --workers 10
```

- If you hit `CUDA out of memory`, drop to `--batch 24` or `16`.
- If `img/s` is low and GPU usage (check `nvidia-smi`) is under ~70%, raise
  `--workers` — you are data-bound, not GPU-bound.
- Checkpoints save every 1000 steps to `court_seg.pt`, and `--resume` picks up
  where it left off, so stopping with Ctrl+C is safe.

Expect roughly 20-30 minutes instead of 3.8 hours.

---

## 4. Run and check

```powershell
.venv\Scripts\python.exe run.py
.venv\Scripts\python.exe validate.py
```

`run.py` writes, next to the screenshots:

- `raider-position-map.html` — the mat with green (successful) / red
  (unsuccessful) pins
- `raider_positions.json`, `pin_editor_seed.json`
- `_debug/` — one overlay per frame; **look at these**, they are how you tell a
  real result from a confident wrong one

`validate.py` scores the output against the 20 hand-plotted points in
`PKL Pin Editor.html`. That comparison is the thing that says whether this
works — mean error in metres, and how many frames land within 0.5 m / 1.0 m.

---

## 5. If you want to keep training on the old laptop meanwhile

Nothing stops both running. `court_seg.pt` from whichever machine finishes
first can just be copied over the other — it is a plain state dict, and
`infer.py` loads it by path.
