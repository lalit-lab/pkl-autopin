# Handoff — read this first if you are a fresh Claude Code session

Paste this into a new session on the RTX 5050 laptop:

> Read HANDOFF.md and README.md in this repo, then continue the work.

---

## What this project is

Lalit has 21 PKL Match 17 tackle screenshots (Bengaluru Bulls vs Patna
Pirates). The goal: automatically work out **where on the mat each tackle
happened**, in metres, and show successful vs unsuccessful tackles on an HTML
map. Fully automatic — he explicitly rejected a hybrid "human clicks once per
frame" version after seeing the evidence below.

Two existing hand-made tools live in the screenshots folder (not in this repo):

- `Tackle Mapper.html` — manual: click 2 points per court line, tag each line,
  solve a homography, click the contact point.
- `PKL Pin Editor.html` — 494 KB bundled page. Its `SEED` constant holds **20
  hand-plotted ground-truth points**. This is the only ground truth and is what
  `validate.py` scores against.

**Outcome is never inferred.** It is read from the filename suffix `" S.png"`
(successful) / `" U.png"` (unsuccessful), exactly as the Pin Editor does.

## State as of 2026-09-12

- Classical colour+Hough pipeline (`court.py`) — **abandoned at 3/21 frames**.
  Kept only for its geometry helpers.
- Synthetic-data + segmentation approach — **built, training barely started.**
  `court_seg.pt` is committed at **step 1000 of 26000**, trained on a slow
  GTX 1650 (~20 img/s, 3.8 h projected). Training moved to the RTX 5050.
- **The pipeline has never been run end-to-end. It is unvalidated.** Do not
  describe any of it as working until `validate.py` says so.

## Next actions, in order

1. Follow `SETUP_OTHER_PC.md` — **torch cu128, not cu124** (RTX 5050 is
   Blackwell/sm_120; cu124 throws "no kernel image is available" *after*
   `cuda.is_available()` returns True).
2. `train.py --steps 26000 --batch 32 --workers 10 --resume`
3. `run.py`, then `validate.py`.
4. **Look at `_debug/` overlays before believing any number** (see below).
5. Report mean error in metres and how many frames land within 0.5 m / 1.0 m.

## Hard-won lessons — do not re-derive these

- **A low reprojection residual means nothing on its own.** Degenerate
  homographies that fan every court line through one vanishing point scored
  0.5 px while being completely wrong, and beat the correct answer. Chamfer
  distance to a line-evidence map plus **IoU of the reprojected mat against the
  detected mat** are the gates that actually work. Adding them dropped an
  apparent 15/21 to a real 3/21 — the 15 were false positives.
- **Finding lines is easy; naming them is the hard part.** A line 3 m from the
  sideline could be baulk or bonus, and guessing wrong rescales the whole depth
  axis. When the mat's ends are out of shot the depth scale is genuinely
  under-determined from geometry alone. This is the entire reason for the
  segmentation model: it labels each line *by type*.
- `04.15 , (3-1) Deepak S.png` is a close-up of legs with **no court visible**.
  Unsolvable by any method. It is also the one frame missing from the
  hand-plotted SEED, i.e. Lalit hit the same wall manually. Expect 20/21 max.
- Two venue palettes appear: purple court + orange lobby (most frames), and
  pink court + yellow lobby (`09.08`, `10.28`). Both are covered by the
  renderer's colour randomisation.
- The scoreboard banner shares the lobby hue and its white text reads as line
  evidence. Top ~5.5% / bottom ~12.5% of the frame must be cut before masking.
- **fp16 AMP gives NaN loss** with the heavy line-class weights. Train in fp32;
  the run is GPU-bound anyway so AMP buys little.
- The renderer was 160 ms/sample until the mat and floor were split into two
  small templates warped separately, with lighting applied after compositing
  rather than to the template. Now 24.6 ms. Training is data-bound if workers
  are too few — watch `img/s` against `nvidia-smi` utilisation.

## Conventions

- `+x` projects to the **right** of the image. `depth` is `|Y|`, distance from
  the mid line. Matches `Tackle Mapper.html`.
- World frame: X across (−5…5 m), Y along (−6.5…6.5 m). Lines at X=±5 (mat
  edge), X=±4 (sideline), Y=0 (mid), Y=±3.75 (baulk), Y=±4.75 (bonus, painted
  dark), Y=±6.5 (end).

## Working style Lalit expects

- Answer in English even when he writes in Hinglish.
- When something is ambiguous, **ask him** rather than guessing from file names
  or git state.
- Be straight about what is and isn't working — he was given the 3/21 result
  with evidence and chose the harder path knowingly. Don't oversell results.
