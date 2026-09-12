# pkl_autopin — automatic raider-position mapping from PKL screenshots

Takes a folder of PKL tackle screenshots and produces, with no clicking:

- where on the mat each tackle happened, in metres (`x` across, `depth` from the mid line)
- whether it was **successful** or **unsuccessful**
- a standalone HTML map with green/red pins, plus a per-frame debug overlay so
  every number can be checked

## Why a trained model

The obvious approach — segment the mat by colour, find the white lines with
Hough, solve a homography — was built first and is still in `court.py`. It
reached **3 of 21 frames**. The failure is structural, not a tuning problem:

- Finding lines is easy; knowing *which* line you found is not. A line 3 m from
  the sideline could be the baulk line or the bonus line, and picking wrong
  rescales the whole depth axis.
- When the mat's ends are out of shot — most zoomed frames — nothing in the
  image resolves that. The depth scale is genuinely under-determined.
- Scoring cannot save you. Degenerate homographies that fan every court line
  through a single vanishing point score a *better* algebraic residual (0.5 px)
  than the correct answer. Chamfer distance and mat-outline IoU were added as
  gates and exposed most of the early "successes" as false positives.

A segmentation model fixes this at the root by **naming each line** (mid /
baulk / bonus / end / sideline / mat edge). Once lines are named, the
homography is determined and the search collapses to a couple of sign choices.

## Why there is no labelling work

The kabaddi mat is a known flat template: a 10 m × 13 m rectangle with lines at
fixed positions. So the training set is *rendered*, not annotated —
`synth.py` projects that template through randomised broadcast cameras onto
crowd backgrounds, then composites piles of players, arena lighting, scoreboard
bars and photometric noise on top. Every pixel label is exact by construction,
and the model never sees the same frame twice.

Real screenshots are used only as background plates (the crowd strip above the
court). They are never used as labels.

## Layout

| file | role |
|---|---|
| `synth.py` | synthetic renderer + camera sampling |
| `model.py` | compact U-Net, 9 classes |
| `train.py` | trains on data generated on the fly |
| `infer.py` | predicted masks → named lines → homography |
| `players.py` | YOLO persons, teams by torso hue, raider = most ringed by the other kit |
| `run.py` | end-to-end runner and HTML output |
| `validate.py` | scores results against the hand-plotted SEED |
| `court.py` | the superseded classical pipeline, kept for its geometry helpers |

## Use

```sh
.venv/Scripts/python.exe train.py --steps 26000 --batch 10     # once
.venv/Scripts/python.exe run.py                                # this folder
.venv/Scripts/python.exe run.py --dir "D:\PKL M18"             # any match
.venv/Scripts/python.exe validate.py                           # accuracy vs SEED
```

Outputs land next to the screenshots: `raider_positions.json`,
`pin_editor_seed.json` (paste into the Pin Editor's `pkl-pin-editor-v2`
localStorage key), `raider-position-map.html`, and `_debug/`.

## Conventions

- `+x` projects to the **right** of the image; `depth` is `|Y|`, distance from
  the mid line — matching `Tackle Mapper.html`.
- Outcome is read from the filename suffix `" S.png"` / `" U.png"`, never
  inferred. This is what `PKL Pin Editor.html` already does.

## Known limits

- A frame with no court visible cannot be solved by any method. In this set
  that is `04.15 , (3-1) Deepak S.png`, a close-up of legs — which is also the
  one frame missing from the hand-plotted SEED.
- Two venue palettes appear in these broadcasts (purple+orange, pink+yellow);
  both are covered by the renderer's colour randomisation. A genuinely new
  palette would want its range added to `mat_template`.
- The raider heuristic assumes the standard raid picture: one attacker among
  defenders. It has no way to be right on a frame where that is not what is
  shown.
