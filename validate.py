"""Score the automatic pins against the hand-plotted SEED in PKL Pin Editor.html.

The SEED constant holds 20 points the user placed by hand with Tackle Mapper.
It is the only ground truth available, so it is what the pipeline is measured
against - per frame, in metres, plus a summary.
"""
import argparse
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SHOTS = os.path.dirname(HERE)


def load_seed(path=None):
    """Pull the SEED object out of the bundled Pin Editor page."""
    path = path or os.path.join(SHOTS, "PKL Pin Editor.html")
    src = open(path, encoding="utf-8", errors="replace").read()
    m = re.search(r"const SEED = (\{.*?\});", src, re.S)
    if not m:
        raise SystemExit("could not find SEED in %s" % path)
    blob = m.group(1)
    # the page is bundled, so the script body is a JS string literal: unescape it
    blob = blob.replace('\\"', '"').replace("\\\\", "\\").replace("\\n", "")
    return json.loads(blob)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=os.path.join(SHOTS, "raider_positions.json"))
    ap.add_argument("--editor", default=os.path.join(SHOTS, "PKL Pin Editor.html"))
    a = ap.parse_args()

    seed = load_seed(a.editor)
    pred = {r["file"]: r for r in json.load(open(a.pred, encoding="utf-8"))}
    print("ground truth: %d frames   predicted: %d frames\n" % (len(seed), len(pred)))

    rows = []
    for f in sorted(seed):
        if f not in pred:
            print("  MISSING  %-42s (hand-plotted, not predicted)" % f[:42])
            continue
        gx, gd = seed[f]["x"], seed[f]["depth"]
        px, pd = pred[f]["x"], pred[f]["depth"]
        dx, dd = px - gx, pd - gd
        rows.append((f, gx, gd, px, pd, dx, dd, float(np.hypot(dx, dd))))

    if not rows:
        raise SystemExit("no overlap between prediction and ground truth")

    print("%-34s %14s %14s %14s" % ("frame", "hand (x,depth)", "auto (x,depth)", "error"))
    for f, gx, gd, px, pd, dx, dd, e in sorted(rows, key=lambda r: -r[7]):
        print("%-34s  %+5.2f,%5.2f   %+5.2f,%5.2f   %5.2f m %s"
              % (f[:34], gx, gd, px, pd, e, "  <-- large" if e > 1.5 else ""))

    err = np.array([r[7] for r in rows])
    dxs = np.array([r[5] for r in rows])
    dds = np.array([r[6] for r in rows])
    print("\nframes compared : %d" % len(rows))
    print("mean error      : %.2f m" % err.mean())
    print("median error    : %.2f m" % np.median(err))
    print("90th pct        : %.2f m" % np.percentile(err, 90))
    print("within 0.5 m    : %d / %d" % (int((err <= 0.5).sum()), len(err)))
    print("within 1.0 m    : %d / %d" % (int((err <= 1.0).sum()), len(err)))
    print("bias x / depth  : %+.2f m / %+.2f m" % (dxs.mean(), dds.mean()))
    # a consistent sign flip would show up as a strong negative correlation
    gxs = np.array([r[1] for r in rows])
    pxs = np.array([r[3] for r in rows])
    if len(rows) > 2 and np.std(gxs) > 1e-6 and np.std(pxs) > 1e-6:
        print("x correlation   : %+.2f  (negative => the +X convention is flipped)"
              % float(np.corrcoef(gxs, pxs)[0, 1]))


if __name__ == "__main__":
    main()
