"""End-to-end: screenshots folder -> raider positions -> HTML map.

    python run.py                      # the folder this package sits in
    python run.py --dir "D:\\PKL M18"  # any other match folder

Outputs, written next to the screenshots:
    raider_positions.json     {filename: {x, depth, outcome}}
    raider-position-map.html  the mat with every tackle plotted, S green / U red
    pin_editor_seed.json      paste into the Pin Editor's localStorage
    _debug/                   per-frame overlays so every result is checkable

Successful vs unsuccessful is NOT inferred: it is read from the filename
suffix " S.png" / " U.png", exactly as PKL Pin Editor.html already does.
"""
import argparse
import glob
import json
import os
import re
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import infer
import players as P
import synth
from court import apply_h


def outcome_of(name):
    """' S.png' -> successful tackle, ' U.png' -> unsuccessful."""
    if re.search(r"\sS\.png$", name, re.I):
        return "S"
    if re.search(r"\sU\.png$", name, re.I):
        return "U"
    return "?"


def draw_debug(bgr, H, cls, box, world, path):
    v = bgr.copy()
    if H is not None:
        for X in (-5, -4, 4, 5):
            p = apply_h(H, [(X, -6.5), (X, 6.5)]).astype(int)
            cv2.line(v, tuple(p[0]), tuple(p[1]), (255, 255, 255), 2)
        for Y, col in ((0.0, (0, 220, 90)), (3.75, (255, 120, 255)),
                       (-3.75, (255, 120, 255)), (4.75, (60, 160, 255)),
                       (-4.75, (60, 160, 255)), (6.5, (0, 230, 230)),
                       (-6.5, (0, 230, 230))):
            p = apply_h(H, [(-5.0, Y), (5.0, Y)]).astype(int)
            cv2.line(v, tuple(p[0]), tuple(p[1]), col, 2)
    if box is not None:
        x1, y1, x2, y2 = [int(t) for t in box[:4]]
        cv2.rectangle(v, (x1, y1), (x2, y2), (0, 210, 255), 3)
        cv2.circle(v, ((x1 + x2) // 2, y2), 7, (0, 0, 255), -1)
    if world is not None:
        cv2.putText(v, "x=%+.2f  depth=%.2f m" % (world[0], abs(world[1])),
                    (12, v.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 0), 5)
        cv2.putText(v, "x=%+.2f  depth=%.2f m" % (world[0], abs(world[1])),
                    (12, v.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)
    seg = cv2.resize(_palette()[cls], (bgr.shape[1] // 3, bgr.shape[0] // 3),
                     interpolation=cv2.INTER_NEAREST)
    v[0:seg.shape[0], v.shape[1] - seg.shape[1]:] = seg
    cv2.imwrite(path, v, [cv2.IMWRITE_JPEG_QUALITY, 85])


def _palette():
    return np.array([[30, 30, 30], [20, 120, 235], [190, 80, 150],
                     [255, 255, 255], [80, 255, 80], [0, 230, 230],
                     [160, 60, 200], [255, 160, 60], [60, 60, 255]], np.uint8)


def build_html(rows, out, title="Raider position map"):
    """Standalone page: the mat, every pin, and the S/U breakdown."""
    VX, VY, VW, VH = -5.5, 7.0, 11.0, 7.0

    def L(x):
        return "%.3f" % (((x - VX) / VW) * 100)

    def T(d):
        return "%.3f" % (((VY - d) / VH) * 100)

    pins, tab = [], []
    for i, r in enumerate(rows, 1):
        cls = "s" if r["outcome"] == "S" else "u"
        pins.append(
            '<div class="pin %s" style="left:%s%%;top:%s%%" title="%s &middot; %+.2f, %.2f m">%d</div>'
            % (cls, L(r["x"]), T(r["depth"]),
               r["file"].replace('"', "&quot;"), r["x"], r["depth"], i))
        tab.append("<tr><td>%d</td><td>%s</td><td class='%s'>%s</td><td>%+.2f</td><td>%.2f</td></tr>"
                   % (i, r["file"], cls,
                      "successful" if r["outcome"] == "S" else "unsuccessful",
                      r["x"], r["depth"]))
    ns = sum(1 for r in rows if r["outcome"] == "S")
    nu = sum(1 for r in rows if r["outcome"] == "U")
    left = sum(1 for r in rows if r["x"] < -1.2)
    mid = sum(1 for r in rows if -1.2 <= r["x"] <= 1.2)
    right = sum(1 for r in rows if r["x"] > 1.2)
    deep = sum(1 for r in rows if r["depth"] > 4.0)

    html = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>%s</title>
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=Barlow:wght@400;500&display=swap" rel="stylesheet">
<style>
 body{margin:0;background:#0d0b14;color:#efe9f7;font-family:Barlow,system-ui,sans-serif;padding:26px}
 h1{font-family:"Barlow Condensed",sans-serif;text-transform:uppercase;font-size:30px;margin:0 0 4px}
 .sub{color:#9d90b6;font-size:13px;margin-bottom:18px}
 .wrap{display:flex;gap:26px;flex-wrap:wrap;align-items:flex-start}
 .mat{position:relative;flex:1 1 620px;max-width:1000px;aspect-ratio:11/7;background:#21365e;border-radius:12px;overflow:hidden}
 .l{position:absolute;background:#f4f2f6}
 .pin{position:absolute;width:26px;height:26px;margin:-13px 0 0 -13px;border-radius:50%%;
      display:flex;align-items:center;justify-content:center;color:#0d0b14;
      border:2px solid rgba(20,16,26,.55);font-family:"Barlow Condensed",sans-serif;
      font-weight:700;font-size:14px;box-shadow:0 2px 8px rgba(0,0,0,.45)}
 .pin.s{background:#2ecc71}.pin.u{background:#e74c3c}
 .key{flex:0 0 240px;font-size:13px;line-height:1.9;color:#b6a9cd}
 .key b{color:#efe9f7}
 .dot{display:inline-block;width:11px;height:11px;border-radius:50%%;margin-right:7px}
 table{border-collapse:collapse;margin-top:26px;font-size:13px;width:100%%}
 th,td{text-align:left;padding:6px 16px 6px 0;border-bottom:1px solid #2c2340}
 th{font-family:"Barlow Condensed",sans-serif;text-transform:uppercase;letter-spacing:.12em;
    font-size:11px;color:#7d6f96}
 td{font-variant-numeric:tabular-nums;color:#cfc4e2}
 td.s{color:#2ecc71}td.u{color:#e74c3c}
 @media(max-width:640px){body{padding:16px}.mat{flex:1 1 100%%}}
</style></head><body>
<h1>%s</h1><div class="sub">%d tackles plotted &middot; green = successful, red = unsuccessful</div>
<div class="wrap">
<div class="mat">
 <div style="position:absolute;left:4.5455%%;width:90.909%%;top:7.143%%;bottom:0;background:#e8721c"></div>
 <div style="position:absolute;left:13.636%%;width:72.727%%;top:7.143%%;bottom:0;background:#9a4fc4"></div>
 <div class="l" style="left:4.5455%%;width:2px;top:7.143%%;bottom:0"></div>
 <div class="l" style="left:95.4545%%;width:2px;top:7.143%%;bottom:0"></div>
 <div class="l" style="left:13.636%%;width:2px;top:7.143%%;bottom:0"></div>
 <div class="l" style="left:86.364%%;width:2px;top:7.143%%;bottom:0"></div>
 <div class="l" style="left:4.5455%%;width:90.909%%;top:7.143%%;height:2px"></div>
 <div class="l" style="left:13.636%%;width:72.727%%;top:46.43%%;height:2px"></div>
 <div style="position:absolute;left:13.636%%;width:72.727%%;top:32.143%%;height:3px;background:#141418"></div>
 <div class="l" style="left:13.636%%;width:72.727%%;bottom:0;height:3px"></div>
 %s
</div>
<div class="key">
 <div><span class="dot" style="background:#2ecc71"></span><b>%d</b> successful</div>
 <div><span class="dot" style="background:#e74c3c"></span><b>%d</b> unsuccessful</div>
 <div style="margin-top:14px">left flank <b>%d</b></div>
 <div>centre <b>%d</b></div>
 <div>right flank <b>%d</b></div>
 <div style="margin-top:14px">deeper than 4 m <b>%d</b></div>
</div>
</div>
<table><tr><th>#</th><th>Frame</th><th>Outcome</th><th>Across (m)</th><th>Depth (m)</th></tr>
%s</table></body></html>""" % (title, title, len(rows), "\n ".join(pins),
                               ns, nu, left, mid, right, deep, "\n".join(tab))
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.dirname(HERE))
    ap.add_argument("--weights", default=os.path.join(HERE, "court_seg.pt"))
    ap.add_argument("--debug", action="store_true", default=True)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.dir, "*.png")))
    print("frames:", len(files))
    net, dev, step = infer.load_model(a.weights)
    print("court model: step", step, "on", dev)
    det = P.load_detector()

    dbg = os.path.join(a.dir, "_debug")
    os.makedirs(dbg, exist_ok=True)

    rows, failed = [], []
    for f in files:
        name = os.path.basename(f)
        bgr = cv2.imread(f)
        H, info, cls = infer.solve(net, dev, bgr)
        if H is None:
            failed.append((name, info.get("fail", "?")))
            print("FAIL %-42s %s" % (name[:42], info.get("fail")))
            continue
        boxes = P.detect(det, bgr)
        idx, world, teams = P.find_raider(bgr, H, boxes)
        if idx is None:
            failed.append((name, "no raider identified"))
            print("FAIL %-42s no raider (%d persons)" % (name[:42], len(boxes)))
            if a.debug:
                draw_debug(bgr, H, cls, None, None,
                           os.path.join(dbg, name.replace(".png", "_dbg.jpg")))
            continue
        w, _ = P.feet_world(H, boxes[idx])
        row = {"file": name, "x": round(float(np.clip(w[0], -5.2, 5.2)), 2),
               "depth": round(float(np.clip(abs(w[1]), 0, 6.8)), 2),
               "outcome": outcome_of(name), "iou": round(info["iou"], 3),
               "err_px": round(info["err_px"], 2)}
        rows.append(row)
        print("ok   %-42s x=%+.2f depth=%.2f  %s  iou=%.2f"
              % (name[:42], row["x"], row["depth"], row["outcome"], info["iou"]))
        if a.debug:
            draw_debug(bgr, H, cls, boxes[idx], w,
                       os.path.join(dbg, name.replace(".png", "_dbg.jpg")))

    with open(os.path.join(a.dir, "raider_positions.json"), "w", encoding="utf-8") as fp:
        json.dump(rows, fp, indent=1)
    seed = {r["file"]: {"x": r["x"], "depth": r["depth"]} for r in rows}
    with open(os.path.join(a.dir, "pin_editor_seed.json"), "w", encoding="utf-8") as fp:
        json.dump(seed, fp)
    build_html(rows, os.path.join(a.dir, "raider-position-map.html"))
    print("\n%d/%d solved  ->  raider-position-map.html" % (len(rows), len(files)))
    for n, why in failed:
        print("   unsolved: %-42s %s" % (n[:42], why))


if __name__ == "__main__":
    main()
