"""Predicted segmentation -> court homography.

Because the network names each line (mid / baulk / bonus / end / sideline /
mat edge), there is no labelling search worth speaking of: the only thing left
to resolve is which side of the court each instance sits on.  That is a global
sign per axis plus one bit for any class where only a single line is visible -
a handful of candidates instead of the tens of thousands the classical
pipeline had to grind through, and no depth ambiguity.

Convention: +X projects to the RIGHT of the image, matching how a viewer reads
"across".  Depth is |Y|, distance from the mid line, so the Y sign is free.
"""
import itertools
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth
from court import homography, pixel_err, apply_h, wline_x, wline_y

NET_W, NET_H = 448, 256
LINE_CLASSES = (synth.L_MATEDGE, synth.L_SIDE, synth.L_END,
                synth.L_BONUS, synth.L_BAULK, synth.L_MID)
REGION_CLASSES = (synth.LOBBY, synth.COURT) + LINE_CLASSES


# --- model -----------------------------------------------------------------
def load_model(path=None, device=None):
    import torch
    from model import UNet
    here = os.path.dirname(os.path.abspath(__file__))
    path = path or os.path.join(here, "court_seg.pt")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    net = UNet().to(device)
    ck = torch.load(path, map_location=device)
    net.load_state_dict(ck["model"])
    net.eval()
    return net, device, ck.get("step", 0)


def predict(net, device, bgr):
    """Full-size frame -> (class map at NET resolution, softmax confidence)."""
    import torch
    small = cv2.resize(bgr, (NET_W, NET_H), interpolation=cv2.INTER_AREA)
    x = small.astype(np.float32) / 255.0
    x = (x - np.array([0.45, 0.43, 0.47], np.float32)) / np.array([0.26, 0.26, 0.27], np.float32)
    t = torch.from_numpy(x.transpose(2, 0, 1)[None].copy()).to(device)
    with torch.no_grad():
        logit = net(t)
        prob = torch.softmax(logit, 1)[0].cpu().numpy()
    return prob.argmax(0).astype(np.uint8), prob


# --- line instances --------------------------------------------------------
def _fit(pts):
    """TLS line through points -> [a,b,c], |(a,b)|=1."""
    p = np.asarray(pts, float)
    c = p.mean(0)
    _, _, vt = np.linalg.svd(p - c)
    d = vt[0]
    n = np.array([-d[1], d[0]])
    n /= np.linalg.norm(n) or 1e-9
    return np.array([n[0], n[1], -n @ c]), c, d


def lines_from_mask(mask, min_px=45, ang_tol=7.0, off_tol=7.0):
    """Split one line-class mask into straight instances.

    Fragments of the same painted line (broken by players) are merged when
    their fitted lines agree; two genuinely different lines of the same class
    (a sideline and its mirror) stay apart because their offsets differ.
    """
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    frags = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_px:
            continue
        ys, xs = np.nonzero(lab == i)
        pts = np.stack([xs, ys], 1)
        L, c, d = _fit(pts)
        frags.append([L, pts, c, d, len(pts)])
    frags.sort(key=lambda f: -f[4])

    groups = []
    for L, pts, c, d, npx in frags:
        ang = np.degrees(np.arctan2(L[1], L[0])) % 180
        placed = False
        for g in groups:
            ang2 = np.degrees(np.arctan2(g["L"][1], g["L"][0])) % 180
            da = min(abs(ang - ang2), 180 - abs(ang - ang2))
            off = abs(g["L"] @ np.array([c[0], c[1], 1.0]))
            if da < ang_tol and off < off_tol:
                g["pts"] = np.vstack([g["pts"], pts])
                g["L"], _, _ = _fit(g["pts"])
                placed = True
                break
        if not placed:
            groups.append({"L": L, "pts": pts})
    out = []
    for g in groups:
        pts = g["pts"]
        if len(pts) < min_px:
            continue
        L, c, d = _fit(pts)
        t = (pts - c) @ d
        out.append((L, c + d * t.min(), c + d * t.max(), len(pts)))
    return out


def detect_lines(cls):
    """class map -> {class: [(line, a, b, npx), ...]}"""
    found = {}
    for c in LINE_CLASSES:
        m = (cls == c).astype(np.uint8) * 255
        if not m.any():
            continue
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        inst = lines_from_mask(m)
        if inst:
            found[c] = sorted(inst, key=lambda t: -t[3])[:2]
    return found


# --- assignment ------------------------------------------------------------
def _axis(c):
    return "x" if c in (synth.L_MATEDGE, synth.L_SIDE) else "y"


def _values(c):
    return [v for _, v in synth.LINE_WORLD[c]]


def candidates(found):
    """Enumerate (image_line, world_line) pair-sets consistent with the classes."""
    xs_cls = [c for c in found if _axis(c) == "x"]
    ys_cls = [c for c in found if _axis(c) == "y"]
    if not xs_cls or not ys_cls:
        return []

    def opts(cls_list, sign):
        """For each class: every way its instances can take its world values."""
        per = []
        for c in cls_list:
            inst = found[c]
            vals = sorted(v * sign for v in _values(c))
            if len(vals) == 1:                       # mid line: unambiguous
                per.append([[(inst[0], vals[0])]])
                continue
            if len(inst) >= 2:
                # two instances of a mirrored pair: order them along the axis
                a, b = inst[0], inst[1]
                ca = (a[1] + a[2]) / 2
                cb = (b[1] + b[2]) / 2
                first, second = (a, b) if ca[0] + ca[1] < cb[0] + cb[1] else (b, a)
                per.append([[(first, vals[0]), (second, vals[1])],
                            [(first, vals[1]), (second, vals[0])]])
            else:
                per.append([[(inst[0], v)] for v in vals])
        return [sum(p, []) for p in itertools.product(*per)]

    out = []
    for sx in (1, -1):
        for sy in (1, -1):
            for gx in opts(xs_cls, sx):
                for gy in opts(ys_cls, sy):
                    pairs = []
                    for (L, a, b, npx), v in gx:
                        pairs.append((L, wline_x(v), a, b))
                    for (L, a, b, npx), v in gy:
                        pairs.append((L, wline_y(v), a, b))
                    if len(pairs) >= 4:
                        out.append(pairs)
    return out


def _norm_T(W, H):
    s = float(np.hypot(W, H))
    T = np.array([[1 / s, 0, -W / (2 * s)], [0, 1 / s, -H / (2 * s)], [0, 0, 1.0]])
    Ti = np.array([[s, 0, W / 2.0], [0, s, H / 2.0], [0, 0, 1.0]])
    return T, Ti


def region_mask(cls):
    m = np.zeros(cls.shape, np.uint8)
    for c in REGION_CLASSES:
        m[cls == c] = 255
    return m


def mat_iou(H, matm):
    c = apply_h(H, [(-5.0, -6.5), (5.0, -6.5), (5.0, 6.5), (-5.0, 6.5)])
    if not np.isfinite(c).all() or np.abs(c).max() > 1e6:
        return 0.0
    proj = np.zeros(matm.shape, np.uint8)
    cv2.fillConvexPoly(proj, c.astype(np.int32), 255)
    inter = int(np.count_nonzero(cv2.bitwise_and(proj, matm)))
    union = int(np.count_nonzero(cv2.bitwise_or(proj, matm)))
    return inter / union if union else 0.0


def solve_from_cls(cls, shape):
    """class map -> (H world->ORIGINAL image pixels, info)."""
    found = detect_lines(cls)
    info = {"classes": {synth.CLASS_NAMES[c]: len(v) for c, v in found.items()}}
    cands = candidates(found)
    if not cands:
        info["fail"] = "no usable line classes (%s)" % info["classes"]
        return None, info

    matm = region_mask(cls)
    T, Ti = _norm_T(NET_W, NET_H)
    Tit = np.linalg.inv(T).T
    best = None
    for pairs in cands:
        nl = []
        for L, Lw, a, b in pairs:
            v = Tit @ L
            nl.append((v / (np.linalg.norm(v[:2]) or 1e-9), Lw))
        Hn = homography(nl)
        if Hn is None:
            continue
        H = Ti @ Hn
        # convention: +X must project to the right of the image
        d = apply_h(H, [(1.0, 0.0)])[0] - apply_h(H, [(-1.0, 0.0)])[0]
        if not np.isfinite(d).all() or d[0] <= 0:
            continue
        iou = mat_iou(H, matm)
        if iou < 0.5:
            continue
        e = pixel_err(H, [(Lw, a, b) for L, Lw, a, b in pairs])
        score = e + 22.0 * (1.0 - iou)
        if np.isfinite(score) and (best is None or score < best[0]):
            best = (score, H, e, iou, len(pairs))
    if best is None:
        info["fail"] = "no consistent assignment"
        return None, info
    _, Hn, e, iou, npairs = best
    # scale from network resolution back to the original frame
    sh, sw = shape[:2]
    Sc = np.diag([sw / NET_W, sh / NET_H, 1.0])
    info.update(err_px=float(e), iou=float(iou), n_lines=npairs)
    return Sc @ Hn, info


def solve(net, device, bgr):
    cls, prob = predict(net, device, bgr)
    H, info = solve_from_cls(cls, bgr.shape)
    return H, info, cls
