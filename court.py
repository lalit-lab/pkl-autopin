"""Court detection -> world<->image homography, from a PKL broadcast frame.

The mat has concentric colour bands whose boundaries ARE court lines:
    blue surround | orange lobby | purple court | orange lobby | blue surround
so the sidelines (X=+-4) and mat edges (X=+-5) fall out of colour segmentation.
The inner markings (mid / baulk / bonus / end) come from Hough inside the mat.

Every detected line is then labelled by an exhaustive search over monotonic
assignments to the known court geometry, scored by pixel reprojection error.
Broadcast views are steeply oblique, so the two parallel families are separated
by vanishing point (RANSAC), never by screen angle.

World frame matches Tackle Mapper.html:  X across (-5..5), Y along (-6.5..6.5).
A world line is [a,b,c] with a*X + b*Y + c = 0.
"""
import cv2
import numpy as np
from itertools import combinations

# --- world constants -------------------------------------------------------
ACROSS_X = (-5.0, -4.0, 4.0, 5.0)          # mat edge, sideline, sideline, mat edge
DEPTH_Y = (-6.5, -4.75, -3.75, 0.0, 3.75, 4.75, 6.5)
MAX_ACROSS, MAX_DEPTH = 4, 5               # strongest lines kept per family
RESCORE_TOP = 60                           # candidates sent to photometric check
MAX_CHAMFER = 6.0                          # px; above this, refuse to guess
MIN_COVER = 0.12                           # court fraction that must be visible
MIN_IOU = 0.60                             # reprojected mat vs detected mat


def wline_x(x):
    return np.array([1.0, 0.0, -x])


def wline_y(y):
    return np.array([0.0, 1.0, -y])


# --- colour masks ----------------------------------------------------------
# Two venue palettes appear in PKL broadcasts: purple court + orange lobby, and
# pink court + yellow lobby.  One widened band covers both; the lobby is then
# constrained to components touching the court, which rejects same-hue sponsor
# boards and the red/green scoreboard banner.
PURPLE = ((128, 50, 50), (172, 255, 255))
ORANGE = ((5, 85, 85), (38, 255, 255))


def _clean(mask, ksize=7, close=2, open_=1):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=open_)
    return mask


def _largest(mask):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return ((lab == i) * 255).astype(np.uint8)


def _touching(mask, anchor, grow=25, min_area=400):
    """Keep only components of `mask` that touch the dilated `anchor`.

    This is what stops the red/green scoreboard banner (same hue band as the
    orange lobby) from being welded onto the mat.
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow))
    near = cv2.dilate(anchor, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        comp = lab == i
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        if np.any(near[comp]):
            out[comp] = 255
    return out


def _fill(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    if cnts:
        cv2.drawContours(out, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
    return out


# Broadcast graphics bands.  The scoreboard is the single worst source of false
# court structure: its red/green banner shares the lobby hue and its white text
# reads as line evidence, so it is cut out before anything else happens.
TOP_CUT, BOT_CUT = 0.055, 0.125


def roi_box(shape):
    h, w = shape[:2]
    return 0, int(round(h * TOP_CUT)), w, int(round(h * (1.0 - BOT_CUT)))


def masks(bgr):
    """Return (purple court, full mat) masks with players filled in."""
    h, w = bgr.shape[:2]
    x0, y0, x1, y1 = roi_box(bgr.shape)
    keep = np.zeros((h, w), np.uint8)
    keep[y0:y1, x0:x1] = 255
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    pur = cv2.bitwise_and(cv2.inRange(hsv, np.array(PURPLE[0], np.uint8),
                                      np.array(PURPLE[1], np.uint8)), keep)
    pur = _largest(_clean(pur))
    orn = cv2.bitwise_and(cv2.inRange(hsv, np.array(ORANGE[0], np.uint8),
                                      np.array(ORANGE[1], np.uint8)), keep)
    orn = _touching(_clean(orn), pur)
    mat = _largest(_clean(cv2.bitwise_or(pur, orn), 13, close=3, open_=0))
    return _fill(pur), _fill(mat)


def mat_quad(mat, shape, margin=10):
    """The mat outline as 4 corners, or None if it is clipped by the ROI.

    When the whole 10m x 13m mat is in frame this is by far the strongest cue
    available: four exact point correspondences instead of a labelling search.
    """
    x0, y0, x1, y1 = roi_box(shape)
    cnts, _ = cv2.findContours(mat, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    quad = None
    for eps in (0.010, 0.015, 0.020, 0.028, 0.040):
        p = cv2.approxPolyDP(c, eps * peri, True).reshape(-1, 2)
        if len(p) == 4:
            quad = p.astype(float)
            break
    if quad is None or not cv2.isContourConvex(quad.astype(np.int32)):
        return None
    for px, py in quad:                       # a corner on the ROI edge is a cut
        if (px < x0 + margin or px > x1 - margin
                or py < y0 + margin or py > y1 - margin):
            return None
    return quad


# --- line primitives -------------------------------------------------------
def _fit_line(pts):
    """Total-least-squares line through points -> [a,b,c] with |(a,b)|=1."""
    p = np.asarray(pts, float)
    c = p.mean(0)
    _, _, vt = np.linalg.svd(p - c)
    d = vt[0]
    n = np.array([-d[1], d[0]])
    n /= np.linalg.norm(n) or 1e-9
    return np.array([n[0], n[1], -n @ c])


def border_edges(mask, margin=10):
    """Polygon edges of the mask that are NOT clipped by the ROI border."""
    bx0, by0, bx1, by1 = roi_box(mask.shape)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    c = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    poly = cv2.approxPolyDP(c, 0.010 * peri, True).reshape(-1, 2).astype(float)
    out = []
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        if np.linalg.norm(b - a) < 0.07 * peri:
            continue
        mid = (a + b) / 2
        if (mid[0] < bx0 + margin or mid[0] > bx1 - margin
                or mid[1] < by0 + margin or mid[1] > by1 - margin):
            continue
        out.append((_fit_line([a, b]), a, b))
    return out


def inner_lines(bgr, mat, min_frac=0.16):
    """White + dark court markings inside the mat."""
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    core = cv2.erode(mat, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
                     iterations=2)
    white = cv2.inRange(hsv, np.array((0, 0, 145), np.uint8),
                        np.array((180, 80, 255), np.uint8))
    dark = cv2.inRange(hsv, np.array((0, 0, 0), np.uint8),
                       np.array((180, 255, 75), np.uint8))
    marks = cv2.bitwise_and(cv2.bitwise_or(white, dark), core)
    marks = cv2.morphologyEx(marks, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    segs = cv2.HoughLinesP(marks, 1, np.pi / 360, threshold=55,
                           minLineLength=int(min_frac * w), maxLineGap=int(0.05 * w))
    if segs is None:
        return []
    return [(_fit_line([(x1, y1), (x2, y2)]), np.array([x1, y1], float),
             np.array([x2, y2], float)) for x1, y1, x2, y2 in segs[:, 0]]


def merge(lines, ang_tol=4.0, off_tol=14.0):
    """Collapse near-duplicate lines, keeping the longest representative."""
    out = []
    for L, a, b in sorted(lines, key=lambda t: -np.linalg.norm(t[2] - t[1])):
        ang = np.degrees(np.arctan2(L[1], L[0])) % 180
        ma = np.array([a[0], a[1], 1.0])
        mb = np.array([b[0], b[1], 1.0])
        dup = False
        for L2, _, _ in out:
            ang2 = np.degrees(np.arctan2(L2[1], L2[0])) % 180
            da = min(abs(ang - ang2), 180 - abs(ang - ang2))
            if da < ang_tol and max(abs(L2 @ ma), abs(L2 @ mb)) < off_tol:
                dup = True
                break
        if not dup:
            out.append((L, a, b))
    return out


# --- vanishing-point family split -----------------------------------------
def _vp_residual(vp, a, b):
    """Angle (deg) between segment a->b and the ray from its midpoint to vp."""
    d = b - a
    nd = np.linalg.norm(d) or 1e-9
    if abs(vp[2]) < 1e-9:                       # vanishing point at infinity
        e = vp[:2]
    else:
        e = (vp[:2] / vp[2]) - (a + b) / 2
    ne = np.linalg.norm(e) or 1e-9
    c = abs(float(np.dot(d / nd, e / ne)))
    return float(np.degrees(np.arccos(min(1.0, c))))


def _consensus(lines, tol=2.2):
    """Largest set of lines sharing a vanishing point (exhaustive over pairs)."""
    best = []
    n = len(lines)
    for i, j in combinations(range(n), 2):
        vp = np.cross(lines[i][0], lines[j][0])
        if np.linalg.norm(vp) < 1e-12:
            continue
        inl = [k for k in range(n)
               if _vp_residual(vp, lines[k][1], lines[k][2]) < tol]
        if len(inl) > len(best):
            best = inl
    return best


def split_families(lines, tol=2.2):
    """Split into the two parallel families by vanishing point consensus."""
    if len(lines) < 4:
        return [], []
    fa = _consensus(lines, tol)
    rest = [i for i in range(len(lines)) if i not in set(fa)]
    fb = _consensus([lines[i] for i in rest], tol) if len(rest) >= 2 else []
    A = [lines[i] for i in fa]
    B = [lines[rest[i]] for i in fb]
    return A, B


# --- homography ------------------------------------------------------------
def homography(pairs):
    """pairs: [(image_line[3], world_line[3])] -> H mapping world -> image."""
    rows = []
    for l, L in pairs:
        M = np.zeros((3, 9))
        for i in range(3):
            for r in range(3):
                M[i, 3 * r + i] = l[r]
        Lx = np.array([[0, -L[2], L[1]], [L[2], 0, -L[0]], [-L[1], L[0], 0]], float)
        rows.extend((Lx @ M)[:2])
    A = np.asarray(rows)
    if A.shape[0] < 8:
        return None
    _, _, vt = np.linalg.svd(A)
    H = vt[-1].reshape(3, 3)
    return None if abs(np.linalg.det(H)) < 1e-12 else H


def _norm_T(shape):
    h, w = shape[:2]
    s = float(np.hypot(w, h))
    T = np.array([[1 / s, 0, -w / (2 * s)], [0, 1 / s, -h / (2 * s)], [0, 0, 1.0]])
    Ti = np.array([[s, 0, w / 2.0], [0, s, h / 2.0], [0, 0, 1.0]])
    return T, Ti


def apply_h(H, pts):
    p = np.atleast_2d(np.asarray(pts, float))
    q = np.hstack([p, np.ones((len(p), 1))]) @ H.T
    w = np.where(np.abs(q[:, 2:3]) < 1e-12, 1e-12, q[:, 2:3])
    return q[:, :2] / w


def pixel_err(H, items):
    """Mean pixel distance from detected segment endpoints to reprojected lines.

    items: [(world_line, endpoint_a, endpoint_b)] in ORIGINAL pixel coords.
    """
    try:
        Hit = np.linalg.inv(H).T
    except np.linalg.LinAlgError:
        return np.inf
    tot = 0.0
    for L, a, b in items:
        l = Hit @ L
        nr = np.linalg.norm(l[:2])
        if nr < 1e-12:
            return np.inf
        l = l / nr
        tot += 0.5 * (abs(l @ np.array([a[0], a[1], 1.0]))
                      + abs(l @ np.array([b[0], b[1], 1.0])))
    return tot / len(items)


CHAMFER_CAP = 18.0                         # px; beyond this a line is just wrong


def evidence_map(bgr, pur, mat):
    """Thin map of everything that could BE a court line: markings + band edges.

    Kept deliberately thin and white-only.  An earlier version dilated heavily
    and included dark pixels, which let players' shorts and shadows masquerade
    as line evidence and made the score undiscriminating.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, np.array((0, 0, 140), np.uint8),
                        np.array((180, 85, 255), np.uint8))
    inside = cv2.erode(mat, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    ev = cv2.bitwise_and(white, inside)
    ev = cv2.morphologyEx(ev, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    for m in (pur, mat):                       # colour-band boundaries are lines too
        ev = cv2.bitwise_or(ev, cv2.Canny(m, 50, 150))
    return ev


def chamfer(H, dt, mat, n=80):
    """Mean distance (px) from reprojected court lines to the nearest evidence.

    Pure algebraic residual cannot tell a correct labelling from one that is
    self-consistent but sitting on the wrong lines.  A chamfer score can,
    because a mislabelled fit paints lines across blank mat, far from any
    evidence.  Returns (mean_distance, coverage) - coverage is the fraction of
    the court that actually falls inside the visible mat.
    """
    h, w = dt.shape
    segs = ([((x, -6.5), (x, 6.5)) for x in ACROSS_X]
            + [((-5.0, y), (5.0, y)) for y in DEPTH_Y])
    vals, seen, total = [], 0, 0
    for a, b in segs:
        t = np.linspace(0, 1, n)[:, None]
        pts = apply_h(H, np.array(a) * (1 - t) + np.array(b) * t)
        total += n
        if not np.isfinite(pts).all():
            continue
        xi = np.round(pts[:, 0]).astype(int)
        yi = np.round(pts[:, 1]).astype(int)
        ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        if not ok.any():
            continue
        xi, yi = xi[ok], yi[ok]
        inmat = mat[yi, xi] > 0
        if inmat.sum() < 10:
            continue
        xi, yi = xi[inmat], yi[inmat]
        seen += len(xi)
        vals.append(np.minimum(dt[yi, xi], CHAMFER_CAP))
    if not vals or seen < 60:
        return CHAMFER_CAP, 0.0
    return float(np.concatenate(vals).mean()), seen / float(total)


def mat_iou(H, mat, shape):
    """IoU of the reprojected 10m x 13m mat against the detected mat mask.

    This is the constraint that kills near-degenerate fits.  A homography can
    fan every court line onto real white pixels and still be nonsense; it
    cannot also make the mat OUTLINE land on the mat.  Only the visible ROI is
    compared, so frames where the mat runs out of shot are not penalised.
    """
    x0, y0, x1, y1 = roi_box(shape)
    c = apply_h(H, [(-5.0, -6.5), (5.0, -6.5), (5.0, 6.5), (-5.0, 6.5)])
    if not np.isfinite(c).all() or np.abs(c).max() > 1e6:
        return 0.0
    proj = np.zeros(mat.shape, np.uint8)
    cv2.fillConvexPoly(proj, c.astype(np.int32), 255)
    roi = np.zeros(mat.shape, np.uint8)
    roi[y0:y1, x0:x1] = 255
    proj = cv2.bitwise_and(proj, roi)
    ref = cv2.bitwise_and(mat, roi)
    inter = int(np.count_nonzero(cv2.bitwise_and(proj, ref)))
    union = int(np.count_nonzero(cv2.bitwise_or(proj, ref)))
    return inter / union if union else 0.0


def support(H, ev, mat, n=60):
    """Backwards-compatible hit-rate score (kept for reporting)."""
    dt = cv2.distanceTransform(255 - ev, cv2.DIST_L2, 3)
    d, cov = chamfer(H, dt, mat, n)
    return max(0.0, 1.0 - d / CHAMFER_CAP) * min(1.0, cov * 3.0)


def _area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _convex(p):
    s = [np.sign(np.cross(p[(i + 1) % 4] - p[i], p[(i + 2) % 4] - p[(i + 1) % 4]))
         for i in range(4)]
    return abs(sum(s)) == 4


def _plausible(H, w, h):
    """Reject homographies that fold the court or blow it up."""
    c = apply_h(H, [(-5, -6.5), (5, -6.5), (5, 6.5), (-5, 6.5)])
    if not np.isfinite(c).all():
        return False
    a = _area(c)
    if a < 0.06 * w * h or a > 30.0 * w * h:
        return False
    return _convex(c) and np.abs(c).max() < 40 * max(w, h)


def _subsets(n, lo, drop=1):
    """Full set plus every way of dropping up to `drop` lines (outlier tolerance).

    Searching *all* subsets is what makes this intractable: keeping only the
    near-complete ones cuts the space ~10x with no practical loss, since the
    detector rarely produces more than one bad line per family.
    """
    out = []
    for k in range(max(lo, n - drop), n + 1):
        out.extend(combinations(range(n), k))
    return out


def solve(bgr, tol=2.2):
    """Full auto: image -> (H mapping world->image, info dict)."""
    h, w = bgr.shape[:2]
    T, Ti = _norm_T(bgr.shape)
    pur, mat = masks(bgr)
    info = {}

    ev = evidence_map(bgr, pur, mat)
    dt = cv2.distanceTransform(255 - ev, cv2.DIST_L2, 3)

    # --- fast path: the whole mat is in frame -> exact 4-point homography ----
    quad = mat_quad(mat, bgr.shape)
    if quad is not None:
        corners = np.array([(-5.0, -6.5), (5.0, -6.5), (5.0, 6.5), (-5.0, 6.5)])
        best = None
        for flip in (False, True):
            q = quad[::-1] if flip else quad
            for r in range(4):
                src = corners.astype(np.float32)
                dst = np.roll(q, r, axis=0).astype(np.float32)
                Hq = cv2.getPerspectiveTransform(src, dst)
                if not _plausible(Hq, w, h):
                    continue
                iou = mat_iou(Hq, mat, bgr.shape)
                if iou < MIN_IOU:
                    continue
                d, cov = chamfer(Hq, dt, mat)
                if best is None or d < best[0]:
                    best = (d, Hq, cov, iou)
        if best is not None and best[0] <= MAX_CHAMFER:
            info.update(method="mat-quad", chamfer=float(best[0]),
                        cover=float(best[2]), iou=float(best[3]),
                        err_px=0.0, n_lines=4)
            return best[1], info

    # --- general path: label detected lines ---------------------------------
    cand = merge(border_edges(pur) + border_edges(mat) + inner_lines(bgr, mat))
    A, B = split_families(cand, tol)
    if len(A) < 2 or len(B) < 2 or len(A) + len(B) < 5:
        info["fail"] = "families too small (%d / %d of %d lines)" % (
            len(A), len(B), len(cand))
        return None, info

    longest = lambda g, m: sorted(g, key=lambda t: -np.linalg.norm(t[2] - t[1]))[:m]
    ctr = np.array([w / 2.0, h / 2.0, 1.0])
    Tit = np.linalg.inv(T).T

    def prep(group, m):
        g = sorted(longest(group, m), key=lambda t: t[0] @ ctr)
        nl = []
        for L, a, b in g:
            v = Tit @ L
            nl.append(v / (np.linalg.norm(v[:2]) or 1e-9))
        return g, nl

    cands = []
    # which family is "across" is unknown -> try both ways round
    for gA, gB in ((A, B), (B, A)):
        ga, na = prep(gA, MAX_ACROSS)
        gd, nd = prep(gB, MAX_DEPTH)
        for sa in _subsets(len(na), 2):
            for sd in _subsets(len(nd), 2):
                if len(sa) + len(sd) < 5:
                    continue
                la = [na[i] for i in sa]
                ld = [nd[i] for i in sd]
                for xs in combinations(ACROSS_X, len(sa)):
                    for ys in combinations(DEPTH_Y, len(sd)):
                        for sx in (1, -1):
                            for sy in (1, -1):
                                wx = [wline_x(sx * x) for x in (xs[::-1] if sx < 0 else xs)]
                                wy = [wline_y(sy * y) for y in (ys[::-1] if sy < 0 else ys)]
                                H = homography(list(zip(la, wx)) + list(zip(ld, wy)))
                                if H is None:
                                    continue
                                Hn = Ti @ H
                                if not _plausible(Hn, w, h):
                                    continue
                                items = ([(L, ga[i][1], ga[i][2]) for L, i in zip(wx, sa)]
                                         + [(L, gd[i][1], gd[i][2]) for L, i in zip(wy, sd)])
                                e = pixel_err(Hn, items)
                                if np.isfinite(e):
                                    cands.append((e, Hn, len(sa) + len(sd),
                                                  [sx * x for x in xs],
                                                  [sy * y for y in ys]))
    if not cands:
        info["fail"] = "no plausible labelling"
        return None, info

    # Photometric rescoring of the strongest algebraic candidates.
    cands.sort(key=lambda t: t[0] / (1.0 + 0.25 * (t[2] - 5)))
    best = None
    for e, Hn, nused, xs, ys in cands[:RESCORE_TOP]:
        iou = mat_iou(Hn, mat, bgr.shape)
        if iou < MIN_IOU:
            continue
        d, cov = chamfer(Hn, dt, mat)
        if cov < MIN_COVER:
            continue
        # outline agreement first, then line chamfer; algebraic residual breaks ties
        score = d + 14.0 * (1.0 - iou) + 0.05 * min(e, 40.0) - 0.4 * (nused - 5)
        if best is None or score < best[0]:
            best = (score, Hn, e, d, cov, iou, nused, xs, ys)

    if best is None:
        info["fail"] = "no candidate matches the mat outline"
        return None, info
    _, H, e, d, cov, iou, nused, xs, ys = best
    info.update(method="lines", err_px=float(e), chamfer=float(d),
                cover=float(cov), iou=float(iou), n_lines=nused,
                across_labels=xs, depth_labels=ys, n_cand=len(cand),
                fam=(len(A), len(B)))
    if d > MAX_CHAMFER:
        info["fail"] = "lines do not match court (chamfer %.1f px)" % d
        return None, info
    return H, info
