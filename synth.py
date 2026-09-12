"""Synthetic PKL court renderer -> training data for the segmentation model.

The kabaddi mat is a known flat template: a 10m x 13m rectangle with lines at
fixed positions.  That means unlimited labelled training data can be rendered
by projecting the template through random but plausible broadcast cameras and
compositing players, crowd and scoreboard on top.  No hand labelling.

The label is a per-pixel CLASS, and lines carry their identity (mid / baulk /
bonus / end / sideline / mat edge).  That is the whole point: the classical
pipeline could find lines but not tell which was which, and the depth scale is
under-determined without that.  A model that names each line removes the
ambiguity.

Speed matters as much as realism here - training is data-bound, not GPU-bound -
so the mat and the surrounding floor are rendered as two small templates at
different resolutions and warped separately, and lighting is applied once to
the finished frame rather than to the template.
"""
import cv2
import numpy as np

# --- geometry (metres), matching Tackle Mapper.html -------------------------
HALF_W, HALF_L = 5.0, 6.5           # mat is 10m across x 13m long
SIDELINE_X = 4.0
BAULK_Y, BONUS_Y, END_Y = 3.75, 4.75, 6.5

# --- label classes ---------------------------------------------------------
BG, LOBBY, COURT, L_MATEDGE, L_SIDE, L_END, L_BONUS, L_BAULK, L_MID = range(9)
NCLASS = 9
CLASS_NAMES = ["bg", "lobby", "court", "matedge", "sideline",
               "end", "bonus", "baulk", "mid"]
# world lines each line-class stands for, used at inference time
LINE_WORLD = {
    L_MATEDGE: [("x", -HALF_W), ("x", HALF_W)],
    L_SIDE: [("x", -SIDELINE_X), ("x", SIDELINE_X)],
    L_END: [("y", -END_Y), ("y", END_Y)],
    L_BONUS: [("y", -BONUS_Y), ("y", BONUS_Y)],
    L_BAULK: [("y", -BAULK_Y), ("y", BAULK_Y)],
    L_MID: [("y", 0.0)],
}

MAT_PPM = 32.0                      # mat template resolution, px per metre
FLOOR_PPM = 8.0                     # surround floor, much coarser
SURROUND_M = 7.0                    # floor rendered around the mat, metres
LINE_W = 0.09                       # painted line width, metres
MID_W = 0.13

MAT_W = int(round(2 * HALF_W * MAT_PPM))
MAT_H = int(round(2 * HALF_L * MAT_PPM))
FL_W = int(round(2 * (HALF_W + SURROUND_M) * FLOOR_PPM))
FL_H = int(round(2 * (HALF_L + SURROUND_M) * FLOOR_PPM))

# template pixel -> world metres
S_MAT = np.array([[1 / MAT_PPM, 0, -HALF_W], [0, -1 / MAT_PPM, HALF_L], [0, 0, 1]])
S_FLOOR = np.array([[1 / FLOOR_PPM, 0, -(HALF_W + SURROUND_M)],
                    [0, -1 / FLOOR_PPM, HALF_L + SURROUND_M], [0, 0, 1]])


def _hsv_bgr(h, s, v):
    """One HSV triple -> a BGR triple, without converting a whole image."""
    px = np.array([[[h, s, v]]], np.uint8)
    return cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0, 0].tolist()


def _mx(x):
    return int(round((x + HALF_W) * MAT_PPM))


def _my(y):
    return int(round((HALF_L - y) * MAT_PPM))


def mat_template(rng):
    """Top-down mat: (bgr, label). Colours randomised over both venue palettes."""
    court = _hsv_bgr(rng.uniform(132, 168), rng.uniform(95, 165), rng.uniform(145, 225))
    lobby = _hsv_bgr(rng.uniform(8, 34), rng.uniform(140, 235), rng.uniform(165, 245))
    img = np.full((MAT_H, MAT_W, 3), court, np.uint8)
    lab = np.full((MAT_H, MAT_W), COURT, np.uint8)

    lx, rx = _mx(-SIDELINE_X), _mx(SIDELINE_X)
    img[:, :lx] = lobby
    img[:, rx:] = lobby
    lab[:, :lx] = LOBBY
    lab[:, rx:] = LOBBY

    wv = rng.uniform(232, 252)
    white = [wv, wv, wv]
    white[int(rng.integers(0, 3))] *= rng.uniform(0.94, 1.0)
    dv = rng.uniform(14, 40)
    dark = [dv, dv, dv]

    def vline(x, cls, col, width=LINE_W):
        a, b = _mx(x - width / 2), _mx(x + width / 2)
        a, b = max(0, a), min(MAT_W, max(b, a + 2))
        img[:, a:b] = col
        lab[:, a:b] = cls

    def hline(y, cls, col, width=LINE_W):
        a, b = _my(y + width / 2), _my(y - width / 2)
        a, b = max(0, a), min(MAT_H, max(b, a + 2))
        img[a:b, :] = col
        lab[a:b, :] = cls

    for x in (-HALF_W, HALF_W):
        vline(x, L_MATEDGE, white)
    for x in (-SIDELINE_X, SIDELINE_X):
        vline(x, L_SIDE, white)
    for y in (-END_Y, END_Y):
        hline(y, L_END, white)
    # the bonus line is painted dark on a real mat; keep that cue, but not always
    bonus_col = dark if rng.random() < 0.75 else white
    for y in (-BONUS_Y, BONUS_Y):
        hline(y, L_BONUS, bonus_col)
    for y in (-BAULK_Y, BAULK_Y):
        hline(y, L_BAULK, white)
    hline(0.0, L_MID, white, MID_W)

    # panel seams and wear, so the net does not read every bright streak as a line
    for _ in range(int(rng.integers(0, 7))):
        y0 = int(rng.integers(0, MAT_H))
        col = np.clip(img[y0, MAT_W // 2].astype(int) + rng.integers(-18, 18, 3), 0, 255)
        cv2.line(img, (0, y0), (MAT_W, y0), tuple(int(v) for v in col), 1)
    return img, lab


def floor_template(rng):
    """The blue/grey sports floor the mat sits on."""
    col = _hsv_bgr(rng.uniform(95, 125), rng.uniform(60, 210), rng.uniform(60, 190))
    img = np.full((FL_H, FL_W, 3), col, np.uint8)
    for _ in range(int(rng.integers(0, 4))):        # floor markings / seams
        y0 = int(rng.integers(0, FL_H))
        c = np.clip(np.array(col) + rng.integers(-45, 45, 3), 0, 255)
        cv2.line(img, (0, y0), (FL_W, y0), tuple(int(v) for v in c), 1)
    return img


# --- camera ----------------------------------------------------------------
def random_homography(rng, W, H):
    """A plausible broadcast camera as a world-plane -> image homography."""
    corners = np.array([[-HALF_W, -HALF_L, 0], [HALF_W, -HALF_L, 0],
                        [HALF_W, HALF_L, 0], [-HALF_W, HALF_L, 0]])
    for _ in range(200):
        fov = np.radians(rng.uniform(18, 52))
        f = (W / 2) / np.tan(fov / 2)
        K = np.array([[f, 0, W / 2 + rng.uniform(-20, 20)],
                      [0, f, H / 2 + rng.uniform(-20, 20)],
                      [0, 0, 1.0]])
        dist = rng.uniform(11, 34)
        az = np.radians(rng.uniform(-52, 52)) + rng.choice([0.0, np.pi])
        elev = np.radians(rng.uniform(14, 52))
        C = np.array([dist * np.cos(elev) * np.sin(az),
                      dist * np.cos(elev) * np.cos(az),
                      dist * np.sin(elev)])
        look = np.array([rng.uniform(-2.5, 2.5), rng.uniform(-5.0, 5.0), 0.0])
        fwd = look - C
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(right) < 1e-6:
            continue
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        roll = np.radians(rng.uniform(-4, 4))
        cr, sr = np.cos(roll), np.sin(roll)
        right, down = cr * right + sr * down, -sr * right + cr * down
        R = np.stack([right, down, fwd])
        P = K @ np.hstack([R, (-R @ C).reshape(3, 1)])
        Hm = P[:, [0, 1, 3]]
        if abs(np.linalg.det(Hm)) < 1e-9:
            continue
        if (R[2] @ (corners - C).T <= 0.5).any():   # mat must be in front
            continue
        c = _proj(Hm, corners[:, :2])
        if not np.isfinite(c).all():
            continue
        vis = _clip_area(c, W, H)
        if vis < 0.18 * W * H or vis > 0.995 * W * H:
            continue
        return Hm
    return None


def _proj(H, pts):
    p = np.asarray(pts, float)
    q = np.hstack([p, np.ones((len(p), 1))]) @ H.T
    w = np.where(np.abs(q[:, 2:3]) < 1e-12, 1e-12, q[:, 2:3])
    return q[:, :2] / w


def _clip_area(quad, W, H):
    m = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(m, quad.astype(np.int32), 255)
    return int(np.count_nonzero(m))


# --- scene dressing --------------------------------------------------------
def crowd_bg(rng, W, H, real_bgs=None):
    """Stand / crowd behind the floor."""
    if real_bgs is not None and len(real_bgs) and rng.random() < 0.55:
        src = real_bgs[int(rng.integers(0, len(real_bgs)))]
        bg = cv2.resize(src, (W, H))
        return cv2.convertScaleAbs(bg, alpha=rng.uniform(0.8, 1.2),
                                   beta=rng.uniform(-20, 20))
    bg = np.zeros((H, W, 3), np.uint8)
    base = np.array([rng.uniform(40, 120), rng.uniform(30, 90), rng.uniform(25, 80)])
    bg[:] = base
    for _ in range(int(rng.integers(3, 9))):       # sponsor / stand bands
        y0 = int(rng.integers(0, H))
        y1 = min(H, y0 + int(rng.integers(8, 70)))
        bg[y0:y1] = np.clip(base + rng.integers(-70, 90, 3), 0, 255)
    n = int(rng.integers(300, 2500))               # crowd speckle
    bg[rng.integers(0, H, n), rng.integers(0, W, n)] = rng.integers(0, 255, (n, 3))
    return cv2.GaussianBlur(bg, (0, 0), rng.uniform(1.0, 3.5))


def add_players(img, lab, H, rng, n=None):
    """Occluders standing on the mat, scaled correctly by the homography.

    Includes a clustered "pile" most of the time, because that is exactly what
    sits on the contact point in a real tackle frame and is the occlusion the
    model has to survive.
    """
    if n is None:
        n = int(rng.integers(4, 16))
    spots = [(rng.uniform(-HALF_W, HALF_W), rng.uniform(-HALF_L, HALF_L))
             for _ in range(n)]
    if rng.random() < 0.75:
        px, py = rng.uniform(-4, 4), rng.uniform(-6, 6)
        spots += [(px + rng.normal(0, 0.6), py + rng.normal(0, 0.6))
                  for _ in range(int(rng.integers(2, 6)))]
    for X, Y in spots:
        foot = _proj(H, [(X, Y)])[0]
        near = _proj(H, [(X, Y + 0.02)])[0]
        scale = np.linalg.norm(near - foot) / 0.02
        hp = rng.uniform(1.5, 1.9) * scale
        if not np.isfinite(hp) or hp < 8 or hp > 4 * img.shape[0]:
            continue
        wp = hp * rng.uniform(0.22, 0.45)
        col = tuple(int(v) for v in rng.integers(0, 255, 3))
        cx, cy = int(foot[0]), int(foot[1] - hp / 2)
        cv2.ellipse(img, (cx, cy), (max(1, int(wp / 2)), max(1, int(hp / 2))),
                    rng.uniform(-25, 25), 0, 360, col, -1)
        cv2.circle(img, (int(foot[0]), int(foot[1] - hp)), max(2, int(wp * 0.28)),
                   tuple(int(v) for v in rng.integers(20, 140, 3)), -1)
        cv2.ellipse(lab, (cx, cy), (max(1, int(wp / 2)), max(1, int(hp / 2))),
                    0, 0, 360, int(BG), -1)
    return img, lab


def add_overlays(img, lab, rng):
    """Scoreboard bar and channel logo - present in every real frame."""
    H, W = img.shape[:2]
    if rng.random() < 0.9:
        y0 = int(H * rng.uniform(0.83, 0.93))
        cv2.rectangle(img, (0, y0), (W, H),
                      tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
        for _ in range(int(rng.integers(2, 6))):
            x0 = int(rng.integers(0, max(1, W - 40)))
            cv2.rectangle(img, (x0, y0 + 4), (x0 + int(rng.integers(20, 120)), H - 4),
                          tuple(int(v) for v in rng.integers(120, 255, 3)), -1)
        lab[y0:, :] = BG
    if rng.random() < 0.7:
        s = int(min(H, W) * rng.uniform(0.05, 0.13))
        cv2.rectangle(img, (8, 8), (8 + s * 2, 8 + s),
                      tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
        lab[8:8 + s, 8:8 + s * 2] = BG
    return img, lab


def _lighting(img, rng):
    """Arena lighting, computed at low resolution and upsampled."""
    H, W = img.shape[:2]
    sh, sw = 32, 56
    gy, gx = np.mgrid[0:sh, 0:sw].astype(np.float32)
    light = (1.0 + 0.16 * (gx / sw - 0.5) * rng.uniform(-1, 1)
             + 0.16 * (gy / sh - 0.5) * rng.uniform(-1, 1))
    for _ in range(int(rng.integers(0, 3))):
        cx, cy = rng.uniform(0, sw), rng.uniform(0, sh)
        r = rng.uniform(0.25, 0.7) * sw
        light += rng.uniform(0.08, 0.26) * np.exp(
            -((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * r * r))
    light = cv2.resize(light, (W, H), interpolation=cv2.INTER_LINEAR)
    return np.clip(img.astype(np.float32) * light[..., None], 0, 255).astype(np.uint8)


def sample(rng, W=448, H=256, real_bgs=None):
    """One synthetic (image, label, H) triple, or None if the camera was rejected."""
    Hm = random_homography(rng, W, H)
    if Hm is None:
        return None
    img = crowd_bg(rng, W, H, real_bgs)
    lab = np.full((H, W), BG, np.uint8)

    # floor first, then the mat on top of it
    fl = cv2.warpPerspective(floor_template(rng), Hm @ S_FLOOR, (W, H))
    fa = cv2.warpPerspective(np.full((FL_H, FL_W), 255, np.uint8),
                             Hm @ S_FLOOR, (W, H), flags=cv2.INTER_NEAREST)
    m = fa > 127
    img[m] = fl[m]

    mt, ml = mat_template(rng)
    mw = cv2.warpPerspective(mt, Hm @ S_MAT, (W, H))
    mlw = cv2.warpPerspective(ml, Hm @ S_MAT, (W, H), flags=cv2.INTER_NEAREST)
    ma = cv2.warpPerspective(np.full((MAT_H, MAT_W), 255, np.uint8),
                             Hm @ S_MAT, (W, H), flags=cv2.INTER_NEAREST)
    m = ma > 127
    img[m] = mw[m]
    lab[m] = mlw[m]

    img, lab = add_players(img, lab, Hm, rng)
    img = _lighting(img, rng)
    img, lab = add_overlays(img, lab, rng)

    # photometric nuisance
    img = cv2.convertScaleAbs(img, alpha=rng.uniform(0.75, 1.3),
                              beta=rng.uniform(-28, 28))
    if rng.random() < 0.5:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.4, 1.8))
    if rng.random() < 0.6:
        img = np.clip(img.astype(np.int16)
                      + rng.normal(0, rng.uniform(2, 11), img.shape),
                      0, 255).astype(np.uint8)
    if rng.random() < 0.4:
        q = int(rng.integers(35, 92))
        img = cv2.imdecode(cv2.imencode(".jpg", img,
                                        [cv2.IMWRITE_JPEG_QUALITY, q])[1],
                           cv2.IMREAD_COLOR)
    return img, lab, Hm
