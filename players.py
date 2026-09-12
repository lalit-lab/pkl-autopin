"""Player detection -> which one is the raider, and where his feet are.

Teams are separated by jersey colour rather than by any trained classifier:
in a kabaddi frame there are exactly two kits, so a 2-way clustering of torso
hue is both sufficient and robust to whatever the two teams happen to wear.

The raider is then the player most surrounded by the OTHER kit.  That is the
definition of a raid: one attacker inside the defending half with defenders
closing on him, which is precisely the frame a tackle screenshot captures.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from court import apply_h

MAT_X, MAT_Y = 5.4, 6.9          # a little slack outside the painted mat


def load_detector(weights="yolo11n.pt"):
    from ultralytics import YOLO
    return YOLO(weights)


def detect(model, bgr, conf=0.25):
    """Person boxes as (x1, y1, x2, y2, conf)."""
    r = model.predict(bgr, classes=[0], conf=conf, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []
    xyxy = r.boxes.xyxy.cpu().numpy()
    cf = r.boxes.conf.cpu().numpy()
    return [(float(a), float(b), float(c), float(d), float(e))
            for (a, b, c, d), e in zip(xyxy, cf)]


def torso_colour(bgr, box):
    """Median hue/sat of the shirt region, ignoring washed-out pixels."""
    x1, y1, x2, y2 = box[:4]
    h = y2 - y1
    w = x2 - x1
    cy0 = int(y1 + 0.18 * h)
    cy1 = int(y1 + 0.52 * h)
    cx0 = int(x1 + 0.22 * w)
    cx1 = int(x2 - 0.22 * w)
    H, W = bgr.shape[:2]
    cy0, cy1 = max(0, cy0), min(H, max(cy1, cy0 + 1))
    cx0, cx1 = max(0, cx0), min(W, max(cx1, cx0 + 1))
    crop = bgr[cy0:cy1, cx0:cx1]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    m = (hsv[..., 1] > 60) & (hsv[..., 2] > 50)
    if m.sum() < 12:
        m = np.ones(hsv.shape[:2], bool)
    hh = hsv[..., 0][m].astype(np.float32) * (np.pi / 90.0)   # hue -> radians
    return np.array([np.cos(hh).mean(), np.sin(hh).mean(),
                     hsv[..., 1][m].mean() / 255.0])


def split_teams(feats):
    """2-means on circular hue features -> team index per player."""
    X = np.asarray(feats, np.float32)
    if len(X) < 2:
        return np.zeros(len(X), int)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2)
    _, lab, _ = cv2.kmeans(X, 2, None, crit, 8, cv2.KMEANS_PP_CENTERS)
    return lab.ravel()


def feet_world(H, box):
    """Bottom-centre of the box, projected onto the mat plane."""
    x1, y1, x2, y2 = box[:4]
    foot = np.array([(x1 + x2) / 2.0, y2])
    Hi = np.linalg.inv(H)
    w = apply_h(Hi, [foot])[0]
    return w, foot


def find_raider(bgr, H, boxes, radius=2.6):
    """Return (raider_index, world_positions, teams) or (None, ...)."""
    if not boxes:
        return None, [], []
    feats, world, keep = [], [], []
    for i, b in enumerate(boxes):
        c = torso_colour(bgr, b)
        if c is None:
            continue
        w, _ = feet_world(H, b)
        if not np.isfinite(w).all() or abs(w[0]) > MAT_X or abs(w[1]) > MAT_Y:
            continue
        feats.append(c)
        world.append(w)
        keep.append(i)
    if len(keep) < 2:
        return None, [], []
    teams = split_teams(feats)
    world = np.asarray(world)

    best, best_score = None, -1.0
    for j in range(len(keep)):
        d = np.linalg.norm(world - world[j], axis=1)
        near = (d < radius) & (d > 1e-9)
        if near.sum() == 0:
            continue
        opp = int((teams[near] != teams[j]).sum())
        same = int((teams[near] == teams[j]).sum())
        # a raider is ringed by the other kit; ties broken by tighter encirclement
        score = opp - 0.75 * same + 0.001 * opp / (d[near].mean() + 1e-6)
        if score > best_score:
            best_score, best = score, j
    if best is None or best_score <= 0:
        return None, world, teams
    return keep[best], world, teams
