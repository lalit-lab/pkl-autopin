"""Train the court segmentation net on synthetic data generated on the fly.

    python train.py --steps 24000 --batch 10

Data is rendered in worker processes, so the model never sees the same frame
twice and there is no dataset to store.  Real screenshots are used only as
background plates (crowd/stands), never as labels.
"""
import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth
from model import UNet, CLASS_WEIGHTS, IN_W, IN_H, NCLASS

HERE = os.path.dirname(os.path.abspath(__file__))
SHOTS = os.path.dirname(HERE)


def load_plates():
    """Crowd/stand strips from the real frames, used as backgrounds only."""
    out = []
    for f in sorted(glob.glob(os.path.join(SHOTS, "*.png"))):
        im = cv2.imread(f)
        if im is not None:
            out.append(im[: int(im.shape[0] * 0.30)].copy())
    return out


class SynthSet(IterableDataset):
    def __init__(self, plates, seed=0):
        self.plates = plates
        self.seed = seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        rng = np.random.default_rng(self.seed + 9973 * (wid + 1))
        mean = np.array([0.45, 0.43, 0.47], np.float32)
        std = np.array([0.26, 0.26, 0.27], np.float32)
        while True:
            s = synth.sample(rng, IN_W, IN_H, self.plates)
            if s is None:
                continue
            img, lab, _ = s
            x = img.astype(np.float32) / 255.0
            x = (x - mean) / std
            yield (torch.from_numpy(x.transpose(2, 0, 1).copy()),
                   torch.from_numpy(lab.astype(np.int64)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=24000)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(HERE, "court_seg.pt"))
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "")

    plates = load_plates()
    print("background plates:", len(plates))
    dl = DataLoader(SynthSet(plates), batch_size=a.batch, num_workers=a.workers,
                    pin_memory=(dev == "cuda"), persistent_workers=a.workers > 0,
                    prefetch_factor=4 if a.workers else None)

    net = UNet().to(dev)
    step0 = 0
    if a.resume and os.path.exists(a.out):
        ck = torch.load(a.out, map_location=dev)
        net.load_state_dict(ck["model"])
        step0 = ck.get("step", 0)
        print("resumed from step", step0)

    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=max(1, a.steps - step0), pct_start=0.15)
    lossf = nn.CrossEntropyLoss(
        weight=torch.tensor(CLASS_WEIGHTS, device=dev, dtype=torch.float32))

    net.train()
    t0 = time.time()
    run = []
    step = step0
    for x, y in dl:
        if step >= a.steps:
            break
        x = x.to(dev, non_blocking=True)
        y = y.to(dev, non_blocking=True)
        loss = lossf(net(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        run.append(float(loss))
        step += 1
        if step % 100 == 0:
            el = time.time() - t0
            ips = (step - step0) * a.batch / el
            print("step %6d/%d  loss %.4f  %.1f img/s  elapsed %.1fm"
                  % (step, a.steps, np.mean(run[-100:]), ips, el / 60), flush=True)
        if step % 1000 == 0 or step == a.steps:
            torch.save({"model": net.state_dict(), "step": step}, a.out)
    torch.save({"model": net.state_dict(), "step": step}, a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
