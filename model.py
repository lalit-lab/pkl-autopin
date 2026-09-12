"""Small U-Net for PKL court segmentation (9 classes, see synth.CLASS_NAMES).

Deliberately small: the input is a 448x256 frame of a high-contrast coloured
template, not natural imagery, so capacity is not the bottleneck - the
sim-to-real gap is.  A compact net also trains in hours on a 4GB GTX 1650.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

NCLASS = 9
IN_W, IN_H = 448, 256


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, nclass=NCLASS, base=24):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.d1, self.d2, self.d3 = block(3, c[0]), block(c[0], c[1]), block(c[1], c[2])
        self.bot = block(c[2], c[3])
        self.u3 = nn.ConvTranspose2d(c[3], c[2], 2, stride=2)
        self.c3 = block(c[3], c[2])
        self.u2 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2)
        self.c2 = block(c[2], c[1])
        self.u1 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2)
        self.c1 = block(c[1], c[0])
        self.head = nn.Conv2d(c[0], nclass, 1)

    def forward(self, x):
        x1 = self.d1(x)
        x2 = self.d2(F.max_pool2d(x1, 2))
        x3 = self.d3(F.max_pool2d(x2, 2))
        xb = self.bot(F.max_pool2d(x3, 2))
        y = self.c3(torch.cat([self.u3(xb), x3], 1))
        y = self.c2(torch.cat([self.u2(y), x2], 1))
        y = self.c1(torch.cat([self.u1(y), x1], 1))
        return self.head(y)


# Line classes are a tiny fraction of pixels; without this the net happily
# predicts "all court" and still scores a high pixel accuracy.
CLASS_WEIGHTS = [1.0, 1.5, 1.0, 12.0, 12.0, 12.0, 14.0, 12.0, 12.0]


def normalise(img_bgr_uint8, device):
    """HWC uint8 BGR -> 1x3xHxW float tensor on device."""
    import numpy as np
    x = torch.from_numpy(np.ascontiguousarray(img_bgr_uint8)).to(device)
    x = x.permute(2, 0, 1).float().div_(255.0)
    mean = torch.tensor([0.45, 0.43, 0.47], device=device).view(3, 1, 1)
    std = torch.tensor([0.26, 0.26, 0.27], device=device).view(3, 1, 1)
    return ((x - mean) / std).unsqueeze(0)
