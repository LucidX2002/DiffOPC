import sys

sys.path.append(".")
import time

import numpy as np
import torch
import torch.nn.functional as func

from src.data.datatype import REALTYPE
from src.data.loaders import glp_seg
from src.litho.simple import LithoSim
from src.utils.adabox_rectangles import mask_to_rectangles


class Basic:
    def __init__(
        self,
        litho: LithoSim,
        device: torch.device,
        thresh=0.5,
    ):
        self._thresh = thresh
        self._device = device
        self._litho = litho.to(self._device)

    def run(self, mask, target, scale=1):
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=REALTYPE, device=self._device)
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, dtype=REALTYPE, device=self._device)
        with torch.no_grad():
            mask[mask >= self._thresh] = 1.0
            mask[mask < self._thresh] = 0.0
            if scale != 1:
                mask = torch.nn.functional.interpolate(mask[None, None, :, :], scale_factor=scale, mode="nearest")[0, 0]
            printedNom, printedMax, printedMin = self._litho(mask)
            binaryNom = torch.zeros_like(printedNom)
            binaryMax = torch.zeros_like(printedMax)
            binaryMin = torch.zeros_like(printedMin)
            binaryNom[printedNom >= self._thresh] = 1
            binaryMax[printedMax >= self._thresh] = 1
            binaryMin[printedMin >= self._thresh] = 1
            # l2loss = func.mse_loss(binaryNom, target, reduction="sum")
            l2_val = (binaryNom - target).abs().sum()
            pvband = torch.sum(binaryMax != binaryMin)
        return l2_val.item(), pvband.item()

    def sim(self, mask, target, scale=1):
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=REALTYPE, device=self._device)
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, dtype=REALTYPE, device=self._device)
        with torch.no_grad():
            mask[mask >= self._thresh] = 1.0
            mask[mask < self._thresh] = 0.0
            if scale != 1:
                mask = torch.nn.functional.interpolate(mask[None, None, :, :], scale_factor=scale, mode="nearest")[0, 0]
            printedNom, printedMax, printedMin = self._litho(mask)
            binaryNom = torch.zeros_like(printedNom)
            binaryMax = torch.zeros_like(printedMax)
            binaryMin = torch.zeros_like(printedMin)
            binaryNom[printedNom >= self._thresh] = 1
            binaryMax[printedMax >= self._thresh] = 1
            binaryMin[printedMin >= self._thresh] = 1
            l2loss = func.mse_loss(binaryNom, target, reduction="sum")
            pvband = torch.sum(binaryMax != binaryMin)
        return mask, binaryNom


EPE_CONSTRAINT = 15
EPE_CHECK_INTERVEL = 40
MIN_EPE_CHECK_LENGTH = 80
EPE_CHECK_START_INTERVEL = 40


def boundaries(target):
    boundary = torch.zeros_like(target)
    corner = torch.zeros_like(target)
    vertical = torch.zeros_like(target)
    horizontal = torch.zeros_like(target)

    padded = func.pad(target[None, None, :, :], pad=(1, 1, 1, 1))[0, 0]
    upper = padded[2:, 1:-1] == 1
    lower = padded[:-2, 1:-1] == 1
    left = padded[1:-1, :-2] == 1
    right = padded[1:-1, 2:] == 1
    upperleft = padded[2:, :-2] == 1
    upperright = padded[2:, 2:] == 1
    lowerleft = padded[:-2, :-2] == 1
    lowerright = padded[:-2, 2:] == 1
    boundary = target == 1
    boundary[upper & lower & left & right & upperleft & upperright & lowerleft & lowerright] = False

    padded = func.pad(boundary[None, None, :, :], pad=(1, 1, 1, 1))[0, 0]
    upper = padded[2:, 1:-1] == 1
    lower = padded[:-2, 1:-1] == 1
    left = padded[1:-1, :-2] == 1
    right = padded[1:-1, 2:] == 1
    center = padded[1:-1, 1:-1] == 1

    vertical = center.clone()
    vertical[left & right] = False
    vsites = vertical.nonzero()
    vindices = np.lexsort((vsites[:, 0].detach().cpu().numpy(), vsites[:, 1].detach().cpu().numpy()))
    vsites = vsites[vindices]
    vstart = torch.cat(
        (
            torch.tensor([True], device=vsites.device),
            vsites[:, 0][1:] != vsites[:, 0][:-1] + 1,
        )
    )
    vend = torch.cat(
        (
            vsites[:, 0][1:] != vsites[:, 0][:-1] + 1,
            torch.tensor([True], device=vsites.device),
        )
    )
    # vstart = vsites[(vstart == True).nonzero()[:, 0], :]
    vstart = vsites[vstart, :]
    vend = vsites[vend, :]
    vposes = torch.stack((vstart, vend), axis=2)

    horizontal = center.clone()
    horizontal[upper & lower] = False
    hsites = horizontal.nonzero()
    hindices = np.lexsort((hsites[:, 1].detach().cpu().numpy(), hsites[:, 0].detach().cpu().numpy()))
    hsites = hsites[hindices]
    hstart = torch.cat(
        (
            torch.tensor([True], device=hsites.device),
            hsites[:, 1][1:] != hsites[:, 1][:-1] + 1,
        )
    )
    hend = torch.cat(
        (
            hsites[:, 1][1:] != hsites[:, 1][:-1] + 1,
            torch.tensor([True], device=hsites.device),
        )
    )
    hstart = hsites[hstart, :]
    hend = hsites[hend, :]
    hposes = torch.stack((hstart, hend), axis=2)

    return vposes.float(), hposes.float()


def check(image, sample, target, direction):
    empty = sample[:0]

    def in_bounds(points):
        return (
            (points[:, 0] >= 0)
            & (points[:, 0] < image.shape[0])
            & (points[:, 1] >= 0)
            & (points[:, 1] < image.shape[1])
        )

    inner = empty
    outer = empty

    if direction == "v":
        if (target[sample[0, 0].long(), sample[0, 1].long() + 1] == 1) and (
            target[sample[0, 0].long(), sample[0, 1].long() - 1] == 0
        ):  # left ,x small
            inner = sample + torch.tensor([0, EPE_CONSTRAINT], dtype=sample.dtype, device=sample.device)
            outer = sample + torch.tensor([0, -EPE_CONSTRAINT], dtype=sample.dtype, device=sample.device)
            inner = inner[in_bounds(inner)]
            outer = outer[in_bounds(outer)]
            inner = sample[image[inner[:, 0].long(), inner[:, 1].long()] == 0, :]
            outer = sample[image[outer[:, 0].long(), outer[:, 1].long()] == 1, :]

        elif (target[sample[0, 0].long(), sample[0, 1].long() + 1] == 0) and (
            target[sample[0, 0].long(), sample[0, 1].long() - 1] == 1
        ):  # right, x large
            inner = sample + torch.tensor([0, -EPE_CONSTRAINT], dtype=sample.dtype, device=sample.device)
            outer = sample + torch.tensor([0, EPE_CONSTRAINT], dtype=sample.dtype, device=sample.device)
            inner = inner[in_bounds(inner)]
            outer = outer[in_bounds(outer)]
            inner = sample[image[inner[:, 0].long(), inner[:, 1].long()] == 0, :]
            outer = sample[image[outer[:, 0].long(), outer[:, 1].long()] == 1, :]

    if direction == "h":
        if (target[sample[0, 0].long() + 1, sample[0, 1].long()] == 1) and (
            target[sample[0, 0].long() - 1, sample[0, 1].long()] == 0
        ):  # up, y small
            inner = sample + torch.tensor([EPE_CONSTRAINT, 0], dtype=sample.dtype, device=sample.device)
            outer = sample + torch.tensor([-EPE_CONSTRAINT, 0], dtype=sample.dtype, device=sample.device)
            inner = inner[in_bounds(inner)]
            outer = outer[in_bounds(outer)]
            inner = sample[image[inner[:, 0].long(), inner[:, 1].long()] == 0, :]
            outer = sample[image[outer[:, 0].long(), outer[:, 1].long()] == 1, :]

        elif (target[sample[0, 0].long() + 1, sample[0, 1].long()] == 0) and (
            target[sample[0, 0].long() - 1, sample[0, 1].long()] == 1
        ):  # low, y large
            inner = sample + torch.tensor([-EPE_CONSTRAINT, 0], dtype=sample.dtype, device=sample.device)
            outer = sample + torch.tensor([EPE_CONSTRAINT, 0], dtype=sample.dtype, device=sample.device)
            inner = inner[in_bounds(inner)]
            outer = outer[in_bounds(outer)]
            inner = sample[image[inner[:, 0].long(), inner[:, 1].long()] == 0, :]
            outer = sample[image[outer[:, 0].long(), outer[:, 1].long()] == 1, :]

    return inner, outer


def epecheck(mask, target, vposes, hposes):
    """
    input: binary image tensor: (b, c, x, y); vertical points pair vposes: (N_v,4,2); horizontal points pair: (N_h, 4, 2), target image (b, c, x, y)
    output the total number of epe violations
    """
    inner = 0
    outer = 0
    epeMap = torch.zeros_like(target)
    vioMap = torch.zeros_like(target)

    for idx in range(vposes.shape[0]):
        center = (vposes[idx, :, 0] + vposes[idx, :, 1]) / 2
        center = center.int().float().unsqueeze(0)  # (1, 2)
        if (vposes[idx, 0, 1] - vposes[idx, 0, 0]) <= MIN_EPE_CHECK_LENGTH:
            sample = center
            epeMap[sample[:, 0].long(), sample[:, 1].long()] = 1
            v_in_site, v_out_site = check(mask, sample, target, "v")
        else:
            sampleY = torch.cat(
                (
                    torch.arange(
                        vposes[idx, 0, 0] + EPE_CHECK_START_INTERVEL,
                        center[0, 0] + 1,
                        step=EPE_CHECK_INTERVEL,
                    ),
                    torch.arange(
                        vposes[idx, 0, 1] - EPE_CHECK_START_INTERVEL,
                        center[0, 0],
                        step=-EPE_CHECK_INTERVEL,
                    ),
                )
            ).unique()
            sample = vposes[idx, :, 0].repeat(sampleY.shape[0], 1)
            sample[:, 0] = sampleY
            epeMap[sample[:, 0].long(), sample[:, 1].long()] = 1
            v_in_site, v_out_site = check(mask, sample, target, "v")
        inner = inner + v_in_site.shape[0]
        outer = outer + v_out_site.shape[0]
        vioMap[v_in_site[:, 0].long(), v_in_site[:, 1].long()] = 1
        vioMap[v_out_site[:, 0].long(), v_out_site[:, 1].long()] = 1

    for idx in range(hposes.shape[0]):
        center = (hposes[idx, :, 0] + hposes[idx, :, 1]) / 2
        center = center.int().float().unsqueeze(0)
        if (hposes[idx, 1, 1] - hposes[idx, 1, 0]) <= MIN_EPE_CHECK_LENGTH:
            sample = center
            epeMap[sample[:, 0].long(), sample[:, 1].long()] = 1
            v_in_site, v_out_site = check(mask, sample, target, "h")
        else:
            sampleX = torch.cat(
                (
                    torch.arange(
                        hposes[idx, 1, 0] + EPE_CHECK_START_INTERVEL,
                        center[0, 1] + 1,
                        step=EPE_CHECK_INTERVEL,
                    ),
                    torch.arange(
                        hposes[idx, 1, 1] - EPE_CHECK_START_INTERVEL,
                        center[0, 1],
                        step=-EPE_CHECK_INTERVEL,
                    ),
                )
            ).unique()
            sample = hposes[idx, :, 0].repeat(sampleX.shape[0], 1)
            sample[:, 1] = sampleX
            epeMap[sample[:, 0].long(), sample[:, 1].long()] = 1
            v_in_site, v_out_site = check(mask, sample, target, "h")
        inner = inner + v_in_site.shape[0]
        outer = outer + v_out_site.shape[0]
        vioMap[v_in_site[:, 0].long(), v_in_site[:, 1].long()] = 1
        vioMap[v_out_site[:, 0].long(), v_out_site[:, 1].long()] = 1
    return inner, outer, vioMap


class EPEChecker:
    def __init__(
        self,
        litho: LithoSim,
        device: torch.device,
        thresh=0.5,
    ):
        self._litho = litho
        self._thresh = thresh
        self._device = device

    def run(self, mask, target, scale=1):
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=REALTYPE, device=self._device)
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, dtype=REALTYPE, device=self._device)
        with torch.no_grad():
            mask[mask >= self._thresh] = 1.0
            mask[mask < self._thresh] = 0.0
            if scale != 1:
                mask = torch.nn.functional.interpolate(mask[None, None, :, :], scale_factor=scale, mode="nearest")[0, 0]
            printedNom, printedMax, printedMin = self._litho(mask)
            binaryNom = torch.zeros_like(printedNom)
            binaryNom[printedNom >= self._thresh] = 1
            vposes, hposes = boundaries(target)
            epeIn, epeOut, _ = epecheck(binaryNom, target, vposes, hposes)
        return epeIn, epeOut


class ShotCounter:
    def __init__(
        self,
        litho: LithoSim,
        device: torch.device,
        thresh=0.5,
    ):
        self._litho = litho
        self._thresh = thresh
        self._device = device

    def run(self, mask, target=None, scale=1, shape=(512, 512)):
        return len(mask_to_rectangles(mask, device=self._device, shape=shape))


def format_metrics(l2, pvb, epe, nshot=None):
    metrics = [f"L2 {l2:.0f}", f"PVBand {pvb:.0f}", f"EPE {epe:.0f}"]
    if nshot is not None and nshot >= 0:
        metrics.append(f"Shot: {nshot:.0f}")
    return "; ".join(metrics)


def evaluate(mask, target, litho, device, scale=1, shots=False, verbose=False):
    test = Basic(litho=litho, thresh=0.5, device=device)
    epeCheck = EPEChecker(litho=litho, thresh=0.5, device=device)
    shotCount = ShotCounter(litho=litho, thresh=0.5, device=device)

    l2, pvb = test.run(mask, target, scale=scale)
    epeIn, epeOut = epeCheck.run(mask, target, scale=scale)
    epe = epeIn + epeOut
    # begin = time.time()
    nshot = shotCount.run(mask, shape=(512, 512)) if shots else -1
    # print(f"Shot counting time: {time.time() - begin:.2f}")
    if verbose:
        print(format_metrics(l2, pvb, epe, nshot))

    return l2, pvb, epe, nshot


if __name__ == "__main__":
    import cv2

    targetfile = sys.argv[1]
    maskfile = sys.argv[2]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    litho = LithoSim("./config/lithosimple.txt")
    test = Basic(litho=litho, thresh=0.5, device=device)
    epeCheck = EPEChecker(litho=litho, thresh=0.5, device=device)
    shotCount = ShotCounter(litho=litho, thresh=0.5, device=device)

    mask = cv2.imread(maskfile)[:, :, 0] / 255
    mask = cv2.resize(mask, (2048, 2048))
    if targetfile[-4:] == ".png":
        target = cv2.imread(targetfile)[:, :, 0] / 255
        target = cv2.resize(target, (2048, 2048))
    else:
        ref = glp_seg.Design(targetfile, down=1)
        ref.center(2048, 2048, 0, 0)
        target = ref.mat(2048, 2048, 0, 0)

    l2, pvb = test.run(mask, target, scale=1)
    epeIn, epeOut = epeCheck.run(mask, target, scale=1)
    epe = epeIn + epeOut
    shot = shotCount.run(mask, shape=(512, 512))

    print(f"[{maskfile}]: {format_metrics(l2, pvb, epe, shot)}")
