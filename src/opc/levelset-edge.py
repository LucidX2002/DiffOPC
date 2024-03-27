import math
import multiprocessing as mp
import os
import sys

sys.path.append(".")
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim

from src.data.datatype import COMPLEXTYPE, REALTYPE

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

from rich import print

import src.data.loaders.glp as glp
import src.data.loaders.segments as SegLoader
import src.opc.evaluation as evaluation
from src.litho.simple import LithoSim
from src.opc.utils import (
    SEG_LENGTH,
    boundaries,
    segment_lines_with_labels,
    validate_segments,
    visualize_segments_with_labels,
)
from src.utils.utils import yaml2Cfg

# import pylitho.exact as lithosim


class LevelSetCfg:
    def __init__(self, config):
        # Read the config from file or a given dict
        if isinstance(config, dict):
            self._config = config
        elif isinstance(config, str):
            self._config = yaml2Cfg(config)
        required = [
            "Iterations",
            "TargetDensity",
            "SigmoidSteepness",
            "WeightEPE",
            "WeightPVBL2",
            "WeightPVBand",
            "StepSize",
            "TileSizeX",
            "TileSizeY",
            "OffsetX",
            "OffsetY",
            "ILTSizeX",
            "ILTSizeY",
        ]
        for key in required:
            assert key in self._config, f"[SimpleILT]: Cannot find the config {key}."
        intfields = [
            "Iterations",
            "TileSizeX",
            "TileSizeY",
            "OffsetX",
            "OffsetY",
            "ILTSizeX",
            "ILTSizeY",
        ]
        for key in intfields:
            self._config[key] = int(self._config[key])
        floatfields = [
            "TargetDensity",
            "SigmoidSteepness",
            "WeightEPE",
            "WeightPVBL2",
            "WeightPVBand",
            "StepSize",
        ]
        for key in floatfields:
            self._config[key] = float(self._config[key])

    def __getitem__(self, key):
        return self._config[key]


def gradImage(image):
    GRAD_STEPSIZE = 1.0
    image = image.view([-1, 1, image.shape[-2], image.shape[-1]])
    padded = func.pad(image, (1, 1, 1, 1), mode="replicate")[:, 0].detach()
    gradX = (padded[:, 2:, 1:-1] - padded[:, :-2, 1:-1]) / (2.0 * GRAD_STEPSIZE)
    gradY = (padded[:, 1:-1, 2:] - padded[:, 1:-1, :-2]) / (2.0 * GRAD_STEPSIZE)
    return gradX.view(image.shape), gradY.view(image.shape)


class _Binarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, levelset, seg_params):
        params = [seg["segment"] for seg in seg_params]
        ctx.save_for_backward(params)
        ctx.seg_params = seg_params
        mask = torch.zeros_like(levelset)
        mask[levelset < 0] = 1.0
        return mask

    @staticmethod
    def backward(ctx, grad_output):
        (levelset,) = ctx.saved_tensors
        gradX, gradY = gradImage(levelset)
        l2norm = torch.sqrt(gradX**2 + gradY**2)
        return -l2norm * grad_output


class Binarize(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, seg_params):
        return _Binarize.apply(seg_params)


# class _Rectangular(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, mask, config):
#         ctx.save_for_backward(mask)
#         mask_rectangular = torch.zeros_like(mask)
#         return mask

#     @staticmethod
#     def backward(ctx, grad_output):
#         mask, = ctx.saved_tensors
#         grad_mask = mask * grad_output
#         return grad_mask, None

# class Rectangular(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self._config = config

#     def forward(self, mask):
#         return _Rectangular.apply(mask, self._config)


class LevelSet(nn.Module):
    def __init__(self, lithosim):
        super().__init__()
        self._binarize = Binarize()
        self._lithosim = lithosim
        # self.add_module("binary", self._binarize)
        # self.add_module("lithosim", self._lithosim)

    def forward(self, seg_params):
        mask = self._binarize(seg_params)
        printedNom, printedMax, printedMin = self._lithosim(mask)
        return mask, printedNom, printedMax, printedMin


class EdgeILT:
    def __init__(
        self,
        config=LevelSetCfg("./configs/opc/default.yaml"),
        lithosim=LithoSim("./configs/litho/default.yaml"),
        device=DEVICE,
        multigpu=False,
    ):
        super().__init__()
        self._config = config
        self._device = device
        # LevelSet
        self._levelset = LevelSet(lithosim).to(DEVICE)
        if multigpu:
            self._levelset = nn.DataParallel(self._levelset)
        # Filter
        self._filter = torch.zeros(
            [self._config["TileSizeX"], self._config["TileSizeY"]],
            dtype=REALTYPE,
            device=self._device,
        )
        self._filter[
            self._config["OffsetX"] : self._config["OffsetX"] + self._config["ILTSizeX"],
            self._config["OffsetY"] : self._config["OffsetY"] + self._config["ILTSizeY"],
        ] = 1

    def solve(self, target, seg_params, curv=None, verbose=0):
        # Initialize
        # backup = params
        # params = params.clone().detach().requires_grad_(True)

        # Optimizer
        # opt = optim.SGD([params], lr=self._config["StepSize"])
        params = [seg["segment"] for seg in seg_params]
        opt = optim.Adam(params, lr=self._config["StepSize"])

        # Optimization process
        lossMin, l2Min, pvbMin = 1e12, 1e12, 1e12
        bestParams = None
        bestMask = None
        for idx in range(self._config["Iterations"]):
            mask, printedNom, printedMax, printedMin = self._levelset(params * self._filter)
            l2loss = func.mse_loss(printedNom, target, reduction="sum")
            pvbl2 = func.mse_loss(printedMax, target, reduction="sum") + func.mse_loss(
                printedMin, target, reduction="sum"
            )
            pvbloss = func.mse_loss(printedMax, printedMin, reduction="sum")
            pvband = torch.sum(
                (printedMax >= self._config["TargetDensity"])
                != (printedMin >= self._config["TargetDensity"])
            )
            loss = (
                l2loss
                + self._config["WeightPVBL2"] * pvbl2
                + self._config["WeightPVBand"] * pvbloss
            )
            if curv is not None:
                kernelCurv = torch.tensor(
                    [
                        [-1.0 / 16, 5.0 / 16, -1.0 / 16],
                        [5.0 / 16, -1.0, 5.0 / 16],
                        [-1.0 / 16, 5.0 / 16, -1.0 / 16],
                    ],
                    dtype=REALTYPE,
                    device=DEVICE,
                )
                curvature = func.conv2d(mask[None, None, :, :], kernelCurv[None, None, :, :])[0, 0]
                losscurv = func.mse_loss(curvature, torch.zeros_like(curvature), reduction="sum")
                loss += curv * losscurv
            if verbose == 1:
                print(f"[Iteration {idx}]: L2 = {l2loss.item():.0f}; PVBand: {pvband.item():.0f}")

            if bestParams is None or bestMask is None or loss.item() < lossMin:
                lossMin, l2Min, pvbMin = loss.item(), l2loss.item(), pvband.item()
                bestParams = params.detach().clone()
                bestMask = mask.detach().clone()

            opt.zero_grad()
            loss.backward()
            opt.step()

        return l2Min, pvbMin, bestParams, bestMask


def serial():
    from omegaconf import OmegaConf

    SCALE = 1
    l2s = []
    pvbs = []
    epes = []
    shots = []
    runtimes = []
    targetsAll = []
    paramsAll = []
    levelSetCfg = OmegaConf.load("./configs/opc/default.yaml")
    levelSetCfg = OmegaConf.to_container(levelSetCfg, resolve=True, throw_on_missing=True)
    cfg = LevelSetCfg(levelSetCfg)

    lithoCfg = OmegaConf.load("./configs/litho/default.yaml")
    lithoCfg = OmegaConf.to_container(lithoCfg, resolve=True, throw_on_missing=True)
    litho = LithoSim(lithoCfg)
    solver = EdgeILT(cfg, litho)
    for idx in range(1, 2):
        # design = glp.Design(f"./benchmark/ICCAD2013/M1_test{idx}.glp", down=SCALE)
        design = glp.Design(f"./benchmark/edge_bench/edge_test{idx}.glp", down=SCALE)
        design.center(cfg["TileSizeX"], cfg["TileSizeY"], cfg["OffsetX"], cfg["OffsetY"])
        target, seg_params = SegLoader.SegmentsInitTorch().run(
            design, cfg["TileSizeX"], cfg["TileSizeY"], cfg["OffsetX"], cfg["OffsetY"]
        )
        print(seg_params)
        begin = time.time()
        l2, pvb, bestParams, bestMask = solver.solve(target, seg_params, curv=None, verbose=1)
        runtime = time.time() - begin

        # ref = glp.Design(f"./benchmark/ICCAD2013/M1_test{idx}.glp", down=1)
        ref = glp.Design(f"./benchmark/edge_bench/edge_test{idx}.glp", down=1)
        ref.center(
            cfg["TileSizeX"] * SCALE,
            cfg["TileSizeY"] * SCALE,
            cfg["OffsetX"] * SCALE,
            cfg["OffsetY"] * SCALE,
        )
        target, params = SegLoader.LevelSetInitTorch().run(
            ref,
            cfg["TileSizeX"] * SCALE,
            cfg["TileSizeY"] * SCALE,
            cfg["OffsetX"] * SCALE,
            cfg["OffsetY"] * SCALE,
        )
        l2, pvb, epe, shot = evaluation.evaluate(bestMask, target, litho, scale=SCALE, shots=True)
        cv2.imwrite(f"./tmp/LevelSet_test{idx}.png", (bestMask * 255).detach().cpu().numpy())

        print(
            f"[Testcase {idx}]: L2 {l2:.0f}; PVBand {pvb:.0f}; EPE {epe:.0f}; Shot: {shot:.0f}; SolveTime: {runtime:.2f}s"
        )

        l2s.append(l2)
        pvbs.append(pvb)
        epes.append(epe)
        shots.append(shot)
        runtimes.append(runtime)

    print(
        f"[Result]: L2 {np.mean(l2s):.0f}; PVBand {np.mean(pvbs):.0f}; EPE {np.mean(epes):.1f}; Shot {np.mean(shots):.1f}; SolveTime: {np.mean(runtimes):.2f}s"
    )


if __name__ == "__main__":
    serial()
    # parallel()
