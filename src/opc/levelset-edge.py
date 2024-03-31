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

from pathlib import Path

from matplotlib import pyplot as plt
from rich import print

import src.data.loaders.glp_seg as glp_seg
import src.data.loaders.segments as SegLoader
import src.opc.evaluation as evaluation
from src.litho.simple import LithoSim
from src.opc.utils import draw_edge_params, edge_params_merge2mask
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


def get_avg_grad(edge, grad_output):
    start_point = edge[:, 0].clone().detach().int()
    end_point = edge[:, 1].clone().detach().int()
    if start_point[1] == end_point[1]:  # horizontal
        if start_point[0] > end_point[0]:
            start_point, end_point = end_point, start_point
        selected_line = grad_output[start_point[1], start_point[0] : end_point[0] + 1]
    else:
        if start_point[1] > end_point[1]:  # vertical
            start_point, end_point = end_point, start_point
        selected_line = grad_output[start_point[1] : end_point[1] + 1, start_point[0]]
    average_value = selected_line.mean()
    return average_value


class StraightThroughEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, params):
        # In the forward pass, round the parameters to the nearest integers
        quantized = torch.round(params / 4) * 4
        # quantized = torch.round(params)
        return quantized

    @staticmethod
    def backward(ctx, grad_output):
        # In the backward pass, directly pass the gradients
        return grad_output


class _Binarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, edge_params, metadata):
        # ctx.seg_params = seg_params
        ctx.save_for_backward(edge_params)
        ctx.metadata_param = metadata
        binary_mask = edge_params_merge2mask(edge_params, metadata)
        return binary_mask

    @staticmethod
    def backward(ctx, grad_output):
        (edge_params,) = ctx.saved_tensors
        metadata = ctx.metadata_param
        polygon_ids = metadata["polygon_ids"]
        direction_vectors = metadata["direction_vectors"]
        velocities = metadata["velocities"]
        average_values = []
        for edge in edge_params:
            average_value = get_avg_grad(edge, grad_output)
            average_values.append(average_value)
        average_values = torch.stack(average_values)
        average_values = average_values.view(-1, 1, 1)
        grad_edge_params = average_values * velocities

        return grad_edge_params, None


class Binarize(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, edge_params, metadata):
        return _Binarize.apply(edge_params, metadata)


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

    def forward(self, edge_params, metadata):
        edge_params = StraightThroughEstimator.apply(edge_params)
        mask = self._binarize(edge_params, metadata)
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

    def solve(self, target, edge_params, metadata, case_id=1, curv=None, verbose=0):
        # Initialize
        # backup = params
        # params = params.clone().detach().requires_grad_(True)

        # Optimizer
        opt = optim.SGD([edge_params], lr=self._config["StepSize"])
        # opt = optim.Adam([edge_params], lr=self._config["StepSize"])

        # Optimization process
        lossMin, l2Min, pvbMin = 1e12, 1e12, 1e12
        bestParams = None
        bestMask = None
        all_masks = []
        all_mask_edges = []
        for idx in range(self._config["Iterations"]):
            # print(f"EdgeILT Iteration {idx}")
            # print(edge_params)
            mask, printedNom, printedMax, printedMin = self._levelset(edge_params, metadata)

            if idx % 5 == 0:
                mask_cpu = mask.clone().detach().cpu().numpy()
                all_masks.append({"mask": mask_cpu, "iteration": idx})
                shape = (self._config["TileSizeX"], self._config["TileSizeY"])
                mask_edge_cpu = draw_edge_params(edge_params, shape, show=False)
                all_mask_edges.append({"mask": mask_edge_cpu, "iteration": idx})

            #     plt.imshow(mask_cpu)
            #     plt.title(f"Iteration {idx}")
            #     plt.show()
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
                self._config["WeightL2"] * l2loss
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
                bestParams = edge_params.detach().clone()
                bestMask = mask.detach().clone()

            opt.zero_grad()
            loss.backward()
            opt.step()

        if self._config["VISUAL_DEBUG"]:
            fig, axs = plt.subplots(2, 4, figsize=(20, 12))
            if len(all_masks) >= 8:
                all_masks = all_masks[-8:]
            for i, ax in enumerate(axs.flat):
                if i < len(all_masks):
                    ax.imshow(all_masks[i]["mask"])
                    ax.set_title(f"Iteration {all_masks[i]['iteration']}")
            plt.tight_layout()
            save_dir = Path(f"./tmp/report/M1_test{case_id}")
            save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(f"{str(save_dir)}/EdgeILT_M1_test{case_id}_mask.png", dpi=300)

            for m in all_masks:
                plt.imsave(
                    f"{str(save_dir)}/EdgeILT_test{idx}_mask_{m['iteration']}.png",
                    m["mask"],
                    dpi=300,
                )
            # plt.show()

            fig, axs = plt.subplots(2, 4, figsize=(20, 12))
            if len(all_mask_edges) >= 8:
                all_mask_edges = all_mask_edges[-8:]
            for i, ax in enumerate(axs.flat):
                if i < len(all_mask_edges):
                    ax.imshow(all_mask_edges[i]["mask"])
                    ax.set_title(f"Iteration {all_masks[i]['iteration']}")
            plt.tight_layout()
            plt.savefig(f"{str(save_dir)}/EdgeILT_M1_test{case_id}_edge.png", dpi=300)

            for m in all_mask_edges:
                plt.imsave(
                    f"{str(save_dir)}/EdgeILT_test{idx}_edge_{m['iteration']}.png",
                    m["mask"],
                    dpi=300,
                )
            # plt.show()
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
        design = glp_seg.Design(f"./benchmark/ICCAD2013/M1_test{idx}.glp", down=SCALE)
        # design = glp_seg.Design(f"./benchmark/edge_bench/edge_test{idx}.glp", down=SCALE)
        # design.center(cfg["TileSizeX"], cfg["TileSizeY"], cfg["OffsetX"], cfg["OffsetY"])
        target, edge_params, metadata = SegLoader.SegmentsInitTorch().run(
            design,
            cfg["TileSizeX"],
            cfg["TileSizeY"],
            cfg["OffsetX"],
            cfg["OffsetY"],
            cfg["SEG_LENGTH"],
        )
        begin = time.time()
        # try:
        l2, pvb, bestParams, bestMask = solver.solve(
            target, edge_params, metadata, case_id=idx, curv=None, verbose=cfg["VERBOSE"]
        )
        # except Exception as e:
        # print(f"Error in testcase {idx}")
        # continue
        runtime = time.time() - begin

        ref = glp_seg.Design(f"./benchmark/ICCAD2013/M1_test{idx}.glp", down=1)
        # ref = glp_seg.Design(f"./benchmark/edge_bench/edge_test{idx}.glp", down=1)
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
