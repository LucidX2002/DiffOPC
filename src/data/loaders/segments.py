import sys

sys.path.append(".")
import math
import multiprocessing as mp

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim

from src.data.datatype import COMPLEXTYPE, REALTYPE

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# import pycommon.utils as common
import src.data.loaders.glp_seg as glp_seg
from src.litho.simple import LithoSim
from src.opc.utils import (
    SEG_LENGTH,
    right_perpendicular_unit_vector,
    segment_polygon_edges_with_labels,
)


class Initializer:
    def __init__(self):
        pass

    def run(self, design, sizeX, sizeY, offsetX, offsetY, dtype=REALTYPE, device=DEVICE):
        pass


def _distMatPolygon(polygon, canvas, offsets):
    if len(canvas) == 4:
        canvas = [[canvas[0], canvas[1]], [canvas[2], canvas[3]]]
    minX, minY, maxX, maxY = canvas[0][0], canvas[0][1], canvas[1][0], canvas[1][1]
    sizeX, sizeY = maxX - minX, maxY - minY

    dist = np.ones([sizeX, sizeY]) * (sizeX * sizeY)
    xs = np.arange(minX, maxX, 1, dtype=np.int32).reshape([sizeX, 1])
    ys = np.arange(minY, maxY, 1, dtype=np.int32).reshape([1, sizeY])
    xs = np.tile(xs, [1, sizeY])
    ys = np.tile(ys, [sizeX, 1])

    frPt = polygon[-1]
    for toPt in polygon:
        frX, frY = frPt
        toX, toY = toPt
        if frX > toX:
            frX, toX = toX, frX
        if frY > toY:
            frY, toY = toY, frY
        frX += offsets[0]
        toX += offsets[0]
        frY += offsets[1]
        toY += offsets[1]

        dist1 = np.sqrt((frX - xs) ** 2 + (frY - ys) ** 2)
        dist2 = np.sqrt((toX - xs) ** 2 + (toY - ys) ** 2)

        dist = np.minimum(dist, np.minimum(dist1, dist2))

        if frX == toX:
            mask = (frY <= ys) * (ys <= toY)
            new = np.minimum(dist, np.abs(frX - xs))
            dist[mask] = new[mask]
        elif frY == toY:
            mask = (frX <= xs) * (xs <= toX)
            new = np.minimum(dist, np.abs(frY - ys))
            dist[mask] = new[mask]

        frPt = toPt
    return dist.T


def _distMatLegacy(design, canvas=[[0, 0], [2048, 2048]], offsets=[512, 512]):
    if len(canvas) == 4:
        canvas = [[canvas[0], canvas[1]], [canvas[2], canvas[3]]]
    minX, minY, maxX, maxY = canvas[0][0], canvas[0][1], canvas[1][0], canvas[1][1]

    mask = design.mat(sizeX=maxX - minX, sizeY=maxY - minY, offsetX=offsets[0], offsetY=offsets[1])
    dist = np.ones([maxX - minX, maxY - minY]) * ((maxX - minX) * (maxY - minY))
    for polygon in design.polygons:
        tmp = _distMatPolygon(polygon, canvas, offsets)
        dist = np.minimum(dist, tmp)
    dist[mask > 0] *= -1
    return dist


def _distMatPolygonTorch(polygon, canvas, offsets):
    if len(canvas) == 4:
        canvas = [[canvas[0], canvas[1]], [canvas[2], canvas[3]]]
    minX, minY, maxX, maxY = canvas[0][0], canvas[0][1], canvas[1][0], canvas[1][1]
    sizeX, sizeY = maxX - minX, maxY - minY

    dist = torch.ones([sizeX, sizeY], dtype=REALTYPE, device=DEVICE) * (sizeX * sizeY)
    xs = np.arange(minX, maxX, 1, dtype=np.int32).reshape([sizeX, 1])
    ys = np.arange(minY, maxY, 1, dtype=np.int32).reshape([1, sizeY])
    xs = torch.tensor(np.tile(xs, [1, sizeY]), dtype=REALTYPE, device=DEVICE)
    ys = torch.tensor(np.tile(ys, [sizeX, 1]), dtype=REALTYPE, device=DEVICE)

    frPt = polygon[-1]
    for toPt in polygon:
        frX, frY = frPt
        toX, toY = toPt
        if frX > toX:
            frX, toX = toX, frX
        if frY > toY:
            frY, toY = toY, frY
        frX += offsets[0]
        toX += offsets[0]
        frY += offsets[1]
        toY += offsets[1]

        dist1 = torch.sqrt((frX - xs) ** 2 + (frY - ys) ** 2)
        dist2 = torch.sqrt((toX - xs) ** 2 + (toY - ys) ** 2)

        dist = torch.minimum(dist, torch.minimum(dist1, dist2))

        if frX == toX:
            mask = (frY <= ys) * (ys <= toY)
            new = torch.minimum(dist, torch.abs(frX - xs))
            dist[mask] = new[mask]
        elif frY == toY:
            mask = (frX <= xs) * (xs <= toX)
            new = torch.minimum(dist, torch.abs(frY - ys))
            dist[mask] = new[mask]

        frPt = toPt
    return dist.T


def _distMatTorch(design, canvas=[[0, 0], [2048, 2048]], offsets=[512, 512], mask=None):
    if len(canvas) == 4:
        canvas = [[canvas[0], canvas[1]], [canvas[2], canvas[3]]]
    minX, minY, maxX, maxY = canvas[0][0], canvas[0][1], canvas[1][0], canvas[1][1]

    if mask is None:
        mask = design.mat(
            sizeX=maxX - minX, sizeY=maxY - minY, offsetX=offsets[0], offsetY=offsets[1]
        )
    dist = torch.ones([maxX - minX, maxY - minY], dtype=REALTYPE, device=DEVICE) * (
        (maxX - minX) * (maxY - minY)
    )
    for polygon in design.polygons:
        tmp = _distMatPolygonTorch(polygon, canvas, offsets)
        dist = torch.minimum(dist, tmp)
    dist[mask > 0] *= -1
    return dist


def _distMat(design, canvas=[[0, 0], [2048, 2048]], offsets=[512, 512]):
    if len(canvas) == 4:
        canvas = [[canvas[0], canvas[1]], [canvas[2], canvas[3]]]
    minX, minY, maxX, maxY = canvas[0][0], canvas[0][1], canvas[1][0], canvas[1][1]

    pool = mp.Pool(processes=mp.cpu_count() // 2)
    procs = []
    for polygon in design.polygons:
        proc = pool.apply_async(_distMatPolygon, (polygon, canvas, offsets))
        procs.append(proc)
    pool.close()
    pool.join()

    dist = np.ones([maxX - minX, maxY - minY]) * ((maxX - minX) * (maxY - minY))
    for proc in procs:
        tmp = proc.get()
        dist = np.minimum(dist, tmp)
    mask = design.mat(sizeX=maxX - minX, sizeY=maxY - minY, offsetX=offsets[0], offsetY=offsets[1])
    dist[mask > 0] *= -1

    return dist


class LevelSetInit(Initializer):
    def __init__(self):
        super().__init__()

    def run(self, design, sizeX, sizeY, offsetX, offsetY, dtype=REALTYPE, device=DEVICE):
        target = torch.tensor(
            design.mat(sizeX, sizeY, offsetX, offsetY), dtype=dtype, device=device
        )
        params = torch.tensor(
            _distMat(design, canvas=[[0, 0], [sizeX, sizeY]], offsets=[offsetX, offsetY]),
            dtype=REALTYPE,
            device=DEVICE,
            requires_grad=True,
        )
        return target, params


class LevelSetInitTorch(Initializer):
    def __init__(self):
        super().__init__()

    def run(self, design, sizeX, sizeY, offsetX, offsetY, dtype=REALTYPE, device=DEVICE):
        target = torch.tensor(
            design.mat(sizeX, sizeY, offsetX, offsetY), dtype=dtype, device=device
        )
        params = (
            _distMatTorch(design, canvas=[[0, 0], [sizeX, sizeY]], offsets=[offsetX, offsetY])
            .detach()
            .clone()
            .requires_grad_(True)
        )
        return target, params


class SegmentsInitTorch(Initializer):
    def __init__(self):
        super().__init__()

    def run(self, design, sizeX, sizeY, offsetX, offsetY, dtype=REALTYPE, device=DEVICE):
        design.center(sizeX, sizeY, offsetX, offsetY)
        target = torch.tensor(
            design.mat(sizeX, sizeY, offsetX, offsetY), dtype=dtype, device=device
        )
        target_edges = design.polygon_edges
        seg_params = segment_polygon_edges_with_labels(target_edges, SEG_LENGTH)
        edge_params = []
        polygon_ids = []
        direction_vectors = []
        velocities = []
        for idx, poly in enumerate(seg_params):
            polygon_ids.append(torch.full((len(poly),), idx, dtype=torch.int32))
            for seg in poly:
                edge_params.append(seg["segment"].detach().clone())
                direction_vectors.append(seg["direction"].detach().clone())
                velocity = (
                    right_perpendicular_unit_vector(seg["direction"])
                    if seg["type"] in ["H", "V"]
                    else torch.tensor([0, 0])
                )
                velocity = torch.stack([velocity, velocity], dim=0)
                velocity = torch.transpose(velocity, 0, 1)
                velocities.append(velocity.round().detach().clone())
        edge_params = torch.stack(edge_params, dim=0).to(device).requires_grad_(True)
        polygon_ids = torch.cat(polygon_ids, dim=0).to(device)
        direction_vectors = torch.stack(direction_vectors, dim=0).to(device)
        velocities = torch.stack(velocities, dim=0).to(device)
        shape = (sizeX, sizeY)
        assert polygon_ids.shape[0] == edge_params.shape[0]
        assert polygon_ids.shape[0] == direction_vectors.shape[0]
        assert edge_params.shape == velocities.shape
        metadata = {
            "img_shape": shape,
            "polygon_ids": polygon_ids,
            "direction_vectors": direction_vectors,
            "velocities": velocities,
        }
        # print(edge_params[1])
        # print(direction_vectors[1])
        # print(velocities[1])
        return target, edge_params, metadata


if __name__ == "__main__":
    import levelset as ilt

    maskfile = "./benchmark/edge_bench/edge_test1.glp"
    mask_shape = (512, 512)
    mref = glp_seg.Design(maskfile, down=1)
