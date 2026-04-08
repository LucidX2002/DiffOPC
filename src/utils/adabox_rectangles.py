import cv2
import numpy as np
import torch

from adabox import proc, tools

from src.data.datatype import REALTYPE
from src.utils.adabox_compat import patch_adabox_mode_compat


patch_adabox_mode_compat(proc, tools)


def binary_array_to_rectangles(binary_array):
    comps, labels, _stats, _centroids = cv2.connectedComponentsWithStats(binary_array)
    rectangles = []
    for label in range(1, comps):
        pixels = []
        for idx in range(labels.shape[0]):
            for jdx in range(labels.shape[1]):
                if labels[idx, jdx] == label:
                    pixels.append([idx, jdx, 0])
        pixels = np.array(pixels)
        x_data = np.unique(np.sort(pixels[:, 0]))
        y_data = np.unique(np.sort(pixels[:, 1]))
        if x_data.shape[0] == 1 or y_data.shape[0] == 1:
            rectangles.append(tools.Rectangle(x_data.min(), x_data.max(), y_data.min(), y_data.max()))
            continue
        rects, _sep = proc.decompose(pixels, 4)
        rectangles.extend(rects)
    return rectangles


def mask_to_rectangles(mask, device, shape=(512, 512)):
    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask, dtype=REALTYPE, device=device)
    image = torch.nn.functional.interpolate(mask[None, None, :, :], size=shape, mode="nearest")[0, 0]
    binary_array = image.detach().cpu().numpy().astype(np.uint8)
    return binary_array_to_rectangles(binary_array)
