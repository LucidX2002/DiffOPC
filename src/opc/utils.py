import sys

sys.path.append(".")
import math
import multiprocessing as mp

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim

from src.data.datatype import COMPLEXTYPE, REALTYPE

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

from src.data.loaders import glp

# import pylitho.simple as lithosim
# import pylitho.exact as lithosim
from src.litho.simple import LithoSim


class Basic:
    def __init__(
        self,
        litho=LithoSim("./configs/litho/default.yaml"),
        thresh=0.5,
        device=DEVICE,
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
                mask = torch.nn.functional.interpolate(
                    mask[None, None, :, :], scale_factor=scale, mode="nearest"
                )[0, 0]
            printedNom, printedMax, printedMin = self._litho(mask)
            binaryNom = torch.zeros_like(printedNom)
            binaryMax = torch.zeros_like(printedMax)
            binaryMin = torch.zeros_like(printedMin)
            binaryNom[printedNom >= self._thresh] = 1
            binaryMax[printedMax >= self._thresh] = 1
            binaryMin[printedMin >= self._thresh] = 1
            l2loss = func.mse_loss(binaryNom, target, reduction="sum")
            pvband = torch.sum(binaryMax != binaryMin)
        return l2loss.item(), pvband.item()

    def sim(self, mask, target, scale=1):
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=REALTYPE, device=self._device)
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, dtype=REALTYPE, device=self._device)
        with torch.no_grad():
            mask[mask >= self._thresh] = 1.0
            mask[mask < self._thresh] = 0.0
            if scale != 1:
                mask = torch.nn.functional.interpolate(
                    mask[None, None, :, :], scale_factor=scale, mode="nearest"
                )[0, 0]
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
SEG_LENGTH = 40


def boundaries(target, dtype=REALTYPE, device=DEVICE):
    """Get the target boundaries lines.

    input: binary image tensor: (b, c, x, y), torch.tensor
    output: vertical edges: (N_v,2,2), horizontal edges: (N_h, 2, 2)
    (2, 2) is the start and end point of the line.
    """
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
    boundary[
        upper & lower & left & right & upperleft & upperright & lowerleft & lowerright
    ] = False

    padded = func.pad(boundary[None, None, :, :], pad=(1, 1, 1, 1))[0, 0]
    upper = padded[2:, 1:-1] == 1
    lower = padded[:-2, 1:-1] == 1
    left = padded[1:-1, :-2] == 1
    right = padded[1:-1, 2:] == 1
    center = padded[1:-1, 1:-1] == 1

    vertical = center.clone()
    vertical[left & right] = False
    vsites = vertical.nonzero()
    vindices = np.lexsort(
        (vsites[:, 0].detach().cpu().numpy(), vsites[:, 1].detach().cpu().numpy())
    )
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
    hindices = np.lexsort(
        (hsites[:, 1].detach().cpu().numpy(), hsites[:, 0].detach().cpu().numpy())
    )
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


def visualizeBoundaries(target, vposes, hposes):
    """Visualize vertical and horizontal boundaries on the given target image, and annotate each
    point with its coordinates, considering the coordinates are in the format [[y1, x1], [y2, x2]].
    The target image, vposes, and hposes are expected to be PyTorch tensors or NumPy arrays.

    Parameters:
    - target: Image tensor or array on which to draw the boundaries.
    - vposes: Tensor or array of shape (N, 2, 2) representing N vertical lines, with points in (y, x) format.
    - hposes: Tensor or array of shape (N, 2, 2) representing N horizontal lines, with points in (y, x) format.

    [[[y1, y2], [x1, x2]],
    [[y1, y2], [x1, x2]]]
    """
    # Check if inputs are tensors and convert them to numpy arrays
    if torch.is_tensor(target):
        target = target.cpu().numpy()
    if torch.is_tensor(vposes):
        vposes = vposes.cpu().numpy()
    if torch.is_tensor(hposes):
        hposes = hposes.cpu().numpy()

    # Convert the image to RGB if it is grayscale to ensure colored lines and text can be drawn
    if target.ndim == 2 or target.shape[2] == 1:
        target = cv2.cvtColor(target, cv2.COLOR_GRAY2BGR)

    def draw_lines_and_text(lines, color):
        for line in lines:
            start_point = (line[1][0].astype(int), line[0][0].astype(int))  # Swap to (x, y)
            end_point = (line[1][1].astype(int), line[0][1].astype(int))  # Swap to (x, y)
            # Draw line
            print(f"start_point: {start_point}, end_point: {end_point}")
            cv2.line(target, start_point, end_point, color, 2)
            # Annotate start point
            cv2.putText(
                target,
                f"{start_point}",
                start_point,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                color,
                1,
                cv2.LINE_AA,
            )
            # Annotate end point
            cv2.putText(
                target,
                f"{end_point}",
                end_point,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                color,
                1,
                cv2.LINE_AA,
            )

    # Draw vertical lines in blue and annotate
    draw_lines_and_text(vposes, (255, 0, 0))
    # Draw horizontal lines in red and annotate
    draw_lines_and_text(hposes, (0, 0, 255))
    # Display the result
    plt.imshow(cv2.cvtColor(target, cv2.COLOR_BGR2RGB))
    plt.axis("off")  # Hide axis
    plt.show()


# Example usage:
# Assuming target, vposes, hposes are your image and lines as either NumPy arrays or PyTorch tensors
# visualizeBoundaries(target, vposes, hposes)


def segment_lines(lines, seg_length):
    """Segment each line into smaller segments of fixed length from the midpoint towards the ends,
    ensuring that all points are integer coordinates."""
    if torch.is_tensor(lines):
        lines = lines.cpu().numpy()
    segmented_lines = []

    for line in lines:
        start_point = np.array([line[1][0], line[0][0]])  # [x, y] format
        end_point = np.array([line[1][1], line[0][1]])  # [x, y] format

        midpoint = (start_point + end_point) / 2.0
        direction = end_point - start_point
        line_length = np.linalg.norm(direction)
        direction_normalized = direction / line_length if line_length != 0 else direction

        segments_count = int(line_length / (2 * seg_length))

        for i in range(segments_count):
            for direction_multiplier in [1, -1]:
                segment_start = (
                    midpoint + direction_normalized * seg_length * i * direction_multiplier
                )
                segment_end = (
                    midpoint + direction_normalized * seg_length * (i + 1) * direction_multiplier
                )
                # Round to nearest integer and convert back to [[y1, y2], [x1, x2]] format
                segment = np.rint([segment_start, segment_end]).astype(int)
                segmented_lines.append(
                    [[segment[0][1], segment[1][1]], [segment[0][0], segment[1][0]]]
                )

    return segmented_lines


def visualize_segments(image, vsegments, hsegments):
    """Visualize vertical (vsegments) and horizontal (hsegments) segments on the image, ensuring
    the image is in RGB format to draw colored lines and text.

    input:
    [[y1, y2], [x1, x2]]
    """
    if torch.is_tensor(image):
        image = image.cpu().numpy()
    # Ensure the image is in RGB format
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    image_copy = image.copy()
    segment_number = 1

    def draw_segments(segments, color):
        nonlocal segment_number
        for segment in segments:
            # Swap to (x, y) format
            start_point = (segment[1][0], segment[0][0])
            end_point = (segment[1][1], segment[0][1])
            mid_point = (
                (start_point[0] + end_point[0]) // 2,
                (start_point[1] + end_point[1]) // 2,
            )

            cv2.line(image_copy, start_point, end_point, color, 1)
            cv2.circle(image_copy, start_point, 3, color, -1)
            cv2.circle(image_copy, end_point, 3, color, -1)
            # cv2.putText(image_copy, f"{segment[0]}", (start_point[0]+5, start_point[1]-5), cv2.FONT_HERSHEY_SIMPLEX,
            #             0.3, color, 1)
            # cv2.putText(image_copy, f"{segment[1]}", (end_point[0]+5, end_point[1]-5), cv2.FONT_HERSHEY_SIMPLEX,
            #             0.3, color, 1)
            cv2.putText(
                image_copy,
                f"{segment_number}",
                mid_point,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
            segment_number += 1

    draw_segments(vsegments, (255, 0, 0))  # Blue color for vertical segments
    draw_segments(hsegments, (0, 0, 255))  # Red color for horizontal segments

    plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


def check(image, sample, target, direction):
    if direction == "v":
        if (target[sample[0, 0].long(), sample[0, 1].long() + 1] == 1) and (
            target[sample[0, 0].long(), sample[0, 1].long() - 1] == 0
        ):  # left ,x small
            inner = sample + torch.tensor(
                [0, EPE_CONSTRAINT], dtype=sample.dtype, device=sample.device
            )
            outer = sample + torch.tensor(
                [0, -EPE_CONSTRAINT], dtype=sample.dtype, device=sample.device
            )
            inner = sample[image[inner[:, 0].long(), inner[:, 1].long()] == 0, :]
            outer = sample[image[outer[:, 0].long(), outer[:, 1].long()] == 1, :]

        elif (target[sample[0, 0].long(), sample[0, 1].long() + 1] == 0) and (
            target[sample[0, 0].long(), sample[0, 1].long() - 1] == 1
        ):  # right, x large
            inner = sample + torch.tensor(
                [0, -EPE_CONSTRAINT], dtype=sample.dtype, device=sample.device
            )
            outer = sample + torch.tensor(
                [0, EPE_CONSTRAINT], dtype=sample.dtype, device=sample.device
            )
            inner = sample[image[inner[:, 0].long(), inner[:, 1].long()] == 0, :]
            outer = sample[image[outer[:, 0].long(), outer[:, 1].long()] == 1, :]

    if direction == "h":
        if (target[sample[0, 0].long() + 1, sample[0, 1].long()] == 1) and (
            target[sample[0, 0].long() - 1, sample[0, 1].long()] == 0
        ):  # up, y small
            inner = sample + torch.tensor(
                [EPE_CONSTRAINT, 0], dtype=sample.dtype, device=sample.device
            )
            outer = sample + torch.tensor(
                [-EPE_CONSTRAINT, 0], dtype=sample.dtype, device=sample.device
            )
            inner = sample[image[inner[:, 0].long(), inner[:, 1].long()] == 0, :]
            outer = sample[image[outer[:, 0].long(), outer[:, 1].long()] == 1, :]

        elif (target[sample[0, 0].long() + 1, sample[0, 1].long()] == 0) and (
            target[sample[0, 0].long() - 1, sample[0, 1].long()] == 1
        ):  # low, y large
            inner = sample + torch.tensor(
                [-EPE_CONSTRAINT, 0], dtype=sample.dtype, device=sample.device
            )
            outer = sample + torch.tensor(
                [EPE_CONSTRAINT, 0], dtype=sample.dtype, device=sample.device
            )
            inner = sample[image[inner[:, 0].long(), inner[:, 1].long()] == 0, :]
            outer = sample[image[outer[:, 0].long(), outer[:, 1].long()] == 1, :]

    return inner, outer


def epecheck(mask, target, vposes, hposes):
    """
    input: binary image tensor: (b, c, x, y);
    vertical points pair vposes: (N_v,4,2);
    horizontal points pair: (N_h, 4, 2),
    target image (b, c, x, y)
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
        litho=LithoSim("./configs/litho/default.yaml"),
        thresh=0.5,
        device=DEVICE,
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
                mask = torch.nn.functional.interpolate(
                    mask[None, None, :, :], scale_factor=scale, mode="nearest"
                )[0, 0]
            printedNom, printedMax, printedMin = self._litho(mask)
            binaryNom = torch.zeros_like(printedNom)
            binaryNom[printedNom >= self._thresh] = 1
            vposes, hposes = boundaries(target)
            epeIn, epeOut, _ = epecheck(binaryNom, target, vposes, hposes)
        return epeIn, epeOut


from adabox import proc, tools


class ShotCounter:
    def __init__(
        self,
        litho=LithoSim("./configs/litho/default.yaml"),
        thresh=0.5,
        device=DEVICE,
    ):
        self._litho = litho
        self._thresh = thresh
        self._device = device

    def run(self, mask, target=None, scale=1, shape=(512, 512)):
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=REALTYPE, device=self._device)
        image = torch.nn.functional.interpolate(
            mask[None, None, :, :], size=shape, mode="nearest"
        )[0, 0]
        image = image.detach().cpu().numpy().astype(np.uint8)
        comps, labels, stats, centroids = cv2.connectedComponentsWithStats(image)
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
                rectangles.append(
                    tools.Rectangle(x_data.min(), x_data.max(), y_data.min(), y_data.max())
                )
                continue
            (rects, sep) = proc.decompose(pixels, 4)
            rectangles.extend(rects)
        return len(rectangles)


def evaluate(mask, target, litho, scale=1, shots=False, verbose=False):
    test = Basic(litho, 0.5)
    epeCheck = EPEChecker(litho, 0.5)
    shotCount = ShotCounter(litho, 0.5)

    l2, pvb = test.run(mask, target, scale=scale)
    epeIn, epeOut = epeCheck.run(mask, target, scale=scale)
    epe = epeIn + epeOut
    nshot = shotCount.run(mask, shape=(512, 512)) if shots else -1
    if verbose:
        print(f"[{maskfile}]: L2 {l2:.0f}; PVBand {pvb:.0f}; EPE {epe:.0f}; Shot: {nshot:.0f}")

    return l2, pvb, epe, nshot


if __name__ == "__main__":
    # targetfile = "./benchmark/edge_bench/edge_test1.glp"
    maskfile = "./benchmark/edge_bench/edge_test1.glp"
    # maskfile = "./benchmark/ICCAD2013/M1_test1.glp"
    # litho = LithoSim("./configs/litho/default.yaml")
    # test = Basic(litho, 0.5)
    # epeCheck = EPEChecker(litho, 0.5)
    # shotCount = ShotCounter(litho, 0.5)

    # ref = glp.Design(targetfile, down=1)
    # ref.center(2048, 2048, 0, 0)
    # target = ref.mat(2048, 2048, 0, 0)

    mref = glp.Design(maskfile, down=1)
    mask_shape = (512, 512)
    # mask_shape = (2048, 2048)
    mref.center(mask_shape[0], mask_shape[1], 0, 0)
    mask = mref.mat(mask_shape[0], mask_shape[1], 0, 0)
    # mask = mref.mat(2048, 2048, 0, 0)
    mask_tensor = torch.tensor(mask, dtype=REALTYPE, device=DEVICE)
    vposes, hposes = boundaries(mask_tensor)
    # visualizeBoundaries(mask_tensor, vposes, hposes)
    vsegs = segment_lines(vposes, SEG_LENGTH)
    hsegs = segment_lines(hposes, SEG_LENGTH)
    visualize_segments(mask_tensor, vsegs, hsegs)
    # l2, pvb = test.run(mask, target, scale=1)
    # epeIn, epeOut = epeCheck.run(mask, target, scale=1)
    # epe = epeIn + epeOut
    # shot = shotCount.run(mask, shape=(512, 512))

    # print(f"[{maskfile}]: L2 {l2:.0f}; PVBand {pvb:.0f}; EPE {epe:.0f}; Shot: {shot:.0f}")
    # print(f"[{maskfile}]: L2 {l2:.0f}; PVBand {pvb:.0f}; EPE {epe:.0f};")

    # import matplotlib.pyplot as plt

    # plt.subplot(1, 2, 1)
    # plt.imshow(mask)
    # plt.subplot(1, 2, 2)
    # plt.imshow(printed[0].detach().cpu().numpy())
    # plt.show()
