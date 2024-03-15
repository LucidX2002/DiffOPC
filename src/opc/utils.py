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
            start_point = (
                line[1][0].astype(int),
                line[0][0].astype(int),
            )  # Swap to (x, y)
            end_point = (
                line[1][1].astype(int),
                line[0][1].astype(int),
            )  # Swap to (x, y)
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


def segment_lines(lines, seg_length):
    """Segments each line in the input tensor into smaller segments of a fixed length. For lines
    shorter than 2 * seg_length, it divides the line into two segments. Ensures that the
    coordinates of the segments are integers.

    Args:
        lines (torch.Tensor): Tensor of shape [N, 2, 2] representing the lines.
                                Each line is stored as [[y1, y2], [x1, x2]].
        seg_length (float): The fixed length for the segments.

    Returns:
        list: A list of segmented lines. Each element in the list is a tensor
                representing the segmented parts of a line, with integer coordinates.
    """

    def split_edge(edge):
        midpoint = torch.mean(edge, dim=1)
        vector = edge[:, 1] - edge[:, 0]
        length = torch.norm(vector)
        direction = vector / length

        segments = []
        if length < 2 * seg_length:
            # Round coordinates to nearest integer
            start_point = torch.round(edge[:, 0])
            end_point = torch.round(midpoint)
            segments.append(torch.stack([start_point, end_point], dim=1))

            start_point = torch.round(midpoint)
            end_point = torch.round(edge[:, 1])
            segments.append(torch.stack([start_point, end_point], dim=1))
        else:
            half_segments = int(length / seg_length / 2)
            for i in range(-half_segments, half_segments + 1):
                start_point = midpoint + direction * (i * seg_length - seg_length / 2)
                end_point = midpoint + direction * (i * seg_length + seg_length / 2)

                # Round coordinates to nearest integer
                start_point = torch.round(start_point)
                end_point = torch.round(end_point)

                segments.append(torch.stack([start_point, end_point], dim=1))
                # Adjust the last segment to ensure the entire edge is covered
                if i == half_segments:
                    segments[-1][:, 1] = torch.round(edge[:, 1])
                elif i == -half_segments:
                    segments[0][:, 0] = torch.round(edge[:, 0])

        return segments

    split_lines = [split_edge(lines[i]) for i in range(lines.shape[0])]

    return split_lines


def segment_lines_with_labels(vposes, hposes, seg_length):
    """Segments vertical and horizontal lines into smaller segments of a fixed length, labels
    vertical (V), horizontal (H), corner vertical (CV), and corner horizontal (CH) segments, and
    assigns a unique ID to each segment.

    Args:
        vposes (torch.Tensor): Tensor of shape [N, 2, 2] representing the vertical lines.
        hposes (torch.Tensor): Tensor of shape [M, 2, 2] representing the horizontal lines.
        seg_length (float): The fixed length for the segments.

    Returns:
        list: A list of segmented lines with labels and IDs. Each element in the list is a dictionary
              with 'segment' (tensor representing the segmented part of a line with integer coordinates),
              'type' (string label of the segment type), and 'id' (unique identifier).
    """

    def split_edge(edge, seg_type_label, segment_id):
        seg_length = SEG_LENGTH
        midpoint = torch.mean(edge, dim=1)
        vector = edge[:, 1] - edge[:, 0]
        length = torch.norm(vector)
        direction = vector / length

        segments = []

        if length < seg_length:
            # Treat both ends as corners if the segment is short
            seg_type_label = "C" + seg_type_label
            start_point = torch.round(edge[:, 0])
            end_point = torch.round(midpoint)
            segments.append(
                {
                    "segment": torch.stack([start_point, end_point], dim=1),
                    "type": seg_type_label,
                    "id": segment_id,
                }
            )
            segment_id += 1

            start_point = torch.round(midpoint)
            end_point = torch.round(edge[:, 1])
            segments.append(
                {
                    "segment": torch.stack([start_point, end_point], dim=1),
                    "type": seg_type_label,
                    "id": segment_id,
                }
            )
            segment_id += 1
        else:
            # Calculate segments from the midpoint to each end.
            steps_to_edge = length / 2 / seg_length
            full_steps = int(steps_to_edge)
            if full_steps == 0:
                full_steps = 1

            for i in range(-full_steps, full_steps + 1):
                if i == -full_steps:
                    start_point = edge[:, 0]
                else:
                    start_point = midpoint + direction * (
                        i * seg_length - min(seg_length / 2, length / 2)
                    )

                if i == full_steps:
                    end_point = edge[:, 1]
                else:
                    end_point = midpoint + direction * (
                        i * seg_length + min(seg_length / 2, length / 2)
                    )

                # Round coordinates to nearest integer
                start_point, end_point = torch.round(start_point), torch.round(end_point)

                # Determine if this segment exceeds the seg_length, split it if necessary
                segment_length = torch.norm(end_point - start_point)
                if segment_length > seg_length:
                    # Calculate new midpoint for splitting the segment
                    new_midpoint = (start_point + end_point) / 2
                    # First half
                    segments.append(
                        {
                            "segment": torch.stack([start_point, new_midpoint], dim=1),
                            "type": "C" + seg_type_label if i in [-full_steps] else seg_type_label,
                            "id": segment_id,
                        }
                    )
                    segment_id += 1
                    # Second half
                    segments.append(
                        {
                            "segment": torch.stack([new_midpoint, end_point], dim=1),
                            "type": "C" + seg_type_label if i in [full_steps] else seg_type_label,
                            "id": segment_id,
                        }
                    )
                    segment_id += 1
                else:
                    segments.append(
                        {
                            "segment": torch.stack([start_point, end_point], dim=1),
                            "type": "C" + seg_type_label
                            if i in [-full_steps, full_steps]
                            else seg_type_label,
                            "id": segment_id,
                        }
                    )
                    segment_id += 1
        return segments, segment_id

    all_segments = []
    segment_id = 0  # Initialize global segment ID

    # Process vertical segments
    for vpose in vposes:
        new_segments, segment_id = split_edge(vpose, "V", segment_id)
        all_segments.extend(new_segments)

    # Process horizontal segments
    for hpose in hposes:
        new_segments, segment_id = split_edge(hpose, "H", segment_id)
        all_segments.extend(new_segments)

    return all_segments


def visualize_segments(image, vsegments, hsegments):
    """Visualizes vertical and horizontal segments over an image. Vertical segments are displayed
    in blue, and horizontal segments in red.

    Args:
        image (numpy.ndarray): The original image on which to overlay the segments.
        vsegments (list of torch.Tensor): A list where each element is a tensor representing
                                        the segmented parts of a vertical line.
        hsegments (list of torch.Tensor): A list where each element is a tensor representing
                                        the segmented parts of a horizontal line.
    """
    if torch.is_tensor(image):
        image = image.cpu().numpy()
    # Convert the image to RGB if it is grayscale for colored line drawing
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    segment_number = 1

    def draw_segments(segments, color):
        """Draws the given segments on the image in the specified color.

        Args:
            segments (list of torch.Tensor): A list of tensors representing line segments.
            color (tuple): The color to use for drawing the segments as (B, G, R).
        """
        nonlocal segment_number
        for seg_list in segments:
            for seg in seg_list:
                start_point = (int(seg[1, 0].item()), int(seg[0, 0].item()))
                end_point = (int(seg[1, 1].item()), int(seg[0, 1].item()))
                mid_point = (
                    (start_point[0] + end_point[0]) // 2,
                    (start_point[1] + end_point[1]) // 2,
                )
                cv2.line(image, start_point, end_point, color, thickness=2)
                cv2.circle(image, start_point, 3, color, -1)
                cv2.putText(
                    image,
                    f"{segment_number}",
                    mid_point,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )
                segment_number += 1

    # Blue for vertical segments, red for horizontal segments
    draw_segments(vsegments, (255, 0, 0))  # Blue in BGR format
    draw_segments(hsegments, (0, 0, 255))  # Red in BGR format

    # Use Matplotlib to display the image
    plt.imshow(image)
    plt.axis("off")  # Hide axis
    plt.title("Segmented Lines on Image")
    plt.show()


def visualize_segments_with_labels(image, segments, seg_name=None):
    """Visualizes segments over an image, coloring them based on their type. Vertical segments are
    blue, horizontal segments are red, and corner segments are yellow.

    Args:
        image (numpy.ndarray): The original image on which to overlay the segments.
        segments (list): A list of dictionaries, each containing a 'segment' (torch.Tensor
                        representing the segmented part of a line with integer coordinates),
                        'type' (string label of the segment type), and 'id' (unique identifier).
    """
    if torch.is_tensor(image):
        image = image.cpu().numpy()
    # Convert the image to RGB if it is grayscale for colored line drawing
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    color_map = {
        "V": (255, 0, 0),  # Blue
        "H": (0, 0, 255),  # Red
        "CV": (255, 255, 0),  # Yellow
        "CH": (255, 255, 0),  # Yellow
    }

    for seg_info in segments:
        seg = seg_info["segment"]
        seg_type = seg_info["type"]
        seg_id = seg_info["id"]
        color = color_map.get(seg_type, (255, 255, 255))  # Default to white if type unknown
        start_point = (int(seg[1, 0].item()), int(seg[0, 0].item()))
        end_point = (int(seg[1, 1].item()), int(seg[0, 1].item()))
        mid_point = (
            (start_point[0] + end_point[0]) // 2,
            (start_point[1] + end_point[1]) // 2,
        )
        cv2.line(image, start_point, end_point, color, thickness=2)
        cv2.circle(image, start_point, 3, color, -1)
        cv2.circle(image, end_point, 3, color, -1)
        # cv2.putText(
        #     image,
        #     f"{start_point[0]}",
        #     start_point,
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.3,
        #     color,
        #     1,
        #     cv2.LINE_AA,
        # )
        cv2.putText(
            image,
            f"{seg_id}",
            mid_point,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    # Use Matplotlib to display the image
    plt.imshow(image)
    plt.axis("off")  # Hide axis
    plt.title("Segmented Lines with Labels on Image")
    if seg_name:
        plt.savefig(seg_name, bbox_inches="tight", dpi=300)
    else:
        plt.show()
    # plt.show()


# Example usage:
# Assuming `image` is a numpy.ndarray representing your image,
# and `segments` is the output from the segment_lines_with_labels function.
# visualize_segments_with_labels(image, segments)


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
    # maskfile = "./benchmark/edge_bench/edge_test1.glp"
    # mask_shape = (512, 512)
    # litho = LithoSim("./configs/litho/default.yaml")
    # test = Basic(litho, 0.5)
    # epeCheck = EPEChecker(litho, 0.5)
    # shotCount = ShotCounter(litho, 0.5)

    # ref = glp.Design(targetfile, down=1)
    # ref.center(2048, 2048, 0, 0)
    # target = ref.mat(2048, 2048, 0, 0)
    for i in range(1, 10):
        maskfile = f"./benchmark/ICCAD2013/M1_test{i}.glp"
        mask_shape = (1280, 1280)
        mref = glp.Design(maskfile, down=1)
        mref.center(mask_shape[0], mask_shape[1], 0, 0)
        mask = mref.mat(mask_shape[0], mask_shape[1], 0, 0)
        mask_tensor = torch.tensor(mask, dtype=REALTYPE, device=DEVICE)
        vposes, hposes = boundaries(mask_tensor)
        segs = segment_lines_with_labels(vposes, hposes, SEG_LENGTH)
        seg_name = f"./tmp/segs/ICCAD2013/M1_test{i}_seg.png"
        visualize_segments_with_labels(mask_tensor, segs, seg_name)
    # visualizeBoundaries(mask_tensor, vposes, hposes)
    # vsegs = segment_lines(vposes, SEG_LENGTH)
    # hsegs = segment_lines(hposes, SEG_LENGTH)
    # visualize_segments(mask_tensor, vsegs, hsegs)

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
