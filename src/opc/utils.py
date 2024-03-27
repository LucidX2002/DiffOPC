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

from rich import print

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
    metadata = {
        "shape": target.shape,
    }
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

    return vposes.float(), hposes.float(), metadata


def visualizeBoundaries(target, vposes, hposes):
    """Visualize vertical and horizontal boundaries on the given target image, and annotate each
    point with its coordinates, considering the coordinates are in the format [[y1, x1], [y2, x2]].
    The target image, vposes, and hposes are expected to be PyTorch tensors or NumPy arrays.

    Parameters:
    - target: Image tensor or array on which to draw the boundaries.
    - vposes: Tensor or array of shape (N, 2, D) representing N vertical lines, with points in (y, x) format.
    - hposes: Tensor or array of shape (N, 2, D) representing N horizontal lines, with points in (y, x) format.

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


def segment_lines_with_labels(vposes, hposes, seg_length=SEG_LENGTH, device=DEVICE):
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
        # seg_length = SEG_LENGTH
        midpoint = torch.mean(edge, dim=1)
        vector = edge[:, 1] - edge[:, 0]
        length = torch.norm(vector)
        direction = vector / length

        backup_seg_id = segment_id
        segments = []

        # deal with the short edges
        if length < seg_length:
            # Treat both ends as corners if the segment is short
            seg_type_label = "C" + seg_type_label
            start_point = torch.round(edge[:, 0])
            end_point = torch.round(midpoint)
            seg = {
                "segment": torch.stack([start_point, end_point], dim=1).requires_grad_(),
                "type": seg_type_label,
                "id": segment_id,
                "start": True,
                "end": False,
                "next": segment_id + 1,
            }
            segments.append(seg)
            segment_id += 1

            start_point = torch.round(midpoint)
            end_point = torch.round(edge[:, 1])
            segments.append(
                {
                    "segment": torch.stack([start_point, end_point], dim=1).requires_grad_(),
                    "type": seg_type_label,
                    "id": segment_id,
                    "start": False,
                    "end": True,
                    "next": None,
                }
            )
            segment_id += 1
        # the long edges
        else:
            # Calculate segments from the midpoint to each end.
            steps_to_edge = length / 2 / seg_length
            full_steps = int(steps_to_edge)
            if full_steps == 0:
                full_steps = 1

            for i in range(-full_steps, full_steps + 1):
                # the left most
                if i == -full_steps:
                    start_point = edge[:, 0]
                    left_most = True
                else:
                    start_point = midpoint + direction * (
                        i * seg_length - min(seg_length / 2, length / 2)
                    )
                    left_most = False

                # the right most
                if i == full_steps:
                    end_point = edge[:, 1]
                    right_most = True
                else:
                    end_point = midpoint + direction * (
                        i * seg_length + min(seg_length / 2, length / 2)
                    )
                    right_most = False

                # Normal edges
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
                            "segment": torch.stack(
                                [start_point, new_midpoint], dim=1
                            ).requires_grad_(),
                            "type": "C" + seg_type_label if i in [-full_steps] else seg_type_label,
                            "id": segment_id,
                            "start": left_most,
                            "end": False,
                            "next": segment_id + 1,
                        }
                    )
                    segment_id += 1
                    # Second half
                    segments.append(
                        {
                            "segment": torch.stack(
                                [new_midpoint, end_point], dim=1
                            ).requires_grad_(),
                            "type": "C" + seg_type_label if i in [full_steps] else seg_type_label,
                            "id": segment_id,
                            "start": False,
                            "end": right_most,
                            "next": segment_id + 1 if right_most is False else None,
                        }
                    )
                    segment_id += 1
                else:
                    seg = {
                        "segment": torch.stack([start_point, end_point], dim=1).requires_grad_(),
                        "type": "C" + seg_type_label
                        if i in [-full_steps, full_steps]
                        else seg_type_label,
                        "id": segment_id,
                        "start": left_most,
                        "end": right_most,
                        "next": segment_id + 1 if right_most is False else None,
                    }
                    segments.append(seg)
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

    # Process the corner segments and connect the end line to the corner start
    # Process the corner type : ┐
    end_segs = [seg for seg in all_segments if seg["end"] is True]
    start_segs = [seg for seg in all_segments if seg["start"] is True]

    for end_seg in end_segs:
        for start_seg in start_segs:
            if torch.equal(end_seg["segment"][:, 1], start_seg["segment"][:, 0]):
                update_seg_next_byid(end_seg, end_seg["id"], start_seg["id"])

    # now, only the end_seg [: 1], start_seg [: 0] are connected
    # still, we need to define the end_seg, vertical and end_seg horizontal.
    none_end_segs = [seg for seg in end_segs if seg["next"] is None]
    CV_none_ends = [seg for seg in none_end_segs if seg["type"] == "CV"]
    CH_none_ends = [seg for seg in none_end_segs if seg["type"] == "CH"]
    # process the corner type : ┘
    for cv in CV_none_ends:
        for ch in CH_none_ends:
            if torch.equal(cv["segment"][:, 1], ch["segment"][:, 1]):
                update_seg_next_byid(cv, cv["id"], ch["id"])
                update_seg_next_byid(ch, ch["id"], cv["id"])
                break

    # process the corner type: ┌, and do special mark
    for start_seg in start_segs:
        for start_seg2 in start_segs:
            # start points are the same
            if "prev" in start_seg2.keys():
                continue
            if start_seg["id"] == start_seg2["id"]:
                continue
            if torch.equal(start_seg["segment"][:, 0], start_seg2["segment"][:, 0]):
                update_seg_prev_byid(start_seg, start_seg["id"], start_seg2["id"])
                update_seg_prev_byid(start_seg2, start_seg2["id"], start_seg["id"])
                break
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


def traverse_from_start(all_segments, start_seg):
    traverse_list = []
    end_pairs = []
    while start_seg["next"] is not None:
        traverse_list.append(start_seg["id"])
        next_start_seg = get_segment_obj_byid(all_segments, start_seg["next"])
        # print(f"{start_seg['id']} -> {next_start_seg['id']}")
        if start_seg["next"] == next_start_seg["id"] and next_start_seg["next"] == start_seg["id"]:
            ep = [start_seg["id"], next_start_seg["id"]]
            ep.sort()
            ep = tuple(ep)
            end_pairs.append(ep)
            break
        else:
            start_seg = next_start_seg.copy()
    # for seg_id in traverse_list:
    # seg_traverse_list_str = [f"{str(seg['id'])} -> " for seg in traverse_list[:-1]]
    # seg_traverse_list_str.append(str(traverse_list[-1]["id"]))
    # print(" -> ".join([str(id) for id in traverse_list]))
    return traverse_list, end_pairs


def remove_duplicates(lst):
    # Create an empty dictionary to store the result
    result_dict = {}
    # Iterate over each sublist in the main list
    for sub_list in lst:
        # Get the tail number of the sublist
        tail_num = sub_list[-1]
        # If the tail number already exists in the dictionary, compare the lengths of sublists
        if tail_num in result_dict:
            if len(sub_list) > len(result_dict[tail_num]):
                result_dict[tail_num] = sub_list
        # If the tail number doesn't exist in the dictionary, add the sublist to the dictionary
        else:
            result_dict[tail_num] = sub_list
    # Return a list of all values (i.e., deduplicated sublists) from the dictionary
    return list(result_dict.values())


# Test code
# lst = [[1, 2, 3], [4, 5, 6, 3], [7, 8, 9], [10, 11, 9]]
# result = remove_duplicates(lst)
# print(result)

# def segments_merge2polygon(segments):
#     start_segs = [seg for seg in segments if seg["start"] == True]
#     all_traverse_list = []
#     all_end_pairs = []
#     all_start_pairs = []
#     # for the corner start we need the prev info to mark all polygon
#     for seg in start_segs:
#         if "prev" in seg.keys():
#             start_pair = [seg["id"], seg["prev"]]
#             start_pair.sort()
#             start_pair = tuple(start_pair)
#         traverse_from_single, end_pairs_from_single = traverse_from_start(segments, seg)
#         all_traverse_list.append(traverse_from_single)
#         all_end_pairs.extend(end_pairs_from_single)
#         all_start_pairs.append(start_pair)
#     all_traverse_list = remove_duplicates(all_traverse_list)
#     all_end_pairs = list(set(all_end_pairs))
#     all_start_pairs = list(set(all_start_pairs))
#     print(all_traverse_list)
#     print(all_end_pairs)
#     print(all_start_pairs)

#     def get_sub_by_start(all_traverse_list, start_id):
#         for sub in all_traverse_list:
#             if sub[0] == start_id:
#                 return sub
#         raise ValueError("Start ID not found in the traverse list.")


#     def is_true_ep(ep):
#         if ep in all_end_pairs:
#             return True
#         else:
#             return False

#     def inside_fake_ep(ep, all_fake_eps):
#         flattened_list = [item for tuple_pair in all_fake_eps for item in tuple_pair]
#         if ep[0] in flattened_list and ep[1] in flattened_list:
#             return True
#         else:
#             return False

#     def get_ep_by_sp(sp):
#         p0 = get_sub_by_start(all_traverse_list, sp[0])
#         p1 = get_sub_by_start(all_traverse_list, sp[1])
#         ep = [p0[-1], p1[-1]]
#         ep.sort()
#         ep = tuple(ep)
#         return ep

#     def get_fake_ep_by_ep(cur_ep, all_eps):
#         # get fake_ep[0]
#         # for all_ep in all_eps:
#             # if all_ep[0] == cur_ep[0]:
#                 # return all_ep
#         fake_ep = [None, None]
#         for i in range(len(cur_ep)):
#             for ep in all_eps:
#                 if ep[0] == cur_ep[i]:
#                     fake_ep[i] = ep[1]
#                     # print(f"remove ep: {ep}")
#                     # all_eps.remove(ep)
#                 if ep[1] == cur_ep[i]:
#                     fake_ep[i] = ep[0]
#                     # print(f"remove ep: {ep}")
#                     # all_eps.remove(ep)
#         fake_ep.sort()
#         fake_ep = tuple(fake_ep)
#         return fake_ep

#     all_polygons = []
#     def recursion_polygon(all_sps, all_eps, all_fake_eps = None, polygon_by_sp = None):
#         print("*"*20, "program start", "*"*20)
#         print(f"current all polygons: {all_polygons}")
#         if polygon_by_sp is None:
#             polygon_by_sp = []
#         if all_fake_eps is None:
#             all_fake_eps = []
#         print(f"recursion level : {len(all_fake_eps)},\n all_sps: {all_sps}, all_eps: {all_eps},\n all_fake_eps: {all_fake_eps}")
#         if len(all_sps) == 0:
#             return
#         while len(all_sps) > 0:
#             for start_pair in all_sps:
#                 # if the start_pair belongs to the original polygon
#                 ep = get_ep_by_sp(start_pair)
#                 print(f"sp: {start_pair}, -> ep: {ep}")
#                 # the polygon ends here
#                 if len(all_fake_eps) > 0:
#                     print("in recursion, all_fake_eps > 0")
#                     same_polygon = False
#                     for tuple_pair in all_fake_eps:
#                         if ep[0] in tuple_pair or ep[1] in tuple_pair:
#                             same_polygon = True
#                             break
#                     if same_polygon:
#                         print(f"same polygon: {start_pair}")
#                         if inside_fake_ep(ep, all_fake_eps):
#                             print(f"same polygon:,{ep} inside fake ep: {all_fake_eps}")
#                             polygon_by_sp.append(start_pair)
#                             all_sps.remove(start_pair)
#                             print(f"all_sps after remove: {all_sps}")
#                             all_polygons.append(polygon_by_sp)
#                             polygon_by_sp = []
#                             all_fake_eps = []
#                         else:
#                             print(f"same polygon,  {ep} NOT inside all fake ep:{all_fake_eps}")
#                             fake_ep = get_fake_ep_by_ep(ep, all_eps)
#                             print(f"generate fake ep: {fake_ep}")
#                             all_fake_eps.append(fake_ep)
#                             all_sps.remove(start_pair)
#                             polygon_by_sp.append(start_pair)
#                             print(f"all_sps after remove: {all_sps}")
#                             print(f"all_fake_eps: {all_fake_eps}")
#                             print(f"all true eps {all_eps}")
#                             recursion_polygon(all_sps, all_eps, all_fake_eps, polygon_by_sp)
#                     else:
#                         print(f"NOT same polygon: {start_pair}")
#                         continue
#                 else:
#                     if is_true_ep(ep):
#                         polygon_by_sp.append(start_pair)
#                         print(f"start_pair: {start_pair}")
#                         all_sps.remove(start_pair)
#                         print(f"all_sps after remove: {all_sps}")
#                         print(f"true ep: {ep}")
#                         all_polygons.append(polygon_by_sp)
#                         polygon_by_sp = []
#                         # all_eps.remove(ep)
#                         if len(all_eps) == 0:
#                             return
#                         else:
#                             recursion_polygon(all_sps, all_eps)
#                     else:
#                         print(f"not true ep: {ep}")
#                         fake_ep = get_fake_ep_by_ep(ep, all_eps)
#                         print(f"generate fake ep: {fake_ep}")
#                         all_fake_eps.append(fake_ep)
#                         all_sps.remove(start_pair)
#                         polygon_by_sp.append(start_pair)
#                         print(f"all_sps after remove: {all_sps}")
#                         print(f"all_fake_eps: {all_fake_eps}")
#                         recursion_polygon(all_sps, all_eps, all_fake_eps, polygon_by_sp)
#     recursion_polygon(all_start_pairs, all_end_pairs)
#     print(f"number of polygons: {len(all_polygons)}")
#     print(all_polygons)


#     return all_traverse_list


def segments_merge2polygon(segments):
    # Get all segments with "start" flag
    start_segs = [seg for seg in segments if seg["start"]]
    all_traverse_list = []
    all_end_pairs = []
    all_start_pairs = []

    for seg in start_segs:
        # If segment has "prev" key, create a sorted tuple of current and previous segment IDs
        if "prev" in seg:
            start_pair = tuple(sorted([seg["id"], seg["prev"]]))
        # Traverse from the current segment and get the traverse list and end pairs
        traverse_from_single, end_pairs_from_single = traverse_from_start(segments, seg)
        all_traverse_list.append(traverse_from_single)
        all_end_pairs.extend(end_pairs_from_single)
        all_start_pairs.append(start_pair)

    # Remove duplicates from traverse list and convert end and start pairs to list of unique tuples
    all_traverse_list = remove_duplicates(all_traverse_list)
    all_end_pairs = list(set(all_end_pairs))
    all_start_pairs = list(set(all_start_pairs))

    def get_sub_by_start(start_id):
        # Get the traverse sublist with the given start ID
        for sub in all_traverse_list:
            if sub[0] == start_id:
                return sub
        raise ValueError("Start ID not found in the traverse list.")

    def get_ep_by_sp(sp):
        # Get the end pair corresponding to the given start pair
        p0, p1 = get_sub_by_start(sp[0]), get_sub_by_start(sp[1])
        return tuple(sorted([p0[-1], p1[-1]]))

    def get_fake_ep(ep, all_eps):
        # Get the fake end pair by replacing each end point with its corresponding fake end point
        fake_ep = [None, None]
        for i, e in enumerate(ep):
            for ep2 in all_eps:
                if e == ep2[0]:
                    fake_ep[i] = ep2[1]
                elif e == ep2[1]:
                    fake_ep[i] = ep2[0]
        return tuple(sorted(fake_ep))

    all_polygons = []

    def recursion_polygon(all_sps, all_eps, all_fake_eps=None, polygon_by_sp=None):
        # Recursive function to find all polygons
        if polygon_by_sp is None:
            polygon_by_sp = []
        if all_fake_eps is None:
            all_fake_eps = []

        while all_sps:
            for start_pair in all_sps:
                ep = get_ep_by_sp(start_pair)
                if all_fake_eps:
                    # If any end point is in fake end pairs
                    if any(ep[0] in fp or ep[1] in fp for fp in all_fake_eps):
                        # If both end points are in fake end pairs, complete the polygon
                        if all(e in sum(all_fake_eps, ()) for e in ep):
                            polygon_by_sp.append(start_pair)
                            all_sps.remove(start_pair)
                            all_polygons.append(polygon_by_sp)
                            polygon_by_sp = []
                            all_fake_eps = []
                        else:
                            # If not, get the fake end pair and continue recursion
                            fake_ep = get_fake_ep(ep, all_eps)
                            all_fake_eps.append(fake_ep)
                            all_sps.remove(start_pair)
                            polygon_by_sp.append(start_pair)
                            recursion_polygon(all_sps, all_eps, all_fake_eps, polygon_by_sp)
                else:
                    # If end pair is in all end pairs, complete the polygon
                    if ep in all_end_pairs:
                        polygon_by_sp.append(start_pair)
                        all_sps.remove(start_pair)
                        all_polygons.append(polygon_by_sp)
                        polygon_by_sp = []
                        if not all_eps:
                            return
                        recursion_polygon(all_sps, all_eps)
                    else:
                        # If not, get the fake end pair and continue recursion
                        fake_ep = get_fake_ep(ep, all_eps)
                        all_fake_eps.append(fake_ep)
                        all_sps.remove(start_pair)
                        polygon_by_sp.append(start_pair)
                        recursion_polygon(all_sps, all_eps, all_fake_eps, polygon_by_sp)

    # Start recursion with all start pairs and end pairs
    recursion_polygon(all_start_pairs, all_end_pairs)
    print(f"number of polygons: {len(all_polygons)}")
    print(all_polygons)
    return all_traverse_list


def get_segment_obj_byid(segments, seg_id):
    for seg in segments:
        if seg["id"] == seg_id:
            return seg
    return None


def update_seg_prev_byid(segment, cur_seg_id, prev_id):
    if segment["id"] == cur_seg_id:
        segment["prev"] = prev_id
        return
    else:
        raise ValueError("Segment ID not match in the list.")


def update_seg_next_byid(segment, cur_seg_id, next_id):
    if segment["id"] == cur_seg_id:
        segment["next"] = next_id
        return
    else:
        raise ValueError("Segment ID not match in the list.")


def validate_segments(segments):
    """Validate the linked list of segments."""
    start_segs = [seg for seg in segments if seg["start"] is True]
    start_segs = [
        {"id": seg["id"], "next": seg["next"], "type": seg["type"]} for seg in start_segs
    ]
    print("start_segs:")
    print(start_segs)

    end_segs = [seg for seg in segments if seg["end"] is True]
    end_segs = [{"id": seg["id"], "next": seg["next"], "type": seg["type"]} for seg in end_segs]
    print("end_segs:")
    print(end_segs)

    print(f"total segments: {len(segments)}")
    ids = [seg["id"] for seg in segments]
    assert len(ids) == len(set(ids)), "ID not unique"

    none_ids = [seg["id"] for seg in segments if seg["next"] is None]
    assert len(none_ids) == 0, "None next segment exists"


def visualize_segments_with_labels(
    image, segments, only_start=False, only_end=False, seg_name=None
):
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
        "CV": (255, 140, 0),  # Orange
        "CH": (255, 140, 0),  # Orange
    }

    for seg_info in segments:
        seg = seg_info["segment"]
        seg_type = seg_info["type"]
        seg_id = seg_info["id"]
        is_start_seg = seg_info["start"]
        is_end_seg = seg_info["end"]
        if only_start and not is_start_seg:
            continue
        if only_end and not is_end_seg:
            continue
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
        cv2.putText(
            image,
            f"{seg_id}",
            (mid_point[0], mid_point[1] + 10),
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


# def seg_params2img(segs):


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
    for i in range(1, 2):
        maskfile = f"./benchmark/ICCAD2013/M1_test{i}.glp"
        # maskfile = f"./benchmark/edge_bench/M1_part_test{i}.glp"
        mask_shape = (1280, 1280)
        # maskfile = f"./benchmark/edge_bench/edge_test{i}.glp"
        # mask_shape = (512, 512)
        mref = glp.Design(maskfile, down=1)
        mref.center(mask_shape[0], mask_shape[1], 0, 0)
        mask = mref.mat(mask_shape[0], mask_shape[1], 0, 0)
        mask_tensor = torch.tensor(mask, dtype=REALTYPE, device=DEVICE)
        vposes, hposes, metadata = boundaries(mask_tensor)
        print(metadata)
        segs = segment_lines_with_labels(vposes, hposes, SEG_LENGTH)
        segments_merge2polygon(segs)
        # validate_segments(segs)
        # seg_name = f"./tmp/segs/edge/edge_test{i}_seg_start.png"
        # visualize_segments_with_labels(mask_tensor, segs, seg_name)
        seg_name = f"./tmp/segs/ICCAD2013/M1_test{i}_seg.png"
        visualize_segments_with_labels(
            mask_tensor, segs, only_start=False, only_end=False, seg_name=seg_name
        )
        seg_name = f"./tmp/segs/ICCAD2013/M1_test{i}_seg_start.png"
        visualize_segments_with_labels(
            mask_tensor, segs, only_start=True, only_end=False, seg_name=seg_name
        )
        seg_name = f"./tmp/segs/ICCAD2013/M1_test{i}_seg_end.png"
        visualize_segments_with_labels(
            mask_tensor, segs, only_start=False, only_end=True, seg_name=seg_name
        )

        # seg_name = f"./tmp/segs/edge/M1_part_test{i}_seg.png"
        # visualize_segments_with_labels(mask_tensor, segs, only_start=False, only_end=False, seg_name=seg_name)
        # seg_name = f"./tmp/segs/edge/M1_part_test{i}_seg_start.png"
        # visualize_segments_with_labels(mask_tensor, segs, only_start=True, only_end=False, seg_name=seg_name)
        # seg_name = f"./tmp/segs/edge/M1_part_test{i}_seg_end.png"
        # visualize_segments_with_labels(mask_tensor, segs, only_start=False, only_end=True, seg_name=seg_name)
