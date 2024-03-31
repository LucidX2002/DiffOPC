import sys

sys.path.append(".")
import math
import multiprocessing as mp
from itertools import groupby

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim

import src.data.loaders.segments as SegLoader
from src.data.datatype import COMPLEXTYPE, INTTYPE, REALTYPE

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

from kornia.utils import draw_convex_polygon
from rich import print

from src.data.loaders import glp_seg

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


def segment_polygon_edges_with_labels(polygon_edges, seg_length, device="cpu"):
    """Segments vertical and horizontal lines into smaller segments of a fixed length, labels
    vertical (V), horizontal (H), corner vertical (CV), and corner horizontal (CH) segments, and
    assigns a unique ID to each segment.

    Args:
        polygon_edges (torch.Tensor): Tensor of shape [P, N, 2, 2] representing the,
        P: polygon number,
        N: edge number,
        2: start and end point of the edge,
        2: 2-D space (x, y)
        seg_length (float): The fixed length for the segments.
    Returns:
        polygon_edge_list: A list of polygons, each contains segmented lines with labels and IDs.
            Each element in the list is a dictionary
            with 'segment' (tensor representing the segmented part of a line with integer coordinates),
            'type' (string label of the segment type), and 'id' (unique identifier).
    """
    device = "cpu"

    def split_edge(edge, segment_id):
        midpoint = torch.mean(edge, dim=1).round()
        if not torch.equal(edge, edge.round()):
            raise ValueError("Edge is not integer.")

        vector = edge[:, 1] - edge[:, 0]
        length = torch.norm(vector)
        direction = vector / length
        is_horizontal = torch.abs(direction[0]) > torch.abs(direction[1])
        seg_type_label = "H" if is_horizontal else "V"

        segments = []
        if length < seg_length:
            # Treat both ends as corners if the segment is short
            seg_type_label = "C" + seg_type_label
            segments.extend(
                create_segment(
                    edge[:, 0], midpoint, seg_type_label, segment_id, True, False, direction
                )
            )
            segments.extend(
                create_segment(
                    midpoint, edge[:, 1], seg_type_label, segment_id + 1, False, True, direction
                )
            )
            segment_id += 2
        else:
            steps_to_edge = length / 2 / seg_length
            full_steps = max(int(steps_to_edge), 1)

            for i in range(-full_steps, full_steps + 1):
                start_point = (
                    edge[:, 0]
                    if i == -full_steps
                    else midpoint + direction * (i * seg_length - min(seg_length / 2, length / 2))
                )
                end_point = (
                    edge[:, 1]
                    if i == full_steps
                    else midpoint + direction * (i * seg_length + min(seg_length / 2, length / 2))
                )

                start_point, end_point = start_point.round(), end_point.round()
                segment_length = torch.norm(end_point - start_point)

                if segment_length > seg_length:
                    new_midpoint = (start_point + end_point) / 2
                    new_midpoint = new_midpoint.round()
                    segments.extend(
                        create_segment(
                            start_point,
                            new_midpoint,
                            seg_type_label,
                            segment_id,
                            i == -full_steps,
                            False,
                            direction,
                        )
                    )
                    segments.extend(
                        create_segment(
                            new_midpoint,
                            end_point,
                            seg_type_label,
                            segment_id + 1,
                            False,
                            i == full_steps,
                            direction,
                        )
                    )
                    segment_id += 2
                else:
                    segments.extend(
                        create_segment(
                            start_point,
                            end_point,
                            seg_type_label,
                            segment_id,
                            i == -full_steps,
                            i == full_steps,
                            direction,
                        )
                    )
                    segment_id += 1
        # the last segment should be the end of the polygon
        return segments, segment_id

    def create_segment(
        start_point, end_point, seg_type_label, segment_id, is_start, is_end, direction
    ):
        return [
            {
                "segment": torch.stack([start_point, end_point], dim=1).requires_grad_(),
                "type": ("C" if is_start or is_end else "") + seg_type_label,
                "id": segment_id,
                "start": is_start,
                "end": is_end,
                "next": None if is_end else segment_id + 1,
                "direction": direction,
            }
        ]

    all_segments = []
    segment_id = 0  # Initialize global segment ID
    # Process vertical segments
    for poly in polygon_edges:
        poly_segments = []
        for poly_id, edge in enumerate(poly):
            edge = torch.tensor(edge, dtype=REALTYPE, device=device)
            new_segments, segment_id = split_edge(edge, segment_id)
            poly_segments.extend(new_segments)
        all_segments.append(poly_segments)

    return all_segments


def create_binary_mask(polygons, width, height, device=None):
    """Create a binary mask where points inside any of the polygons are marked as 1 and others as
    0.

    Args:
        polygons: A list of polygons, where each polygon is represented by its vertex coordinates with shape (n, 2),
                where n is the number of vertices.
        width: Width of the plane.
        height: Height of the plane.

    Returns:
        A binary mask with shape (height, width), where points inside any of the polygons are marked as 1 and others as 0.
    """
    # Determine the device to use
    if device is None:
        device = polygons[0].device

    # Create a grid representing all points on the plane
    x = torch.arange(width, dtype=torch.float32, device=device)
    y = torch.arange(height, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    points = torch.stack([grid_x, grid_y], dim=-1)

    # Initialize the binary mask
    mask = torch.zeros_like(grid_x, dtype=torch.bool)

    # Extract edges and polygon IDs
    edges = []
    polygon_ids = []
    for i, polygon in enumerate(polygons):
        polygon_edges = torch.cat([polygon, polygon[:1]], dim=0)
        edges.append(polygon_edges)
        polygon_ids.append(torch.full((len(polygon_edges),), i, dtype=torch.int32))
    edges = torch.cat(edges, dim=0)
    polygon_ids = torch.cat(polygon_ids, dim=0)

    # Initialize the intersection counter
    count = torch.zeros_like(grid_x, dtype=torch.int32)
    for i in range(len(edges) - 1):
        if polygon_ids[i] == polygon_ids[i + 1]:
            # Calculate the vectors from each point to the edge endpoints
            v1 = edges[i] - points
            v2 = edges[i + 1] - points

            # Calculate the cross product of v1 and v2
            cross = v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]
            # Check if the point is on the edge
            on_edge = (v1[..., 0] == 0) & (v1[..., 1] >= 0) & (v2[..., 1] < 0)
            # Check if the point is inside the polygon
            inside = ((v1[..., 0] < 0) & (v2[..., 0] >= 0) & (cross < 0)) | (
                (v1[..., 0] >= 0) & (v2[..., 0] < 0) & (cross > 0)
            )
            # Increment the count for points inside the polygon or on the edge
            count += (inside | on_edge).int()
    # If the count is odd, the point is inside at least one polygon
    mask = count % 2 == 1
    return mask


def create_binary_mask_from_vertices(vertices, vertices_polygon_ids, width, height, device=None):
    """Create a binary mask where points inside any of the polygons are marked as 1 and others as
    0.

    Args:
    vertices (torch.Tensor): Vertice tensor of shape [M, 2], where M is the total number of vertices,                            and 2 represents 2-D coordinates (x, y)
    vertices_polygon_ids (torch.Tensor): Tensor of shape [M] containing polygon IDs for each vertex
    width: Width of the plane.
    height: Height of the plane.
    device: Device to use for computations (default: None)

    Returns:
    A binary mask with shape (height, width), where points inside any of the polygons are marked as 1 and others as 0.
    """
    # Determine the device to use
    if device is None:
        device = vertices.device

    # Create a grid representing all points on the plane
    x = torch.arange(width, dtype=torch.float32, device=device)
    y = torch.arange(height, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    points = torch.stack([grid_x, grid_y], dim=-1)

    # Initialize the binary mask
    mask = torch.zeros_like(grid_x, dtype=torch.bool)

    # Get the unique polygon IDs
    unique_ids = torch.unique(vertices_polygon_ids)

    # Initialize the intersection counter
    count = torch.zeros_like(grid_x, dtype=torch.int32)

    # Iterate over each polygon ID
    for idx in unique_ids:
        # Get the vertices corresponding to the current polygon ID
        polygon_vertices = vertices[vertices_polygon_ids == idx]
        # Create edges by connecting consecutive vertices and closing the polygon
        polygon_edges = torch.cat([polygon_vertices, polygon_vertices[:1]], dim=0)

        for i in range(len(polygon_edges) - 1):
            # Calculate the vectors from each point to the edge endpoints
            v1 = polygon_edges[i] - points
            v2 = polygon_edges[i + 1] - points

            # Calculate the cross product of v1 and v2
            cross = v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]

            # Check if the point is on the edge
            on_edge = (v1[..., 0] == 0) & (v1[..., 1] >= 0) & (v2[..., 1] < 0)

            # Check if the point is inside the polygon
            inside = ((v1[..., 0] < 0) & (v2[..., 0] >= 0) & (cross < 0)) | (
                (v1[..., 0] >= 0) & (v2[..., 0] < 0) & (cross > 0)
            )

            # Increment the count for points inside the polygon or on the edge
            count += (inside | on_edge).int()

    # If the count is odd, the point is inside at least one polygon
    mask = count % 2 == 1

    return mask


def right_perpendicular_unit_vector(vector):
    # Check if the input is a 2D vector
    assert vector.size(0) == 2, "Input must be a 2D vector"
    # Extract the components of the vector
    x, y = vector
    # Compute the magnitude of the vector
    magnitude = torch.sqrt(x**2 + y**2)
    # Check if the vector is a zero vector
    if magnitude == 0:
        raise ValueError("Input cannot be a zero vector")
    # Compute the right-hand perpendicular unit vector
    unit_vector = torch.tensor([y, -x]) / magnitude
    return unit_vector


def edges_to_vertices(edges, polygon_ids):
    """Convert polygon edge representation to vertice representation.

    Args:
    edges (torch.Tensor): Edge tensor of shape [N, 2, 2], where N is the number of edges,
                        2 represents the start and end points, and 2 represents 2-D coordinates (x, y)
    polygon_ids (torch.Tensor): Tensor of shape [N] containing polygon IDs for each edge

    Returns:
    vertices (torch.Tensor): Vertice tensor of shape [M, 2], where M is the total number of vertices,
                            and 2 represents 2-D coordinates (x, y)
    vertices_polygon_ids (torch.Tensor): Tensor of shape [M] containing polygon IDs for each vertex
    """
    # Get the unique polygon IDs
    unique_ids = torch.unique(polygon_ids)

    vertices_list = []
    vertices_polygon_ids_list = []

    # Iterate over each polygon ID
    for idx in unique_ids:
        # Get the indices of edges corresponding to the current polygon ID
        polygon_edge_indices = torch.where(polygon_ids == idx)[0]
        # Get the edges corresponding to the current polygon ID
        polygon_edges = edges[polygon_edge_indices]

        # Initialize the polygon vertices list with the start point of the first edge
        polygon_vertices = []

        # Iterate over the edges and add unique vertices to the polygon vertices list
        for edge in polygon_edges:
            start_point = edge[:, 0]
            end_point = edge[:, 1]

            if len(polygon_vertices) == 0 or not torch.equal(start_point, polygon_vertices[-1]):
                polygon_vertices.append(start_point)

            if not torch.equal(end_point, polygon_vertices[-1]):
                polygon_vertices.append(end_point)

        # Convert the polygon vertices list to a tensor
        polygon_vertices = torch.stack(polygon_vertices)

        vertices_list.append(polygon_vertices)

        # Create polygon IDs for each vertex
        polygon_ids_for_vertices = torch.full(
            (polygon_vertices.shape[0],), idx, dtype=polygon_ids.dtype
        )
        vertices_polygon_ids_list.append(polygon_ids_for_vertices)

    # Concatenate the vertices and polygon IDs from all polygons
    vertices = torch.cat(vertices_list, dim=0)
    vertices_polygon_ids = torch.cat(vertices_polygon_ids_list, dim=0)
    return vertices, vertices_polygon_ids


def edge_params_merge2mask(edge_params, metadata):
    edge_params = edge_params.clone().detach()
    img_shape = metadata["img_shape"]
    polygon_ids = metadata["polygon_ids"]
    vertices, vertices_polygon_ids = edges_to_vertices(edge_params, polygon_ids)
    width, height = img_shape
    binary_mask = create_binary_mask_from_vertices(vertices, vertices_polygon_ids, width, height)
    binary_mask = binary_mask.float()
    # plt.imshow(binary_mask.cpu().numpy())
    # plt.show()
    return binary_mask


def segments_merge2polygon(poly_segments, width, height):
    # Get all segments with "start" flag

    def get_seg_by_id(poly, sid):
        for seg in poly:
            if seg["id"] == sid:
                return seg
        return None

    # from edge to vertices
    poly_vertices = []
    for poly in poly_segments:
        vertices = []
        vertices.append(poly[0]["segment"][:, 0])
        # now the corner edge don't know who is the next
        for seg in poly:
            if seg["end"] is not True:
                next_id = seg["next"]
                next_seg = get_seg_by_id(poly, next_id)
                if not torch.equal(seg["segment"][:, 1], next_seg["segment"][:, 0]):
                    # line is not continuous
                    vertices.append(seg["segment"][:, 1])
                    vertices.append(next_seg["segment"][:, 0])
            else:
                # the corner segment
                vertices.append(seg["segment"][:, 1])
        vertices = torch.stack(vertices).detach()
        poly_vertices.append(vertices)

    binary_mask = create_binary_mask(poly_vertices, width=width, height=height)
    plt.imshow(binary_mask.cpu().numpy())
    plt.show()
    return poly_vertices, binary_mask


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


def validate_poly_edge_segments(polygon_edges_segments):
    print(f"Total polygon: {len(polygon_edges_segments)}")
    for pid, poly in enumerate(polygon_edges_segments):
        for seg in poly:
            if seg["next"] is not None:
                last_id = seg["next"]
            if not torch.equal(seg["segment"], torch.round(seg["segment"])):
                print(f"{seg['id']} is not integer: {seg['segment']}")

        assert last_id == poly[-1]["id"], f"Last id {last_id} segment last {poly[-1]} not match"


def visualize_segments_with_labels(
    image, poly_segments, only_start=False, only_end=False, seg_name=None
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

    for poly in poly_segments:
        for seg_info in poly:
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
            # start_point = (int(seg[1, 0].item()), int(seg[0, 0].item()))
            # end_point = (int(seg[1, 1].item()), int(seg[0, 1].item()))
            start_point = tuple(seg[:, 0].detach().cpu().numpy().astype(int))
            end_point = tuple(seg[:, 1].detach().cpu().numpy().astype(int))
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
    # plt.title("Segmented Lines with Labels on Image")
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


def draw_edge_params(edge_params, shape, show=True):
    image = torch.zeros(shape).to(device=DEVICE)
    edge_params_clone = edge_params.clone().detach()
    for edge in edge_params_clone:
        start_point = edge[:, 0].clone().detach().int()
        end_point = edge[:, 1].clone().detach().int()

        if start_point[1] == end_point[1]:  # horizontal
            if start_point[0] > end_point[0]:
                start_point, end_point = end_point, start_point
            image[start_point[1], start_point[0] : end_point[0] + 1] = 255
        else:
            if start_point[1] > end_point[1]:  # vertical
                start_point, end_point = end_point, start_point
            image[start_point[1] : end_point[1] + 1, start_point[0]] = 255
    if show:
        plt.imshow(image.cpu().numpy())
        plt.show()
    return image.cpu().numpy()


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
        # maskfile = f"./benchmark/edge_bench/edge_test{i}.glp"
        design = glp_seg.Design(maskfile)
        shape = (2048, 2048)
        offset = (512, 512)
        target, edge_params, metadata = SegLoader.SegmentsInitTorch().run(
            design, shape[0], shape[1], offset[0], offset[1]
        )
        # plt.imshow(target.cpu().numpy())
        # plt.show()
        direction_vectors = metadata["direction_vectors"]
        velocities = metadata["velocities"]
        print(edge_params[1])
        print("dir", direction_vectors[1])
        print("vel", velocities[1])
        edge_params_1 = edge_params[1].clone().detach()
        # edge_params_1 += velocities[1].T
        # print(edge_params_1)
        # edge_params_1 = edge_params_1.unsqueeze(0)
        vel_1 = velocities[1]
        vel_1 = vel_1 * 0.5
        print("vel_1", vel_1)
        print(edge_params_1)
        print(edge_params_1 + vel_1)
        velocities = velocities[:5]
        averages = torch.randn(velocities.shape[0], 1, device=DEVICE)
        averages = averages[:5].view(-1, 1, 1)
        print(averages)
        print(velocities)
        print(averages * velocities)

        # image = torch.zeros(shape).to(device=DEVICE)
        # for edge in edge_params:
        #     edge = edge.detach().cpu().numpy().astype(int)
        #     start_point = tuple(edge[:, 0])
        #     end_point = tuple(edge[:, 1])
        #     image[start_point[0]:end_point[0], start_point[1], end_point[1]] = 1
        # plt.imshow(image.cpu().numpy())
        # plt.show()

        # draw_edge_params(edge_params, shape)

        # maskfile = f"./benchmark/edge_bench/M1_part_test{i}.glp"
        # mask_shape = (1280, 1280)
        # # maskfile = f"./benchmark/edge_bench/edge_test{i}.glp"
        # # mask_shape = (512, 512)
        # mref = glp_seg.Design(maskfile, down=1)
        # mref.center(mask_shape[0], mask_shape[1], 0, 0)
        # mask = mref.mat(mask_shape[0], mask_shape[1], 0, 0)
        # mask_tensor = torch.tensor(mask, dtype=REALTYPE, device=DEVICE)
        # # vposes, hposes, metadata = boundaries(mask_tensor)
        # # print(metadata)
        # mask_edges = mref.polygon_edges
        # # print(mask_edges)
        # # mask_edges_tensor = torch.tensor(mask_edges, dtype=REALTYPE, device=DEVICE)
        # segs = segment_polygon_edges_with_labels(mask_edges, SEG_LENGTH)
        # # validate_poly_edge_segments(segs)
        # merged_polygons = segments_merge2polygon(segs, mask_shape[0], mask_shape[1])

        # visual_seg2poly(segs, metadata)
        # seg_name = f"./tmp/segs/edge/edge_test{i}_seg_start.png"
        # visualize_segments_with_labels(mask_tensor, segs)
        # seg_name = f"./tmp/segs/ICCAD2013/M1_test{i}_seg.png"
        # visualize_segments_with_labels(
        #     mask_tensor, segs, only_start=False, only_end=False, seg_name=seg_name
        # )
        # seg_name = f"./tmp/segs/ICCAD2013/M1_test{i}_seg_start.png"
        # visualize_segments_with_labels(
        #     mask_tensor, segs, only_start=True, only_end=False, seg_name=seg_name
        # )
        # seg_name = f"./tmp/segs/ICCAD2013/M1_test{i}_seg_end.png"
        # visualize_segments_with_labels(
        #     mask_tensor, segs, only_start=False, only_end=True, seg_name=seg_name
        # )

        # seg_name = f"./tmp/segs/edge/M1_part_test{i}_seg.png"
        # visualize_segments_with_labels(
        #     mask_tensor, segs, only_start=False, only_end=False, seg_name=seg_name
        # )
        # seg_name = f"./tmp/segs/edge/M1_part_test{i}_seg_start.png"
        # visualize_segments_with_labels(
        #     mask_tensor, segs, only_start=True, only_end=False, seg_name=seg_name
        # )
        # seg_name = f"./tmp/segs/edge/M1_part_test{i}_seg_end.png"
        # visualize_segments_with_labels(
        #     mask_tensor, segs, only_start=False, only_end=True, seg_name=seg_name
        # )

        # seg_name = f"./tmp/segs/edge/edge_test{i}_seg.png"
        # visualize_segments_with_labels(
        #     mask_tensor, segs, only_start=False, only_end=False, seg_name=seg_name
        # )
        # seg_name = f"./tmp/segs/edge/edge_test{i}_seg_start.png"
        # visualize_segments_with_labels(
        #     mask_tensor, segs, only_start=True, only_end=False, seg_name=seg_name
        # )
        # seg_name = f"./tmp/segs/edge/edge_test{i}_seg_end.png"
        # visualize_segments_with_labels(
        #     mask_tensor, segs, only_start=False, only_end=True, seg_name=seg_name
        # )
