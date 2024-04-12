import torch


def create_binary_mask_from_vertices_best_but_edge_wrong(vertices, vertices_polygon_ids, width, height, device=None):
    """Create a binary mask where points inside any of the polygons are marked as 1 and others as
    0.

    Args:
        vertices (torch.Tensor): Vertice tensor of shape [M, 2], where M is the total number of vertices, and 2 represents 2-D coordinates (x, y)
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

    # Initialize the binary mask
    mask = torch.zeros((height, width), dtype=torch.bool, device=device)

    # Get the unique polygon IDs
    unique_ids = torch.unique(vertices_polygon_ids)

    # Iterate over each polygon ID
    for idx in unique_ids:
        # Get the vertices corresponding to the current polygon ID
        polygon_vertices = vertices[vertices_polygon_ids == idx]

        # Determine the bounding box of the current polygon
        min_x, _ = torch.min(polygon_vertices[:, 0], dim=0)
        max_x, _ = torch.max(polygon_vertices[:, 0], dim=0)
        min_y, _ = torch.min(polygon_vertices[:, 1], dim=0)
        max_y, _ = torch.max(polygon_vertices[:, 1], dim=0)

        # Create a grid representing points within the bounding box of the current polygon
        x = torch.arange(min_x, max_x + 1, dtype=torch.float32, device=device)
        y = torch.arange(min_y, max_y + 1, dtype=torch.float32, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        points = torch.stack([grid_x, grid_y], dim=-1)

        # Initialize the intersection counter for the current polygon
        count = torch.zeros_like(grid_x, dtype=torch.int32)

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
            inside = ((v1[..., 0] < 0) & (v2[..., 0] >= 0) & (cross < 0)) | ((v1[..., 0] >= 0) & (v2[..., 0] < 0) & (cross > 0))

            # Increment the count for points inside the polygon or on the edge
            count += (inside | on_edge).int()

        # If the count is odd, the point is inside the current polygon
        mask[min_y.int() : max_y.int() + 1, min_x.int() : max_x.int() + 1] |= count % 2 == 1
    return mask

    # print(f"points: {points.shape}")
    # print(points[0, 0])
    # print("polygon_vertices")
    # print(polygon_vertices)
    # print(f"inside: {inside.shape}")
    # print(inside)
    # print(f"on_edge: {on_edge.shape}")
    # print(on_edge)
    # print(f"count: {count.shape}")
    # print(count)
    # print(f"count[0]: {count[0]}")
    # return


def create_binary_mask_from_vertices_full_region(vertices, vertices_polygon_ids, width, height, device=None):
    """Create a binary mask where points inside any of the polygons are marked as 1 and others as
    0.

    Args:
    vertices (torch.Tensor): Vertice tensor of shape [M, 2], where M is the total number of vertices,
    and 2 represents 2-D coordinates (x, y)
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
            inside = ((v1[..., 0] < 0) & (v2[..., 0] >= 0) & (cross < 0)) | ((v1[..., 0] >= 0) & (v2[..., 0] < 0) & (cross > 0))

            # Increment the count for points inside the polygon or on the edge
            count += (inside | on_edge).int()

    # If the count is odd, the point is inside at least one polygon
    mask = count % 2 == 1

    return mask


def create_binary_mask_from_vertices_max_vertices_region(vertices, vertices_polygon_ids, width, height, device=None):
    """Create a binary mask where points inside any of the polygons are marked as 1 and others as
    0.

    Args:
        vertices (torch.Tensor): Vertice tensor of shape [M, 2], where M is the total number of vertices, and 2 represents 2-D coordinates (x, y)
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

    # Determine the bounding box of the vertices
    min_x, _ = torch.min(vertices[:, 0], dim=0)
    max_x, _ = torch.max(vertices[:, 0], dim=0)
    min_y, _ = torch.min(vertices[:, 1], dim=0)
    max_y, _ = torch.max(vertices[:, 1], dim=0)

    # Create a grid representing points within the bounding box
    x = torch.arange(min_x, max_x + 1, dtype=torch.float32, device=device)
    y = torch.arange(min_y, max_y + 1, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    points = torch.stack([grid_x, grid_y], dim=-1)

    # Initialize the binary mask
    mask = torch.zeros((height, width), dtype=torch.bool, device=device)

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
            inside = ((v1[..., 0] < 0) & (v2[..., 0] >= 0) & (cross < 0)) | ((v1[..., 0] >= 0) & (v2[..., 0] < 0) & (cross > 0))

            # Increment the count for points inside the polygon or on the edge
            count += (inside | on_edge).int()

    # If the count is odd, the point is inside at least one polygon
    mask[min_y.int() : max_y.int() + 1, min_x.int() : max_x.int() + 1] = count % 2 == 1

    return mask


def create_binary_mask_from_vertices_with_padding(vertices, vertices_polygon_ids, width, height, device=None):
    """Create a binary mask where points inside any of the polygons are marked as 1 and others as
    0.

    Args:
        vertices (torch.Tensor): Vertice tensor of shape [M, 2], where M is the total number of vertices, and 2 represents 2-D coordinates (x, y)
        vertices_polygon_ids (torch.Tensor): Tensor of shape [M] containing polygon IDs for each vertex
        width: Width of the plane.
        height: Height of the plane.
        padding: Number of pixels to pad around the polygon bounding box (default: 1)
        device: Device to use for computations (default: None)
    Returns:
        A binary mask with shape (height, width), where points inside any of the polygons are marked as 1 and others as 0.
    """
    # Determine the device to use
    if device is None:
        device = vertices.device

    # Initialize the binary mask
    mask = torch.zeros((height, width), dtype=torch.bool, device=device)

    # Get the unique polygon IDs
    unique_ids = torch.unique(vertices_polygon_ids)
    padding = 2
    # Iterate over each polygon ID
    for idx in unique_ids:
        # Get the vertices corresponding to the current polygon ID
        polygon_vertices = vertices[vertices_polygon_ids == idx]

        # Determine the bounding box of the current polygon with padding
        min_x, _ = torch.min(polygon_vertices[:, 0], dim=0)
        max_x, _ = torch.max(polygon_vertices[:, 0], dim=0)
        min_y, _ = torch.min(polygon_vertices[:, 1], dim=0)
        max_y, _ = torch.max(polygon_vertices[:, 1], dim=0)

        # Add padding to the bounding box
        min_x -= padding
        max_x += padding
        min_y -= padding
        max_y += padding

        # Ensure the padded bounding box is within the image bounds
        # min_x = max(min_x.item(), 0)
        # max_x = min(max_x.item(), width - 1)
        # min_y = max(min_y.item(), 0)
        # max_y = min(max_y.item(), height - 1)

        # Create a grid representing points within the padded bounding box of the current polygon
        x = torch.arange(min_x, max_x + 1, dtype=torch.float32, device=device)
        y = torch.arange(min_y, max_y + 1, dtype=torch.float32, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        points = torch.stack([grid_x, grid_y], dim=-1)

        # Initialize the intersection counter for the current polygon
        count = torch.zeros_like(grid_x, dtype=torch.int32)

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
            inside = ((v1[..., 0] < 0) & (v2[..., 0] >= 0) & (cross < 0)) | ((v1[..., 0] >= 0) & (v2[..., 0] < 0) & (cross > 0))

            # Increment the count for points inside the polygon or on the edge
            count += (inside | on_edge).int()

        # If the count is odd, the point is inside the current polygon
        mask[min_y.int() : max_y.int() + 1, min_x.int() : max_x.int() + 1] |= count % 2 == 1

    return mask


def create_binary_mask_from_vertices_bk(vertices, vertices_polygon_ids, width, height, device=None):
    """Create a binary mask where points inside any of the polygons are marked as 1 and others as
    0.

    Args:
    vertices (torch.Tensor): Vertice tensor of shape [M, 2], where M is the total number of vertices,
    and 2 represents 2-D coordinates (x, y)
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

    # Get the unique polygon IDs
    unique_ids = torch.unique(vertices_polygon_ids)
    masks = []

    # Iterate over each polygon ID
    for idx in unique_ids:
        # Initialize the binary mask
        mask = torch.zeros_like(grid_x, dtype=torch.bool)
        # Initialize the intersection counter
        count = torch.zeros_like(grid_x, dtype=torch.int32)
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
            inside = ((v1[..., 0] < 0) & (v2[..., 0] >= 0) & (cross < 0)) | ((v1[..., 0] >= 0) & (v2[..., 0] < 0) & (cross > 0))

            # Increment the count for points inside the polygon or on the edge
            count += (inside | on_edge).int()

        # If the count is odd, the point is inside at least one polygon
        mask = count % 2 == 1
        masks.append(mask)
    mask = torch.stack(masks, dim=0).any(dim=0)
    return mask
