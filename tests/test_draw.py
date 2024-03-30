import math
import os
import sys

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from test_data import edges, vertice_polygon_ids, vertices


def sort_polygon_vertices(vertices):
    # Calculate the centroid coordinates
    centroid_x = sum(v[0] for v in vertices) / len(vertices)
    centroid_y = sum(v[1] for v in vertices) / len(vertices)

    # Calculate the polar angle of each vertex relative to the centroid
    angles = []
    for v in vertices:
        dx = v[0] - centroid_x
        dy = v[1] - centroid_y
        angle = math.atan2(dy, dx)
        angles.append((angle, v))

    # Sort the vertices based on their polar angles
    sorted_vertices = sorted(
        angles, key=lambda x: (x[0], math.hypot(x[1][0] - centroid_x, x[1][1] - centroid_y))
    )

    # Extract the sorted vertex coordinates
    sorted_vertices = [v[1] for v in sorted_vertices]

    return sorted_vertices


def draw_filled_polygon(image, vertices):
    """Function to draw a filled polygon on a given image.

    Arguments:
    image -- A numpy array representing the image
    vertices -- A list of vertex coordinates in the format [(x1, y1), (x2, y2), ..., (xn, yn)]

    Returns:
    The modified image with the filled polygon
    """
    # Create a copy of the original image to avoid modifying it directly
    overlay = image.copy()

    # Convert the list of vertices to a numpy array
    vertices = np.array(vertices, dtype=np.int32)

    # Reshape the vertices array to have shape (num_vertices, 1, 2)
    vertices = vertices.reshape((-1, 1, 2))

    # Draw the filled polygon on the overlay image
    # cv2.fillPoly(overlay, [vertices], (255, 255, 255))
    cv2.drawContours(overlay, [vertices], -1, (255, 255, 255), -1)

    # Combine the original image with the overlay using bitwise OR operation
    result = cv2.bitwise_or(image, overlay)

    return result


def draw_image_by_vertices(vertices, vertice_polygon_ids):
    # Create a blank image
    blank_image = np.zeros((2048, 2048, 3), dtype=np.int32)

    # Define the vertices of the polygon
    # vertices = [(100, 100), (500, 100), (500, 500), (100, 500)]
    unique_ids = torch.unique(vertice_polygon_ids)

    for idx in unique_ids:
        polygon_vertices = vertices[vertice_polygon_ids == idx]
        if idx == 0:
            print(polygon_vertices)
            # Draw the filled polygon on the blank image
            # result_image = draw_filled_polygon(blank_image, sorted_vertices)
            blank_image = cv2.fillPoly(
                blank_image, [polygon_vertices.numpy().astype(int)], (255, 255, 255)
            )

    # Display the result
    # cv2.imshow("Filled Polygon", result_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    plt.imshow(blank_image)
    plt.show()


if __name__ == "__main__":
    vertices = torch.tensor(vertices)
    edges = torch.tensor(edges)
    vertice_polygon_ids = torch.tensor(vertice_polygon_ids)
    draw_image_by_vertices(vertices, vertice_polygon_ids)
