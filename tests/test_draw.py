polygon = [
    [466.0, 896.0],
    [466.0, 960.0],
    [534.0, 636.0],
    [534.0, 896.0],
    [598.0, 636.0],
    [598.0, 960.0],
]


polygon = [
    [384.0, 449.0],
    [384.0, 470.0],
    [384.0, 492.0],
    [384.0, 532.0],
    [384.0, 554.0],
    [384.0, 575.0],
    [412.0, 449.0],
    [412.0, 575.0],
    [452.0, 449.0],
    [452.0, 575.0],
    [492.0, 449.0],
    [492.0, 575.0],
    [532.0, 449.0],
    [532.0, 575.0],
    [572.0, 449.0],
    [572.0, 575.0],
    [612.0, 449.0],
    [612.0, 575.0],
    [640.0, 449.0],
    [640.0, 470.0],
    [640.0, 492.0],
    [640.0, 532.0],
    [640.0, 554.0],
    [640.0, 575.0],
]

polygon = [
    [620.0, 870.0],
    [630.0, 900.0],
    [620.0, 920.0],
    [620.0, 960.0],
    [620.0, 1000.0],
    [620.0, 1040.0],
    [620.0, 1080.0],
    [620.0, 1120.0],
    [620.0, 1150.0],
    [620.0, 1170.0],
    [640.0, 1170.0],
    [680.0, 1180.0],
    [720.0, 1180.0],
    [760.0, 1180.0],
    [800.0, 1180.0],
    [840.0, 1180.0],
    [880.0, 1180.0],
    [920.0, 1180.0],
    [960.0, 1180.0],
    [1000.0, 1180.0],
    [1040.0, 1180.0],
    [1080.0, 1180.0],
    [1120.0, 1180.0],
    [1160.0, 1180.0],
    [1200.0, 1180.0],
    [1240.0, 1180.0],
    [1280.0, 1180.0],
    [1320.0, 1170.0],
    [1360.0, 1170.0],
    [1400.0, 1170.0],
    [1420.0, 1170.0],
    [1420.0, 1150.0],
    [1420.0, 1120.0],
    [1430.0, 1080.0],
    [1430.0, 1040.0],
    [1430.0, 1000.0],
    [1420.0, 960.0],
    [1420.0, 920.0],
    [1420.0, 900.0],
    [1420.0, 870.0],
    [1400.0, 880.0],
    [1360.0, 870.0],
    [1320.0, 870.0],
    [1280.0, 870.0],
    [1240.0, 870.0],
    [1200.0, 860.0],
    [1160.0, 860.0],
    [1120.0, 860.0],
    [1080.0, 860.0],
    [1040.0, 860.0],
    [1000.0, 860.0],
    [960.0, 860.0],
    [920.0, 860.0],
    [880.0, 860.0],
    [840.0, 860.0],
    [800.0, 870.0],
    [760.0, 870.0],
    [720.0, 870.0],
    [680.0, 880.0],
    [640.0, 870.0],
]
import math

import cv2
import numpy as np


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


# Create a blank 1000x1000 image
blank_image = np.zeros((1280, 1280, 3), dtype=np.uint8)

# Define the vertices of the polygon
# vertices = [(100, 100), (500, 100), (500, 500), (100, 500)]
vertices = polygon
sorted_vertices = sort_polygon_vertices(vertices)
print(sorted_vertices)

# Draw the filled polygon on the blank image
# result_image = draw_filled_polygon(blank_image, sorted_vertices)
result_image = cv2.fillPoly(
    blank_image, [np.array(sorted_vertices, dtype=np.int32)], (255, 255, 255)
)

# Display the result
cv2.imshow("Filled Polygon", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
