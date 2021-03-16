import math
from tool.annotation_utils import middle_point

def sum_vectors(vector1, vector2):
  return [p1+p2 for (p1, p2) in zip(vector1, vector2)]

def vector_by_points(vector1, vector2):
  return [p2-p1 for (p1, p2) in zip(vector1, vector2)]

def dot_product(vector1, vector2):
  return sum((p1*p2) for (p1, p2) in zip(vector1, vector2))

def magnitude_of_vector(vector):
  x, y = vector
  return math.sqrt(x*x + y*y)

def angle_between_vectors(vector1, vector2):
  product_of_magnitudes = magnitude_of_vector(vector1) * magnitude_of_vector(vector2)
  angle = dot_product(vector1, vector2) / product_of_magnitudes
  if angle > 1.0:
    angle -= 0.000001
  elif angle < -1.0:
    angle += 0.000001
  return math.acos(angle)

def min_bounding_box(polygon_points):
  if len(polygon_points) < 0:
    return -1
  min_x = polygon_points[0][0]
  min_y = polygon_points[0][1]
  max_x = polygon_points[0][0]
  max_y = polygon_points[0][1]
  for (x, y) in polygon_points:
    if x < min_x:
      min_x = x
    if x > max_x:
      max_x = x
    if y < min_y:
      min_y = y
    if y > max_x:
      max_y = y
  return min_x, min_y, max_x, max_y

def order_polygon_points(polygon_points):
  x1, y1, x2, y2 = min_bounding_box(polygon_points)
  mid_x, mid_y = middle_point(x1, y1, x2, y2)
  angles_points = {}
  for point in polygon_points:
    angle = math.atan2(point[1] - mid_y, point[0] - mid_x)
    angles_points[angle] = point
  sorted_keys = sorted(angles_points.keys())
  return [angles_points[key] for key in sorted_keys]
