import math

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
