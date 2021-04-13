import cv2

from tool.detection_utils import get_personal_area
from tool.annotation_utils import middle_point

from tool.detection_utils import ID
from tool.detection_utils import BOX
from tool.detection_utils import TYPE
from tool.detection_utils import ACTION
from tool.detection_utils import DETECTION_SCORE
from tool.detection_utils import DIRECTION_VECTOR
from tool.detection_utils import POSITION_CHANGE


BGR_RED = (0, 0, 255)
BGR_GREEN = (0, 255, 0)
BGR_BLUE = (255, 0, 0)
BGR_WHITE = (255, 255, 255)
BGR_BLACK = (0, 0, 0)
RGB_PURPLE = (216,191,216)


def draw_personal_area(frame, x1, y1, x2, y2):
  p_x1, p_y1, p_x2, p_y2 = get_personal_area(x1, y1, x2, y2)
  cv2.rectangle(frame, (p_x1,p_y1), (p_x2, p_y2), BGR_GREEN, 1)


def draw_color_recognition_position(frame, position):
  x1, y1, x2, y2 = position
  cv2.rectangle(frame, (x1,y1), (x2,y2), RGB_PURPLE, 2)


def draw_polygon(frame, polygon):
  for idx in range(len(polygon)):
    cv2.line(frame, polygon[idx-1], polygon[idx], RGB_PURPLE, 2)


def vizualize_detection(frame, detection):
  x1, y1, x2, y2 = detection[BOX]

  score = detection[DETECTION_SCORE]
  id = detection[ID]
  type = detection[TYPE]
  direction_vector = detection[DIRECTION_VECTOR]
  position_change = detection[POSITION_CHANGE]

  # draw bounding box
  cv2.rectangle(frame, (x1, y1),(x2, y2), BGR_BLUE, 2)
  # draw object info label
  object_label = "ID: {}, Type: {}".format(id, type)
  cv2.putText(frame, object_label, (x1, y1-10), 
    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, BGR_WHITE, 1)
  # draw action info label
  #if ACTION in detection:
  #  action_label = "Actions: {}".format(str(detection[ACTION]))
  #  cv2.putText(frame, action_label, (x1, y1-30), 
  #    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, BGR_WHITE, 1)

  # draw direction      
  x_dir, y_dir = direction_vector
  detection_point = middle_point(x1, y1, x2, y2, type=="person")
  direction_point = x_dir + detection_point[0], y_dir + detection_point[1]
  
  cv2.line(frame, detection_point, direction_point, BGR_GREEN, 2)
  cv2.circle(frame, direction_point, 5, BGR_RED, -1)

  position_change_label = "{:.1f}".format(position_change)
  cv2.putText(frame, str(position_change_label), detection_point, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, BGR_WHITE, 1)

