import cv2

from tool.detection_utils import simplify_direction, get_personal_area
from tool.annotation_utils import middle_point, point_in_box, is_inside_polygon

from tool.detection_utils import ID
from tool.detection_utils import TRAJECTORY
from tool.detection_utils import BOX
from tool.detection_utils import TYPE
from tool.detection_utils import ACTION
from tool.detection_utils import DIRECTION_VECTOR
from tool.detection_utils import POSITION_CHANGE

from tool.vizualize_utils import BGR_RED
from tool.vector_utils import vector_by_points, angle_between_vectors, sum_vectors

from fastdtw import fastdtw as fdtw
from scipy.spatial.distance import cosine

def action_inside_area(detection, area, action):
  detection_point = middle_point(*detection[BOX], detection[TYPE] == "person")
  if is_inside_polygon(area, detection_point):
    add_action(detection, action)
  else:
    del_action(detection, action)


def action_going_together(frame, dets, idx=0):
  act_det = dets[idx]
  act_det_dir = simplify_direction(act_det[DIRECTION_VECTOR])
  act_det_trj = act_det[TRAJECTORY]
  act_det_p_area = get_personal_area(*act_det[BOX])
  act_det_mid_p = middle_point(*act_det[BOX], act_det[TYPE]=='person')

  for jdx in range(idx+1, len(dets)):
    cmp_det = dets[jdx]

    if cmp_det == act_det:
      continue
    if cmp_det[TYPE] != act_det[TYPE]:
      continue

    cmp_det_mid_p = middle_point(*cmp_det[BOX], cmp_det[TYPE]=='person')
    if not point_in_box(cmp_det_mid_p, act_det_p_area):
      continue          

    A, B = act_det_mid_p, sum_vectors(act_det_mid_p, act_det[DIRECTION_VECTOR])
    C, D = cmp_det_mid_p, sum_vectors(cmp_det_mid_p, cmp_det[DIRECTION_VECTOR])
    AB = vector_by_points(A, B)
    CD = vector_by_points(C, D)
    if (AB[0] == 0 and AB[1] == 0) or (CD[0] == 0 and CD[1] == 0):
      continue

    cmp_det_dir = simplify_direction(
      cmp_det[DIRECTION_VECTOR])
    if (angle_between_vectors(AB, CD) > 0.30) and (act_det_dir != cmp_det_dir):
      continue

    speed_diff = abs(
      act_det[POSITION_CHANGE] - cmp_det[POSITION_CHANGE])
    if speed_diff > 0.7:
      continue
    
    cmp_det_trj = cmp_det[TRAJECTORY]
    dist, _ = fdtw(act_det_trj, cmp_det_trj, dist=cosine)
    if dist > 0.3:
      continue

    cv2.line(frame, act_det_mid_p, cmp_det_mid_p, BGR_RED, 2)
    add_action(act_det, str(cmp_det[ID]))
    add_action(cmp_det, str(act_det[ID]))
    

def add_action(detection, action):
  if ACTION in detection:
    try:
      detection[ACTION].index(action)
    except:
      detection[ACTION].append(action)
  else:
    detection[ACTION] = []
    detection[ACTION].append(action)


def del_action(detection, action):
  try:
    index = detection[ACTION].index(action)
    detection[ACTION].pop(index)
  except:
    pass

