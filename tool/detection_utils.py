from tool.utils import load_class_names
from slowfast.utils.misc import get_class_names

ACTION_TYPES, _, _ = get_class_names('./data/label_map_ava.json', None, None)
ALL_TYPES = load_class_names('./data/coco.names')

FRAME = 'frame'
DETECTIONS = 'detections'

ID = 'id'
BOX = 'box'
DETECTION_SCORE = 'detection_score'
DIRECTION_VECTOR = 'direction_vector'
POSITION_CHANGE = 'position_change'
TYPE = 'type'
ACTION = 'person_action'
ACTION_SCORE = 'action_score'

def get_detection_ids(detection):
  boxes = [one[ID] for one in detection]
  return boxes

def get_detection_boxes(detection):
  boxes = [one[BOX] for one in detection]
  return boxes

def get_detection_scores(detection):
  scores = [one[DETECTION_SCORE] for one in detection]
  return scores

def get_detection_types(detection):
  types = [one[TYPE] for one in detection]
  return types

def get_detections_by_types(sequence_detections, types):
  
  detections = sequence_detections.copy()
  for key in detections.keys():
    
    types_detections = []
    frame_detections = detections[key]

    for detection in frame_detections:
      if detection[TYPE] in types:
        types_detections.append(detection)
    
    detections[key] = types_detections

  return detections


def get_middle_detections(detections):
  middle_key = get_middle_key(detections.keys())
  if middle_key < 0:
    return ()
  else:
    return detections[middle_key]


def get_middle_key(keys):
  keys = list(keys)
  if len(keys) == 0:
    return -1
  middle_key_idx = len(keys) // 2
  return keys[middle_key_idx]


def get_middle_detections_boxes(detections):
  middle_detection = get_middle_detections(detections)
  return middle_detection[BOXES]
