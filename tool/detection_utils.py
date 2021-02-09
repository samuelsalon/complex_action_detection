from tool.utils import load_class_names
from slowfast.utils.misc import get_class_names

ACTION_TYPES, _, _ = get_class_names('./data/label_map_ava.json', None, None)
ALL_TYPES = load_class_names('./data/coco.names')
IDS = 'ids'
BOXES = 'boxes'
SCORES = 'scores'
TYPES = 'types'
ACTIONS = 'actions'

OBJECT_ID = 'object_id'
BOX = 'box'
OBJECT_SCORE = 'object_score'
OBJECT_TYPE = 'object_type'
PERSON_ACTION = 'person_action'

def get_detection_ids(detection):
  boxes = [one[ID] for one in detection]
  return boxes

def get_detection_boxes(detection):
  boxes = [one[BOX] for one in detection]
  return boxes

def get_detection_scores(detection):
  scores = [one[OBJECT_SCORE] for one in detection]
  return scores

def get_detection_types(detection):
  types = [one[OBJECT_TYPE] for one in detection]
  return types

def get_detections_by_types(sequence_detections, types):
  
  detections = sequence_detections.copy()
  for key in detections.keys():
    
    types_detections = []
    frame_detections = detections[key]

    for detection in frame_detections:
      if detection[OBJECT_TYPE] in types:
        types_detections.append(detection)
    
    detections[key] = types_detections

  return detections


def get_middle_detections(detections):
  middle_key = get_middle_key(detections.keys())
  return detections[middle_key]


def get_middle_key(keys):
  keys = list(keys)
  middle_key_idx = len(keys) // 2
  return keys[middle_key_idx]


def get_middle_detections_boxes(detections):
  middle_detection = get_middle_detections(detections)
  return middle_detection[BOXES]
