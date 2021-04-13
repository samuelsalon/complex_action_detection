from deepsort.nn_matching import NearestNeighborDistanceMetric
from deepsort import generate_detections as gdet
from deepsort.tracker import Tracker
from deepsort.detection import Detection
from deepsort.preprocessing import non_max_suppression

import numpy as np

from tool.detection_utils import get_detection_scores
from tool.detection_utils import get_detection_types
from tool.detection_utils import get_detection_boxes
from tool.detection_utils import ID
from tool.detection_utils import TRAJECTORY
from tool.detection_utils import BOX
from tool.detection_utils import TYPE
from tool.detection_utils import DETECTION_SCORE
from tool.detection_utils import DIRECTION_VECTOR
from tool.detection_utils import POSITION_CHANGE

class DeepSort:
  def __init__(self, encoder_model_filename, max_cosine_distance, nn_budget):
    self.feature_encoder = gdet.create_box_encoder(encoder_model_filename, batch_size=1)
    metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    self.tracker = Tracker(metric)

  def __call__(self, sequence):
    tracked_detections = {}
    for (idx, frame) in enumerate(sequence.frames):

      if idx not in sequence.detections.keys():
        continue
      
      frame_detections = sequence.detections[idx]
      scores = get_detection_scores(frame_detections)
      object_types = get_detection_types(frame_detections)
      bboxes_xywh = [
        (x1, y1, x2-x1, y2-y1) 
        for (x1, y1, x2, y2) in get_detection_boxes(frame_detections)
      ]

      features = self.feature_encoder(frame, bboxes_xywh)
      detections = [
        Detection(bbox, score, feature, class_name)
        for (bbox, score, class_name, feature) in zip(bboxes_xywh, scores, object_types, features)
      ]

      boxs = np.array([d.tlwh for d in detections])
      scores = np.array([d.confidence for d in detections])
      indices = non_max_suppression(boxs, 1.0, scores)
      detections = [detections[i] for i in indices]

      self.tracker.predict()
      self.tracker.update(detections)

      frame_detections = list()
      for track in self.tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
          continue 

        detection = dict()
        detection[ID] = track.track_id
        detection[TRAJECTORY] = track.trajectory
        detection[BOX] = ([int(i) for i in track.to_tlbr()])
        detection[TYPE] = track.class_idx
        detection[DETECTION_SCORE] = track.score
        detection[DIRECTION_VECTOR] = track.direction_vector
        detection[POSITION_CHANGE] = track.position_change

        frame_detections.append(detection)

      tracked_detections[idx] = frame_detections
    
    sequence.set_detections(tracked_detections)
    return sequence
