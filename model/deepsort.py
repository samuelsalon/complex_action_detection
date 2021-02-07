from deepsort.nn_matching import NearestNeighborDistanceMetric
from deepsort import generate_detections as gdet
from deepsort.tracker import Tracker
from deepsort.detection import Detection

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
      scores = [score for score in frame_detections['scores']]
      classes_idx = [class_idx for class_idx in frame_detections['classes']]
      bboxes_xywh = [
        (x1, y1, x2-x1, y2-y1) for (x1, y1, x2, y2) in frame_detections['boxes']
      ]

      features = self.feature_encoder(frame, bboxes_xywh)
      detections = [
        Detection(bbox, score, feature, class_name)
        for (bbox, score, class_name, feature) in zip(bboxes_xywh, scores, classes_idx, features)
      ]
      self.tracker.predict()
      self.tracker.update(detections)

      ids = []
      boxes = []
      scores = []
      classes_idx = []

      for track in self.tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
          continue 
        ids.append(track.track_id)
        boxes.append([int(i) for i in track.to_tlbr()])
        classes_idx.append(track.class_idx)
        scores.append(track.score)
      tracked_detections[idx] = {"ids":ids, "classes":classes_idx, "boxes":boxes, "scores":scores}
    
    sequence.set_detections(tracked_detections)
    return sequence
