import torch
import cv2
import math

from tool.detection_utils import *
import slowfast.utils.checkpoint as cu
from slowfast.datasets import cv2_transform
from slowfast.models import build_model
from slowfast.visualization.utils import process_cv2_inputs

class SlowFast:
  def __init__(self, cfg, gpu_id=None):
    self.cfg = cfg
    
    if self.cfg.NUM_GPUS:
      self.gpu_id = (
        torch.cuda.current_device() if gpu_id is None else gpu_id
      )    
    self.model = build_model(self.cfg, self.gpu_id)
    self.model.eval()
    cu.load_test_checkpoint(self.cfg, self.model)
  
  def __call__(self, sequence):
    frames, detections = sequence.frames, sequence.detections
    person_detections = get_detections_by_types(detections, ("person"))
    middle_person_detection = get_middle_detections(person_detections)

    bboxes = get_detection_boxes(middle_person_detection)
    bboxes = torch.FloatTensor(bboxes).cuda()
    
    if bboxes is not None:
      frame_height, frame_width, _ = sequence.frame_shape()
      bboxes = cv2_transform.scale_boxes(
          self.cfg.DATA.TEST_CROP_SIZE,
          bboxes,
          frame_height,
          frame_width,
      )
    if self.cfg.DEMO.INPUT_FORMAT == "BGR":
      frames = [
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames
      ]

    frames = [
      cv2_transform.scale(self.cfg.DATA.TEST_CROP_SIZE, frame)
      for frame in frames
    ]
    inputs = process_cv2_inputs(frames, self.cfg)
    if bboxes is not None:
      index_pad = torch.full(
        size=(bboxes.shape[0], 1),
        fill_value=float(0),
        device=bboxes.device,
      )

    bboxes = torch.cat([index_pad, bboxes], axis=1)
    if self.cfg.NUM_GPUS > 0:
      if isinstance(inputs, (list,)):
        for i in range(len(inputs)):
          inputs[i] = inputs[i].cuda(
            device=torch.device(self.gpu_id), non_blocking=True
          )
      else:
        inputs = inputs.cuda(
          device=torch.device(self.gpu_id), non_blocking=True
        )
    if self.cfg.DETECTION.ENABLE and not bboxes.shape[0]:
      preds = torch.tensor([])
    else:
      preds = self.model(inputs, bboxes)

    if self.cfg.NUM_GPUS:
      preds = preds.cpu()
      if bboxes is not None:
        bboxes = bboxes.detach().cpu()

    preds = preds.detach()
    sequence.set_actions(preds)

    if isinstance(preds, torch.Tensor):
      if preds.ndim == 1:
        preds = preds.unsqueeze(0)
        n_instances = preds.shape[0]
    elif isinstance(preds, list):
      n_instances = len(preds)
    else:
      logger.error("Unsupported type of prediction input.")
      return
        
    scores, action_types = torch.topk(preds, k=1)
    scores, action_types = scores.tolist(), action_types.tolist()

    action_types_names = []
    for action_type in action_types:
      actn = [
        ACTION_TYPES[action_type_idx] 
        for action_type_idx in action_type
      ]
      action_types_names.append(actn)

    if bboxes is not None:
      frame_height, frame_width, _ = sequence.frame_shape()
      bboxes = cv2_transform.revert_scaled_boxes(
          self.cfg.DATA.TEST_CROP_SIZE,
          bboxes,
          frame_height,
          frame_width,
      )

    bboxes = [
      (
        int(math.ceil(x1)), 
        int(math.ceil(y1)), 
        int(math.ceil(x2)), 
        int(math.ceil(y2))
      )
      for zero, x1, y1, x2, y2 in bboxes.tolist() 
    ]

    sequence_detections = detections.copy()
    middle_key = get_middle_key(sequence_detections.keys())
    
    ids_and_actions = dict()
    for detection in sequence_detections[middle_key]:
      try:
        index = bboxes.index(tuple(detection[BOX]))
      except:
        continue
      ids_and_actions[detection[OBJECT_ID]] = action_types_names[index]

    sequence_action_detections = dict()    
    for key in sequence_detections.keys():
      
      action_detections = list()
      for detection in sequence_detections[key]:
        if detection[OBJECT_ID] in ids_and_actions.keys():
          detection[PERSON_ACTION] = ids_and_actions[detection[OBJECT_ID]]
        action_detections.append(detection)
    
      sequence_action_detections[key] = action_detections

    sequence.set_detections(sequence_action_detections)
    return sequence
