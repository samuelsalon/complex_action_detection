import cv2
from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
from tool.detection_utils import *

class YOLOv4:

  def __init__(self, weight_file, cfg_file, jump_until=None, jump_every=None):
    self.use_cuda = True if torch.cuda.is_available() else False
    self.model = self._load_model(weight_file, cfg_file, self.use_cuda)
    self.class_names = load_class_names('data/coco.names')
    self.jump_until = jump_until
    self.jump_every = jump_every

  def __call__(self, sequence, conf_thres=0.4, nms_thres=0.6):
    sequence_detections = dict()
    for (idx, frame) in enumerate(sequence.frames):
      frame_number = sequence.frames_idx + idx
      if self.jump_every and frame_number % self.jump_every == 0:
        continue
      elif self.jump_until and frame_number % self.jump_until != 0:
        continue

      sized = cv2.resize(frame, (self.model.width, self.model.height))
      sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
    
      result = do_detect(self.model, sized, conf_thres, nms_thres, self.use_cuda)[0]
      
      frame_detections_struct = list()
      frame_height, frame_width = frame.shape[:2]
      for detection in result:
        detection_struct = dict()
        x1, y1, x2, y2 = detection[:4]
        detection_struct[BOX] = get_image_positions(
          (x1, y1, x2, y2), frame_height, frame_width)
        detection_struct[OBJECT_TYPE] = self.class_names[detection[-1]]
        detection_struct[OBJECT_SCORE] = detection[-2]
        frame_detections_struct.append(detection_struct)

      sequence_detections[idx] = frame_detections_struct

    sequence.set_detections(sequence_detections)
    return sequence

  def _load_model(self, weight_file, cfg_file, use_cuda):
    model = Darknet(cfg_file)
    model.load_weights(weight_file)   
    if use_cuda:
      model.cuda()
    return model

