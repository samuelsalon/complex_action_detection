import cv2
from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet

class YOLOv4:

  def __init__(self, weight_file, cfg_file, jump_until=None, jump_every=None):
    self.use_cuda = True if torch.cuda.is_available() else False
    self.model = self._load_model(weight_file, cfg_file, self.use_cuda)
    self.class_names = load_class_names('data/coco.names')
    self.jump_until = jump_until
    self.jump_every = jump_every

  def __call__(self, sequence, conf_thres=0.4, nms_thres=0.6):
    detections = dict()
    for (idx, frame) in enumerate(sequence.frames):
      frame_number = sequence.frames_idx + idx
      if self.jump_every and frame_number % self.jump_every == 0:
        continue
      elif self.jump_until and frame_number % self.jump_until != 0:
        continue

      frame_height, frame_width = frame.shape[:2]
      sized = cv2.resize(frame, (self.model.width, self.model.height))
      sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
    
      result = do_detect(self.model, sized, conf_thres, nms_thres, self.use_cuda)[0]
      dets = dict()
      dets['boxes'] = [get_image_positions((det[0], det[1], det[2], det[3]), frame_height, frame_width) for det in result]
      dets['classes'] = [det[-1] for det in result]
      dets['scores'] = [det[-2] for det in result]
      detections[idx] = dets
    sequence.set_detections(detections)
    return sequence

  def _load_model(self, weight_file, cfg_file, use_cuda):
    model = Darknet(cfg_file)
    model.load_weights(weight_file)   
    if use_cuda:
      model.cuda()
    return model

