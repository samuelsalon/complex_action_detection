from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet

class YOLOv4:

  def __init__(self, weight_file, cfg_file):
    self.use_cuda = True if torch.cuda.is_available() else False
    self.model = self._load_model(weight_file, cfg_file, self.use_cuda)
    self.class_names = load_class_names('data/coco.names')

  def __call__(self, img, conf_thres=0.4, nms_thres=0.6):
    sized = cv2.resize(img, (self.model.width, self.model.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
    
    result = do_detect(self.model, sized, conf_thres, nms_thres, self.use_cuda)[0]
    dets = dict()
    dets['boxes'] = [(det[:4]) for det in result]
    dets['classes'] = [det[-1] for det in result]
    dets['scores'] = [det[-2] for det in result]

    return dets

  def _load_model(self, weight_file, cfg_file, use_cuda):
    model = Darknet(cfg_file)
    model.load_weights(weight_file)   
    if use_cuda:
      model.cuda()
    return model

