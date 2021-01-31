import cv2

class Video(cv2.VideoCapture):
  
  def __init__(self, video_name):
    super().__init__(video_name)
    self.name = video_name
    self.width = int(self.get(cv2.CAP_PROP_FRAME_WIDTH))
    self.height = int(self.get(cv2.CAP_PROP_FRAME_HEIGHT))
    self.frames_count = int(self.get(cv2.CAP_PROP_FRAME_COUNT))
    self.fps = self.get(cv2.CAP_PROP_FPS)
