import cv2

class Video(cv2.VideoCapture):
  
  def __init__(self, video_name):
    super().__init__(video_name)
    self.name = video_name
    self.width = int(self.get(cv2.CAP_PROP_FRAME_WIDTH))
    self.height = int(self.get(cv2.CAP_PROP_FRAME_HEIGHT))
    self.frames_count = int(self.get(cv2.CAP_PROP_FRAME_COUNT))
    self.fps = self.get(cv2.CAP_PROP_FPS)

class VideoManager:
  def __init__(self, video_path, clip_sequence_size, output_path, output_codec="mp4v"):
    self.video = Video(video_path)
    self.video_frame_counter = 0
    self.clip_sequence_size = clip_sequence_size
    fourcc = cv2.VideoWriter_fourcc(*output_codec)
    self.writer = cv2.VideoWriter(output_path, fourcc, self.video.fps,
                               (self.video.width, self.video.height))
  def next(self):
    sequence = Sequence(self.video_frame_counter, self.clip_sequence_size)

    frames = []
    ret = True
    while ret and len(frames) < self.clip_sequence_size:
      ret, frame = self.video.read()
      if not ret:
        break

      self.video_frame_counter += 1
      frames.append(frame)
    sequence.set_frames(frames)
    return ret, sequence

  def write(self, image):
    self.writer.write(image)

  def release(self):
    self.video.release()
    self.writer.release()
    

class Sequence:
  def __init__(self, frames_idx=0, sequence_size=8, frames=None):
    self.frames_idx = frames_idx
    self.frames = []
    self.sequence_size = sequence_size
    self.detections = {}

  def __iter__(self):
    return self

  def frame_shape(self):
    if len(self.frames) > 0:
      return self.frames[0].shape
    else:
      return (0, 0, 0)

  def set_frames(self, frames):
    self.frames = frames

  def set_detections(self, detections):
    self.detections = detections

  def set_actions(self, actions):
    self.actions = actions
