from deepsort import generate_detections as gdet
from deepsort.detection import Detection
from deepsort.tracker import Tracker
from deepsort.nn_matching import NearestNeighborDistanceMetric
from model.yolov4 import YOLOv4

import cv2
import time

def get_image_positions(box, img_height, img_width):
  x1 = int(box[0] * img_width)
  y1 = int(box[1] * img_height)
  x2 = int(box[2] * img_width)
  y2 = int(box[3] * img_height)
  return x1, y1, x2, y2

video_name = '/content/video.mp4'
output_video_name = '/content/pred_video.mp4'
model_filename = 'data/mars-small128.pb'
nn_budget = None
max_cosine_distance = 0.4

encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

model = YOLOv4('yolov4.weights', 'cfg/yolov4.cfg')
names = model.class_names

video = cv2.VideoCapture(video_name)
video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
video_fps = video.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video_name, fourcc, video_fps,
                               (video_width, video_height))

frame_counter = 0
start_time = time.time()
while True:
  t0 = time.time()

  ret, frame = video.read()
  if not ret:
    break

  frame_counter += 1

  result = model(frame)
  boxes = [get_image_positions(b, video_height, video_width) for b in result['boxes']]
  scores = [i for i in result['scores']]
  class_names = [names[i] for i in result['classes']]

  bboxes = []
  for box in boxes:
    x1, y1, x2, y2 = box
    bboxes.append((x1, y1, x2-x1, y2-y1))

  features = encoder(frame, bboxes)
  detections = []
  for (bbox, score, class_name, feature) in zip(bboxes, scores, class_names, features):
    detections.append(Detection(bbox, score, feature))

  tracker.predict()
  tracker.update(detections)

  for track in tracker.tracks:
    if not track.is_confirmed() or track.time_since_update > 1:
      continue 
    x1, y1, x2, y2 = [int(i) for i in track.to_tlbr()]
    cv2.rectangle(frame, (x1, y1),(x2, y2), (255, 0, 0), 2)

  video_writer.write(frame)
  t1 = time.time()

  print("""
    FRAME: {}/{}  
    FPS: {:.2f}""".format(frame_counter, video_frames, 1.0 / (t1 - t0)))

end_time = time.time()

print("""
    INPUT: {}
    OUTPUT: {}
    TOTAL Processing TIME: {:.1f}s
    FPS Processing: {:.2f}
    TOTAL FRAMES: {}
    VIDEO_SIZE: {}x{}
    VIDEO_FPS: {}
    """.format(input_name, output_video_name, end_time - start_time, 1.0 / (end_time - start_time), video_frames, video_width, video_height, video_fps))

video.release()
video_writer.release()



