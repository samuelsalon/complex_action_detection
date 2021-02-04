from deepsort import generate_detections as gdet
from deepsort.detection import Detection
from deepsort.tracker import Tracker
from deepsort.nn_matching import NearestNeighborDistanceMetric
from tool.video import Video
from model.yolov4 import YOLOv4

import cv2
import argparse
import time


def parse_arguments():
  parser = argparse.ArgumentParser('Program used to detect and track objects in video')
  parser.add_argument('--video', '-v', default="/content/video.mp4", help="Input path to video")
  parser.add_argument('--output', '-o', default="/content/pred_video.mp4", help="Output path to video")
  parser.add_argument('--codec', '-c', default="mp4v", help="Codec for output video")
  parser.add_argument('--ann_file_output', '-afo', default="./data/annotation_file.txt", help="Output annotation file")

  return parser.parse_args()

def get_image_positions(box, img_height, img_width):
  x1 = int(box[0] * img_width)
  y1 = int(box[1] * img_height)
  x2 = int(box[2] * img_width)
  y2 = int(box[3] * img_height)
  return x1, y1, x2, y2

def main(args):
  video_name = args.video
  output_video_name = args.output
  model_filename = 'data/mars-small128.pb'
  nn_budget = None
  max_cosine_distance = 0.4

  encoder = gdet.create_box_encoder(model_filename, batch_size=1)
  metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
  tracker = Tracker(metric)

  model = YOLOv4('yolov4.weights', 'cfg/yolov4.cfg')
  names = model.class_names

  video = Video(video_name)

  fourcc = cv2.VideoWriter_fourcc(*args.codec)
  video_writer = cv2.VideoWriter(output_video_name, fourcc, video.fps,
                               (video.width, video.height))

  frame_counter = -1
  start_time = time.time()

  annotation_text = {}
  annotation_file = open(args.ann_file_output, mode="w")
  while True:
    t0 = time.time()

    ret, frame = video.read()
    if not ret:
      break

    frame_counter += 1

    result = model(frame)
    boxes = [get_image_positions(b, video.height, video.width) for b in result['boxes']]
    scores = [i for i in result['scores']]
    class_names = [names[i] for i in result['classes']]

    bboxes = []
    for box in boxes:
      x1, y1, x2, y2 = box
      bboxes.append((x1, y1, x2-x1, y2-y1))

    features = encoder(frame, bboxes)
    detections = []
    for (bbox, score, class_name, feature) in zip(bboxes, scores, class_names, features):
      detections.append(Detection(bbox, score, feature, class_name))

    tracker.predict()
    tracker.update(detections)

    for track in tracker.tracks:
      if not track.is_confirmed() or track.time_since_update > 1:
        continue 
    
      annotation_text['frame'] = frame_counter
      annotation_text['object'] = {'id':track.track_id, 'type':track.class_name}
      annotation_text['bbox'] = (x1, y1, x2, y2)
    
      annotation_file.write(str(annotation_text) + "\n")
      x1, y1, x2, y2 = [int(i) for i in track.to_tlbr()]
    
      id_and_class_name = str(track.track_id) + ". " + track.class_name
      cv2.putText(frame, id_and_class_name, (x1-10, y1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
      cv2.rectangle(frame, (x1, y1),(x2, y2), (255, 0, 0), 2)

    video_writer.write(frame)
    t1 = time.time()

    print("""FRAME: {}/{}  FPS: {:.2f}""".format(frame_counter, video.frames_count, 1.0 / (t1 - t0)))

  video.release()
  video_writer.release()
  annotation_file.close()

  end_time = time.time()
  print("""
      INPUT: {}
      OUTPUT: {}
      TOTAL Processing TIME: {:.1f}s
      FPS Processing: {:.2f}
      TOTAL FRAMES: {}
      VIDEO_SIZE: {}x{}
      VIDEO_FPS: {}
      """.format(video.name, 
               output_video_name, end_time - start_time, 
               frame_counter / (end_time - start_time), 
               video.frames_count, video.width, video.height, video.fps))

if __name__ == "__main__":
  args = parse_arguments()
  main(args)

