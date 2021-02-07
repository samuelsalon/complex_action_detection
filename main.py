from deepsort import generate_detections as gdet
from deepsort.detection import Detection
from deepsort.tracker import Tracker
from deepsort.nn_matching import NearestNeighborDistanceMetric
from tool.video_utils import VideoManager
from model.yolov4 import YOLOv4
from model.deepsort import DeepSort

import cv2
import argparse
import time

def parse_arguments():
  parser = argparse.ArgumentParser('Program used to detect and track objects in video')
  parser.add_argument('--video', '-v', default="/content/video.mp4", help="Input path to video")
  parser.add_argument('--output', '-o', default="/content/pred_video.mp4", help="Output path to video")
  parser.add_argument('--codec', '-c', default="mp4v", help="Codec for output video")
  parser.add_argument('--ann_file_output', '-afo', default="./data/annotation_file.txt", help="Output annotation file")
  parser.add_argument('--sequence_size', '-seq', default=8, type=int, help="Size of frames sequence for detection")
  parser.add_argument('--jump_until', '-ju', default=None, type=int, help="Jump until frame number divisible by this number")
  parser.add_argument('--jump_every', '-je', default=None, type=int, help="Jump every frame number divisible by this number")

  return parser.parse_args()

def main(args):
  video_name = args.video
  video_seq = args.sequence_size
  output_path = args.output
  model_filename = 'data/mars-small128.pb'
  nn_budget = None
  max_cosine_distance = 0.4

  jump_every = args.jump_every
  jump_until = args.jump_until

  if jump_until and jump_every:
    raise ValueError("--jump_until and --jump_every flags cannot be set at the same time")
  elif jump_until and jump_until < 2:
    raise ValueError("--jump_until value have to be bigger that 1")
  elif jump_every and jump_every < 2:
    raise ValueError("--jump_every value have to be bigger that 1")

  detection_model = YOLOv4('yolov4.weights', 'cfg/yolov4.cfg', jump_until, jump_every)
  tracker_model = DeepSort(model_filename, max_cosine_distance, nn_budget)
  
  names = detection_model.class_names

  video_manager = VideoManager(video_name, video_seq, output_path, args.codec)
  video_width = video_manager.video.width
  video_height = video_manager.video.height
  video_fps = video_manager.video.fps

  start_time = time.time()

  annotation_file = open(args.ann_file_output, mode="w")
  ret = True
  while ret:

    ret, sequence = video_manager.next()

    t0 = time.time()
    detected_sequence = detection_model(sequence)
    tracked_sequence = tracker_model(detected_sequence)
    
    for (idx, frame) in enumerate(tracked_sequence.frames):
      annotation_text = {}
      if idx not in tracked_sequence.detections.keys():
        video_manager.write(frame)
        continue
      result = tracked_sequence.detections[idx]

      ids = [id for id in result['ids']]
      boxes = [box for box in result['boxes']]
      scores = [score for score in result['scores']]
      class_names = [names[class_idx] for class_idx in result['classes']]

      for (id, box, score, class_name) in zip(ids, boxes, scores, class_names):
        x1, y1, x2, y2 = [int(i) for i in box]

        annotation_text['frame'] = tracked_sequence.frames_idx + idx
        annotation_text['object'] = {'id':id, 'type':class_name}
        annotation_text['bbox'] = (x1, y1, x2, y2)

        annotation_file.write(str(annotation_text) + "\n")

        id_and_class_name = str(id) + ". " + class_name
        cv2.putText(frame, id_and_class_name, (x1-10, y1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y1),(x2, y2), (255, 0, 0), 2)
      
      video_manager.write(frame)
    t1 = time.time()

    print("""FRAME: {}/{}  FPS: {:.2f}""".format(tracked_sequence.frames_idx, video_manager.video.frames_count, video_manager.clip_sequence_size / (t1 - t0)))

  video_manager.release()
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
      """.format(video_manager.video.name, 
               output_path, end_time - start_time, 
               video_manager.video.frames_count / (end_time - start_time), 
               video_manager.video.frames_count, video_width, video_height, video_fps))

if __name__ == "__main__":
  args = parse_arguments()
  main(args)
