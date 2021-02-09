from slowfast.config.defaults import get_cfg
from tool.video_utils import VideoManager
from tool.detection_utils import *
from tool.annotation_utils import *
from model.yolov4 import YOLOv4
from model.deepsort import DeepSort
from model.slowfast import SlowFast

import cv2
import argparse
import time

BGR_RED = (0, 0, 255)
BGR_GREEN = (0, 255, 0)
BGR_BLUE = (255, 0, 0)
BGR_WHITE = (255, 255, 255)
BGR_BLACK = (0, 0, 0)

def parse_arguments():
  parser = argparse.ArgumentParser('Program used to detect and track objects in video')
  parser.add_argument('--video', '-v', default="/content/video.mp4", help="Input path to video")
  parser.add_argument('--output', '-o', default="/content/pred_video.mp4", help="Output path to video")
  parser.add_argument('--codec', '-c', default="mp4v", help="Codec for output video")
  parser.add_argument('--ann_file_output', '-afo', default="./data/annotation_file.txt", help="Output annotation file")
  parser.add_argument('--jump_until', '-ju', default=None, type=int, help="Jump until frame number divisible by this number")
  parser.add_argument('--jump_every', '-je', default=None, type=int, help="Jump every frame number divisible by this number")

  return parser.parse_args()

def main(args):
  detection_weights = './checkpoints/yolov4.weights'
  detection_cfg = './cfg/yolov4.cfg'
  action_cfg = get_cfg()
  action_cfg.merge_from_file('./cfg/SLOWFAST_32x2_R101_50_50.yaml')

  video_name = args.video
  video_seq = action_cfg.DATA.NUM_FRAMES * action_cfg.DATA.SAMPLING_RATE
  output_path = args.output
  model_filename = './data/mars-small128.pb'
  nn_budget = None
  max_cosine_distance = 0.4

  video_manager = VideoManager(video_name, video_seq, output_path, args.codec)
  video_width = video_manager.video.width
  video_height = video_manager.video.height
  video_fps = video_manager.video.fps

  jump_every = args.jump_every
  jump_until = args.jump_until
  if jump_until and jump_every:
    raise ValueError("--jump_until and --jump_every flags cannot be set at the same time")
  elif jump_until and jump_until < 2:
    raise ValueError("--jump_until value have to be bigger that 1")
  elif jump_every and jump_every < 2:
    raise ValueError("--jump_every value have to be bigger that 1")
  elif not jump_every and not jump_until:
    jump_until = int(video_fps / 6)

  detection_model = YOLOv4(detection_weights, detection_cfg, jump_until, jump_every)
  tracker_model = DeepSort(model_filename, max_cosine_distance, nn_budget)
  action_model = SlowFast(action_cfg)

  names = detection_model.class_names


  start_time = time.time()

  annotation_file = open(args.ann_file_output, mode="w")
  ret = True
  while ret:

    ret, sequence = video_manager.next()

    t0 = time.time()
    detected_sequence = detection_model(sequence)
    tracked_sequence = tracker_model(detected_sequence)
    action_sequence = action_model(detected_sequence)
    
    for (idx, frame) in enumerate(action_sequence.frames):
      annotation_text = {}
      if idx in action_sequence.detections.keys():
        detections = action_sequence.detections[idx]

      for detection in detections:
        x1, y1, x2, y2 = detection[BOX]
        object_score = detection[OBJECT_SCORE]
        object_id = detection[OBJECT_ID]
        object_type = detection[OBJECT_TYPE]

        annotation_text[FRAME] = action_sequence.frames_idx + idx
        if PERSON_ACTION in detection:
          annotation_text[OBJECT] = { 
            TYPE : object_type, 
            ID : object_id,
            SCORE : object_score,
            ACTION : detection[PERSON_ACTION]
          }
        else:
          annotation_text[OBJECT] = { 
            TYPE : object_type, 
            ID : object_id,
            SCORE : object_score
          }
        annotation_text[BBOX] = (x1, y1, x2, y2)       
        annotation_file.write(str(annotation_text) + "\n")

        cv2.rectangle(frame, (x1, y1),(x2, y2), BGR_BLUE, 2)
        object_label = "Object ID: {}, Object type: {}".format(object_id, object_type)
        cv2.putText(frame, object_label, (x1, y1), 
          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, BGR_WHITE, 1)
        if PERSON_ACTION in detection:
          action_label = "Actions: {}".format(str(detection[PERSON_ACTION]))
          cv2.putText(frame, action_label, (x1, y1-20), 
            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, BGR_WHITE, 1)
      
      video_manager.write(frame)

    t1 = time.time()
    print("""FRAME: {}/{}  FPS: {:.2f}""".format(action_sequence.frames_idx, video_manager.video.frames_count, video_manager.clip_sequence_size / (t1 - t0)))


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
