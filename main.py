from slowfast.config.defaults import get_cfg
from tool.video_utils import VideoManager
from model.yolov4 import YOLOv4
from model.deepsort import DeepSort
from model.slowfast import SlowFast

from tool.action_utils import add_action, del_action
from tool.action_utils import action_going_together, action_inside_area

from tool.color_recognition_utils import colored_area, crop_image

from tool.vector_utils import order_polygon_points

from tool.vizualize_utils import draw_polygon
from tool.vizualize_utils import draw_color_recognition_position
from tool.vizualize_utils import vizualize_detection
from tool.annotation_utils import middle_point, is_inside_polygon, point_in_box
from tool.detection_utils import get_personal_area

from tool.detection_utils import ID
from tool.detection_utils import TRAJECTORY
from tool.detection_utils import BOX
from tool.detection_utils import TYPE
from tool.detection_utils import ACTION
from tool.detection_utils import DETECTION_SCORE
from tool.detection_utils import DIRECTION_VECTOR
from tool.detection_utils import POSITION_CHANGE
from tool.detection_utils import DETECTIONS
from tool.detection_utils import FRAME

import cv2
import argparse
import time

DATECTIONS_PER_SECOND = 6

# crossroad.mp4 semafor
x1_s, y1_s, x2_s, y2_s = 780, 40, 810, 140
ZEBRA_ACTION = "on zebra"
ZEBRA_POLYGON = (
  (575,500), 
  (1070,500), 
  (1045,650), 
  (465,650)
)

POLYGON_VIDEO = (
  (700, 150), 
  (900, 150), 
  (950, 900), 
  (800, 880), 
  (750, 800)
)

BETWEEN_CAR_ACTION = "between cars"
BETWEEN_CAR_POLYGON = (
  (390, 150), 
  (550, 150),
  (730, 180), 
  (620, 420), 
  (180, 360)
)

BETWEEN_CAR_POLYGON_1 = (
  (370, 150),
  (450, 300),
  (740, 130),
  (690, 280),
  (800, 350),
  (800, 600),
  (100, 400)
)

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
    jump_until = int(video_fps / DATECTIONS_PER_SECOND)

  annotation_file = open(args.ann_file_output, mode="w")
  
  detection_model = YOLOv4(detection_weights, detection_cfg, jump_until, jump_every)
  tracker_model = DeepSort(model_filename, max_cosine_distance, nn_budget)
  action_model = SlowFast(action_cfg)

  start_time = time.time()
  ret = True
  while ret:

    ret, frames_sequence = video_manager.next()

    t0 = time.time()
    detected_sequence = detection_model(frames_sequence)
    tracked_sequence = tracker_model(detected_sequence)
    action_sequence = action_model(tracked_sequence)
  
    write_annotation = True
    for (sequence_frame_idx, frame) in enumerate(action_sequence.frames):
      if sequence_frame_idx in action_sequence.detections.keys():
        detections = action_sequence.detections[sequence_frame_idx]
        write_annotation = True
      else:
        write_annotation = False
      
      color_recognition_area = crop_image(frame, (x1_s, y1_s), (x2_s, y2_s))
      if colored_area(color_recognition_area, (55, 155, 225)) > 100:
        cv2.putText(frame, "True", (x1_s, y1_s-30), 
          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (55, 155, 225), 1)

      for idx in range(len(detections)):
        action_going_together(frame, detections, idx)

      for (idx, detection) in enumerate(detections):
        action_inside_area(detection, BETWEEN_CAR_POLYGON_1, BETWEEN_CAR_ACTION)
        vizualize_detection(frame, detection)
      
      draw_polygon(frame, BETWEEN_CAR_POLYGON_1)
      draw_color_recognition_position(frame, (x1_s, y1_s, x2_s, y2_s))
      video_manager.write(frame)

      if write_annotation and len(detections) > 0:
        frame_number = action_sequence.frames_idx + sequence_frame_idx
        frame_detections_annotation = {
          FRAME : frame_number,
          DETECTIONS : detections
        }
        annotation_file.write(str(frame_detections_annotation) + "\n")

    t1 = time.time()
    print("""FRAME: {}/{}  FPS: {:.2f}""".format(
      action_sequence.frames_idx, 
      video_manager.video.frames_count, 
      video_manager.clip_sequence_size / (t1 - t0)))

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
