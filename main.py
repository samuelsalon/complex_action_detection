from slowfast.config.defaults import get_cfg
from tool.video_utils import VideoManager
from model.yolov4 import YOLOv4
from model.deepsort import DeepSort
from model.slowfast import SlowFast

from tool.annotation_utils import middle_point, is_inside_polygon, point_in_box

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

from fastdtw import fastdtw as fdtw
from scipy.spatial.distance import cosine

DATECTIONS_PER_SECOND = 6

BGR_RED = (0, 0, 255)
BGR_GREEN = (0, 255, 0)
BGR_BLUE = (255, 0, 0)
BGR_WHITE = (255, 255, 255)
BGR_BLACK = (0, 0, 0)
RGB_PURPLE = (216,191,216)

# crossroad.mp4 semafor
x1_semafor, y1_semafor, x2_semafor, y2_semafor = 780, 40, 810, 140
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
BETWEEN_CAR_POLYGON = (
  (390, 150), 
  (550, 150),
  (730, 180), 
  (620, 420), 
  (180, 360)
)


def personal_area(x1, y1, x2, y2):
  half_w = (x2 - x1)
  half_h = int((y2 - y1) / 2)
  x1_area = x1 - half_w
  y1_area = y1 - half_h
  x2_area = x2 + half_w
  y2_area = y2 + half_h
  return x1_area, y1_area, x2_area, y2_area


def draw_personal_area(frame, x1, y1, x2, y2):
  p_x1, p_y1, p_x2, p_y2 = personal_area(x1, y1, x2, y2)
  cv2.rectangle(frame, (p_x1,p_y1), (p_x2, p_y2), BGR_GREEN, 1)


def going_together(frame, act_det, dets, start=0):
  act_det_dir = simplify_direction(act_det[DIRECTION_VECTOR])
  act_det_trj = act_det[TRAJECTORY]
  act_det_p_area = personal_area(*act_det[BOX])
  act_det_mid_p = middle_point(*act_det[BOX], act_det[TYPE]=='person')

  for jdx in range(start, len(dets)):
    cmp_det = dets[jdx]

    if cmp_det == act_det:
      continue
    if cmp_det[TYPE] != act_det[TYPE]:
      continue
          
    cmp_det_dir = simplify_direction(
      cmp_det[DIRECTION_VECTOR])
    if act_det_dir != cmp_det_dir:
      continue

    speed_diff = abs(
      act_det[POSITION_CHANGE] - cmp_det[POSITION_CHANGE])
    if speed_diff > 1.5:
      continue

    cmp_det_mid_p = middle_point(*cmp_det[BOX], cmp_det[TYPE]=='person')
    if not point_in_box(cmp_det_mid_p, act_det_p_area):
      continue
    
    cmp_det_trj = cmp_det[TRAJECTORY]
    dist, _ = fdtw(act_det_trj, cmp_det_trj, dist=cosine)
    if dist > 0.3:
      continue

    cv2.line(frame, act_det_mid_p, cmp_det_mid_p, BGR_RED, 2)
    add_action(act_det, str(cmp_det[ID]))
    add_action(cmp_det, str(act_det[ID]))
    
 
def draw_color_recognition_position(frame, position):
  x1, y1, x2, y2 = position
  cv2.rectangle(frame, (x1,y1), (x2,y2), RGB_PURPLE, 2)


def draw_polygon(frame, polygon):
  for idx in range(len(polygon)):
    cv2.line(frame, polygon[idx-1], polygon[idx], RGB_PURPLE, 2)


def vizualize_detection(frame, detection):
  x1, y1, x2, y2 = detection[BOX]

  score = detection[DETECTION_SCORE]
  id = detection[ID]
  type = detection[TYPE]
  direction_vector = detection[DIRECTION_VECTOR]
  position_change = detection[POSITION_CHANGE]

  # draw bounding box
  cv2.rectangle(frame, (x1, y1),(x2, y2), BGR_BLUE, 2)
  # draw object info label
  object_label = "ID: {}, Type: {}".format(id, type)
  cv2.putText(frame, object_label, (x1, y1-10), 
    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, BGR_WHITE, 1)
  # draw action info label
  if ACTION in detection:
    action_label = "Actions: {}".format(str(detection[ACTION]))
    cv2.putText(frame, action_label, (x1, y1-30), 
      cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, BGR_WHITE, 1)

  # draw direction      
  x_dir, y_dir = direction_vector
  detection_point = middle_point(x1, y1, x2, y2, type=="person")
  direction_point = x_dir + detection_point[0], y_dir + detection_point[1]
  
  cv2.line(frame, detection_point, direction_point, BGR_GREEN, 2)
  cv2.circle(frame, direction_point, 5, BGR_RED, -1)

  position_change_label = "{:.4f}".format(position_change)
  cv2.putText(frame, str(position_change_label), detection_point, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, BGR_WHITE, 1)


def parse_arguments():
  parser = argparse.ArgumentParser('Program used to detect and track objects in video')
  parser.add_argument('--video', '-v', default="/content/video.mp4", help="Input path to video")
  parser.add_argument('--output', '-o', default="/content/pred_video.mp4", help="Output path to video")
  parser.add_argument('--codec', '-c', default="mp4v", help="Codec for output video")
  parser.add_argument('--ann_file_output', '-afo', default="./data/annotation_file.txt", help="Output annotation file")
  parser.add_argument('--jump_until', '-ju', default=None, type=int, help="Jump until frame number divisible by this number")
  parser.add_argument('--jump_every', '-je', default=None, type=int, help="Jump every frame number divisible by this number")
  return parser.parse_args()


def add_action(detection, action):
  if ACTION in detection:
    try:
      detection[ACTION].index(action)
    except:
      detection[ACTION].append(action)
  else:
    detection[ACTION] = []
    detection[ACTION].append(action)


def del_action(detection, action):
  try:
    index = detection[ACTION].index(action)
    detection[ACTION].pop(index)
  except:
    pass


def simplify_direction(direction_vector):
  x, y = direction_vector
  if x > 0:
    x = 1
  elif x < 0:
    x = -1
  if y > 0:
    y = 1
  elif y < 0:
    y = -1
  return x, y


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
    #action_sequence = action_model(detected_sequence)
  
    action_sequence = tracked_sequence

    write_annotation = True
    for (sequence_frame_idx, frame) in enumerate(action_sequence.frames):
      if sequence_frame_idx in action_sequence.detections.keys():
        write_annotation = True
        detections = action_sequence.detections[sequence_frame_idx]
      else:
        write_annotation = False
      
      draw_polygon(frame, BETWEEN_CAR_POLYGON)

      for (idx, detection) in enumerate(detections):

        action = "in polygon"
        detection_point = middle_point(*detection[BOX], detection[TYPE] == "person")
        if is_inside_polygon(BETWEEN_CAR_POLYGON, detection_point):
          add_action(detection, action)
        else:
          del_action(detection, action)
        
        vizualize_detection(frame, detection)
        going_together(frame, detection, detections, idx+1)
      
      draw_color_recognition_position(frame, 
                      (x1_semafor,y1_semafor,x2_semafor,y2_semafor))
      video_manager.write(frame)

      if write_annotation and len(detections) > 0:
        frame_number = action_sequence.frames_idx + sequence_frame_idx
        frame_detections_annotation = {
          FRAME : frame_number,
          DETECTIONS : detections
        }
        annotation_file.write(str(frame_detections_annotation) + "\n")

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
