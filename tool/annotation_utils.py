import ast
import cv2
from tool.video_utils import Video
from tool.detection_utils import FRAME
from tool.detection_utils import DETECTIONS
from tool.detection_utils import BOX
from tool.detection_utils import DETECTION_SCORE
from tool.detection_utils import DIRECTION_VECTOR
from tool.detection_utils import ID
from tool.detection_utils import ACTION
from tool.detection_utils import TYPE
import numpy as np

INT_MAX = 10000

def parse_annotations(annotation_file):
  annotations = []
  with open(annotation_file, mode='r') as ann_file:
    annotation_list = ann_file.read().split("\n")
    for annotation in annotation_list:
      if len(annotation) < 1:
        continue
      annotations.append(ast.literal_eval(annotation))
  return annotations


def middle_point(x1, y1, x2, y2, is_person=False):
  w, h = int(x2-x1), int(y2-y1)
  if is_person:
    return int(x1 + (w/2)), int(y1 + h)
  else:
    return int(x1 + (w/2)), int(y1 + (h/2))


def get_annotation_by_object_id(anns, id):
  new_anns = []
  for ann in anns:
    detections = []
    for detection in ann[DETECTIONS]:
      if detection[ID] == id:
        detections.append(detection)
        break
    if len(detections) == 0:
      continue
    new_ann = {
      FRAME : ann[FRAME],
      DETECTIONS : detections
    }
    new_anns.append(new_ann)
  return new_anns


def image_cut(img, x1, y1, x2, y2):
  return img[y1:y2, x1:x2]


def point_in_box(point, box):
  if len(box) != 4 or len(point) != 2:
    return False

  box_x1, box_y1, box_x2, box_y2 = box
  x, y = point

  if box_x1 <= x <= box_x2 and box_y1 <= y <= box_y2:
    return True
  else:
    return False


def whole_in_box(it, box):
  if len(box) != 4 or len(it) != 4:
    raise ValueError("""ERROR: it and box arguments should be len 4 but is 
                        it:{} and box:{}""".format(len(it), len(box)))
  box_x1, box_y1, box_x2, box_y2 = box
  x1, y1, x2, y2 = it

  x_in_box = box_x1 <= x1 <= box_x2 and box_x1 <= x2 <= box_x2
  y_in_box = box_y1 <= y1 <= box_y2 and box_y1 <= y2 <= box_y2

  if x_in_box and y_in_box:
    return True
  else:
    return False


def get_annotation_by_frame(anns, frame):
  return [ann for ann in anns if ann[FRAME] == frame]


def pass_trought_line(line, annotations, detection_count=60):
  if len(line) != 4:
    raise ValueError("ERROR: line argument should be len 4 but is " + len(line))
  p1, q1 = (line[0], line[1]), (line[2], line[3]) 
  detection_index = -1
  for idx in range(len(annotations) - 1):
    detection = annotations[idx][DETECTIONS][0]
    detection_next = annotations[idx+1][DETECTIONS][0]
    p2 = get_middle_detection_points(detection)
    q2 = get_middle_detection_points(detection_next)
    if intersect(p1, q1, p2, q2):
      detection_index = idx
      break
  
  start = max(detection_index-detection_count//2, 0)
  end = min(detection_index+detection_count//2, len(annotations))
  pass_trought_line_detections = []
  for idx in range(start, end):
    pass_trought_line_detections.append(annotations[idx])
  return pass_trought_line_detections


def on_segment(p, q, r): 
    if ((q[0] <= max(p[0], r[0])) and 
        (q[0] >= min(p[0], r[0])) and 
        (q[1] <= max(p[1], r[1])) and 
        (q[1] >= min(p[1], r[1]))): 
        return True
    return False

  
def orientation(p, q, r): 
  val = ((float(q[1] - p[1]) * (r[0] - q[0])) - (float(q[0] - p[0]) * (r[1] - q[1]))) 
  if (val > 0): 
    return 1
  elif (val < 0): 
    return 2
  else: 
    return 0


def intersect(p1, q1, p2, q2): 
    o1 = orientation(p1, q1, p2) 
    o2 = orientation(p1, q1, q2) 
    o3 = orientation(p2, q2, p1) 
    o4 = orientation(p2, q2, q1) 
    if ((o1 != o2) and (o3 != o4)): 
        return True
    if ((o1 == 0) and on_segment(p1, p2, q1)): 
        return True
    if ((o2 == 0) and on_segment(p1, q2, q1)): 
        return True
    if ((o3 == 0) and on_segment(p2, p1, q2)): 
        return True
    if ((o4 == 0) and on_segment(p2, q1, q2)): 
        return True
    return False


def is_inside_polygon(polygon_points, point) -> bool:
  polygon_points_count = len(polygon_points)
  if polygon_points_count < 3:
      return False
  extreme = (INT_MAX, point[1])
  count = i = 0
  while True:
    next = (i + 1) % polygon_points_count
    polygon_point = polygon_points[i]
    next_polygon_point = polygon_points[next]
    if (intersect(polygon_point, next_polygon_point, point, extreme)):
      if orientation(polygon_point, point, next_polygon_point) == 0:
        return on_segment(polygon_point, point, next_polygon_point)
      count += 1
    i = next
    if (i == 0):
      break
  return (count % 2 == 1)


def reconstruct_video_event(video_name, output_video_name, annotations):
  video = Video(video_name)
  width = video.width
  height = video.height
  fps = video.fps

  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  out = cv2.VideoWriter(output_video_name, fourcc, fps, (width, height))

  start = annotations[0][FRAME]
  end = annotations[-1][FRAME]

  video.set(cv2.CAP_PROP_POS_FRAMES, start)

  detections = []
  annotation_index = 0
  for frame_number in range(start, end):
    ret, img = video.read()
    if not ret:
      break

    if frame_number == annotations[annotation_index][FRAME]:
      detections = annotations[annotation_index][DETECTIONS]
      annotation_index += 1 

    for detection in detections:
      vizualize_detection(img, detection)

    out.write(img)

  out.release()
  video.release()


def vizualize_detection(frame, detection):
  x1, y1, x2, y2 = detection[BOX]
  w, h = int(x2-x1), int(y2-y1)

  score = detection[DETECTION_SCORE]
  id = detection[ID]
  type = detection[TYPE]
  direction_vector = detection[DIRECTION_VECTOR]

  # draw bounding box
  cv2.rectangle(frame, (x1, y1),(x2, y2), (255, 0, 0), 2)
  # draw object info label
  object_label = "Object ID: {}, Object type: {}".format(id, type)
  cv2.putText(frame, object_label, (x1, y1-10), 
    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)
  # draw action info label
  if ACTION in detection:
    action_label = "Actions: {}".format(str(detection[ACTION]))
    cv2.putText(frame, action_label, (x1, y1-30), 
      cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)

  # draw direction      
  x_dir, y_dir = direction_vector
  if type == "person":
    middle_point = int(x1 + (w/2)), int(y1 + h)
  else:
    middle_point = int(x1 + (w/2)), int(y1 + (h/2))
  direction_point = x_dir + middle_point[0], y_dir + middle_point[1]
  
  cv2.line(frame, middle_point, direction_point, (0, 255, 0), 2)
  cv2.circle(frame, direction_point, 5, (255, 0, 0), -1)
  cv2.putText(frame, str(id), (x_dir, y_dir), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)

