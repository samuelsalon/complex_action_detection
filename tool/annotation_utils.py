import ast
import cv2
from tool.video import Video
import numpy as np


def parse_annotations(annotation_file):
  annotations = []
  with open(annotation_file, mode='r') as ann_file:
    annotation_list = ann_file.read().split("\n")
    for annotation in annotation_list:
      if len(annotation) < 1:
        continue
      annotations.append(ast.literal_eval(annotation))
  return annotations


def middle_point(x1, y1, x2, y2):
  return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_middle_annotation_points(annotation):
  x1, y1, x2, y2 = annotation['bbox']
  mid_x, mid_y = middle_point(x1, y1, x2, y2)
  return Point(mid_x, mid_y)


def image_cut(img, x1, y1, x2, y2):
  return img[y1:y2, x1:x2]


def pos_in_box(position, box):
  if len(box) != 4 or len(position) != 2:
    return False

  box_x1, box_y1, box_x2, box_y2 = box
  x, y = position

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


def get_annotation_by_object_id(anns, id):
  return [ann for ann in anns if ann['object']['id'] == id]


def get_annotation_by_frame(anns, frame):
  return [ann for ann in anns if ann['frame'] == frame]


class Point: 
  def __init__(self, x, y): 
    self.x = x 
    self.y = y 


def pass_trought_line(line, annotations):
  if len(line) != 4:
    raise ValueError("ERROR: line argument should be len 4 but is " + len(line))
  p1, q1 = Point(line[0], line[1]), Point(line[2], line[3]) 
  intersected_annotations = []
  for idx in range(len(annotations) - 1):
    p2 = get_middle_annotation_points(annotations[idx])
    q2 = get_middle_annotation_points(annotations[idx + 1])
    if intersect(p1, q1, p2, q2):
      intersected_annotations.append((annotations[idx], annotations[idx + 1]))
  return intersected_annotations


def on_segment(p, q, r): 
    if ( (q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and 
           (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))): 
        return True
    return False

  
def orientation(p, q, r): 
    val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y)) 
    if (val > 0): 
        return 1
    elif (val < 0): 
        return 2
    else: 
        return 0


def intersect(p1,q1,p2,q2): 
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


def reconstruct_video_event(video_name, output_video_name, annotations):
  video = Video(video_name)
  width = video.width
  height = video.height
  fps = video.fps

  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  out = cv2.VideoWriter(output_video_name, fourcc, fps, (width, height))

  start = annotations[0]['frame']
  end = annotations[-1]['frame']

  video.set(cv2.CAP_PROP_POS_FRAMES, start)

  ann_counter = 0
  for frame in range(start, end+1):
    ret, img = video.read()
    if not ret:
      break

    annotation = annotations[ann_counter]
    if annotation['frame'] == frame:
      id_and_class_name = str(annotation["object"]["id"]) + ". " + annotation["object"]["object_type"]
      x1, y1, x2, y2 = annotation['bbox']
      cv2.putText(img, id_and_class_name, (x1-20, y1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0, 0), 2)
      cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
      ann_counter += 1

    out.write(img)

  out.release()
  video.release()
