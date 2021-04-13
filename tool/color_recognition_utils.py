import cv2
import numpy as np 
import matplotlib.pyplot as plt

def is_olored_area(image, color):
  lower, upper = get_hsv_color_bounds(color)
  hsv_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)
  mask = cv2.inRange(hsv_image, lower, upper)
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
  opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
  cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]
  area = 0
  for c in cnts:
    area += cv2.contourArea(c)
    cv2.drawContours(image,[c], 0, (0,0,0), 2)
  return area


def get_hsv_color_bounds(color, h_bound=18, s_bound=105, v_bound=40):
  image_color = np.uint8([[color]])
  h,s,v = cv2.cvtColor(image_color, cv2.COLOR_BGR2HSV).flatten()
  lower = [max(0, h-h_bound), max(0, s-s_bound), max(0, v-v_bound)]
  upper = [min(255, h+h_bound), min(255, s+s_bound), min(255, v+v_bound)]
  return np.array(lower, dtype='uint8'), np.array(upper, dtype='uint8')


def crop_image(image, start, end):
  x1, y1 = start
  x2, y2 = end
  return image[y1:y2, x1:x2]
