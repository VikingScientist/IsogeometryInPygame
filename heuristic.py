import cv2
import numpy as np
from math import *

class Heuristic:
  
  def __init__(self, physical):
      self.img = np.zeros(physical, np.uint8)

  # manually add smoke by mouse-clicking input
  def add_smoke(self, x):
      size      = 20
      intensity = 200
      for i in range(x[0]-size,x[0]+size):
          if i<1 or i>=self.img.shape[0]:
              continue
          for j in range(x[1]-size,x[1]+size):
              if j<1 or j>=self.img.shape[1]:
                  continue
              dist2 = (x[0]-i)**2+(x[1]-j)**2
              if dist2 > size*size:
                  continue;
              newI = self.img[i,j] + intensity*(1-dist2/size/size)
              newI = int(max(0,min(newI, 255)))
              self.img[i,j] = newI

  # diffuse existing smoke (physics)
  def diffuse(self, dt):
      self.img = cv2.blur(self.img, (15,15))

  # get evaluated solution field
  def get_image(self):
      return self.img


