import pygame, sys
import cv2
from math import *
from splipy import *
import numpy as np
from pygame.locals import *

# manually add smoke by mouse-clicking input
def add_smoke(x, img):
    size      = 20
    intensity = 200
    for i in range(x[0]-size,x[0]+size):
        if i<1 or i>=img.shape[0]:
            continue
        for j in range(x[1]-size,x[1]+size):
            if j<1 or j>=img.shape[1]:
                continue
            dist2 = (x[0]-i)**2+(x[1]-j)**2
            if dist2 > size*size:
                continue;
            newI = img[i,j] + intensity*(1-dist2/size/size)
            newI = int(max(0,min(newI, 255)))
            img[i,j] = newI

# diffuse existing smoke (physics)
def diffuse(img):
    return cv2.blur(img, (15,15))

# color convertion: hue-saturation-value (HSV) to red-green-blue (RGB) format
def hsv_to_rgb(hsv):
    # as provided by http://www.rapidtables.com/convert/color/hsv-to-rgb.htm
    rgb = []
    for (h,s,v) in hsv:
        c = v*s
        x = c*(1-abs((h/60)%2-1))
        m = v-c
        if 0<=h<=60:
            rgb.append((int((c+m)*255), int((x+m)*255), int((0+m)*255)))
        elif 60<=h<=120:
            rgb.append((int((x+m)*255), int((c+m)*255), int((0+m)*255)))
        elif 120<=h<=180:
            rgb.append((int((0+m)*255), int((c+m)*255), int((x+m)*255)))
        elif 180<=h<=240:
            rgb.append((int((0+m)*255), int((x+m)*255), int((c+m)*255)))
        elif 240<=h<=300:
            rgb.append((int((x+m)*255), int((0+m)*255), int((c+m)*255)))
        elif 300<=h<=360:
            rgb.append((int((c+m)*255), int((0+m)*255), int((x+m)*255)))
    return rgb



# initialize stuff
WIDTH  = 1200 # image size
HEIGHT = 600
pygame.init()
surf = pygame.display.set_mode((WIDTH,HEIGHT), HWSURFACE|HWPALETTE, 8)
clock = pygame.time.Clock()
pygame.display.set_caption('Poisson!')

# setup buffer image and colorscheme
grayscale = [(i,i,i) for i in range(256)]
jet       = hsv_to_rgb([((255-i)*240/256,.8,.8) for i in range(256)])
surf.set_palette(jet)
img = np.zeros((WIDTH,HEIGHT), np.uint8)


# main-loop
frameCount = 0
while True:
    # process all events
    for event in pygame.event.get():
        if pygame.mouse.get_pressed()[0]:
            x = pygame.mouse.get_pos()
            add_smoke(x, img)
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

    # advance physics
    img = diffuse(img)

    # report performance
    clock.tick()
    frameCount += 1
    if frameCount % 60 == 0:
        print('FPS: %f' % (clock.get_fps()))

    # draw graphics
    pygame.surfarray.blit_array(surf, img)
    pygame.display.update()
