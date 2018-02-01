import pygame, sys
from splipy import *
import numpy as np
from nutils import mesh, function, plot
from pygame.locals import *
from heuristic import *
from fem import *
import cProfile

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


def run_game():
    # initialize stuff
    WIDTH  = 1200 # image size
    HEIGHT = 600
    physics2 = Heuristic([WIDTH, HEIGHT])
    physics  = Fem([WIDTH, HEIGHT], [72,36], [2,2])
    pygame.init()
    surf = pygame.display.set_mode((WIDTH,HEIGHT), HWSURFACE|HWPALETTE, 8)
    clock = pygame.time.Clock()
    pygame.display.set_caption('Poisson!')


    # setup buffer image and colorscheme
    grayscale = [(i,i,i) for i in range(256)]
    jet       = hsv_to_rgb([((255-i)*240/256,.8,.8) for i in range(256)])
    surf.set_palette(jet)


    # main-loop
    frameCount = 0
    while True:
        # process all events
        for event in pygame.event.get():
            if pygame.mouse.get_pressed()[0]:
                x = pygame.mouse.get_pos()
                physics.add_smoke(x)
                # physics2.add_smoke(x)
            else:
                physics.stop_smoke()
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        # advance physics
        physics.diffuse(clock.get_time()/1000)
        # physics2.diffuse(clock.get_time())

        # report performance
        clock.tick()
        frameCount += 1
        if frameCount % 60 == 0:
            t = physics.get_time()
            print('FPS: %f' % (clock.get_fps()))
            for i in t:
                print('  '+str(i)+':\t'+str(t[i]))

        # draw graphics
        pygame.surfarray.blit_array(surf, physics.get_image())
        pygame.display.update()


if __name__ == '__main__':
    # cProfile.run('run_game()')
    run_game()
