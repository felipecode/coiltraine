import colorsys
import pygame
import numpy as np
import matplotlib.pyplot as plt
from random import randint

from skimage import transform as trans

import time
import math
import scipy
import cv2

clock = pygame.time.Clock()

rsrc = \
    [[43.45456230828867, 118.00743250075844],
     [104.5055617352614, 69.46865203761757],
     [114.86050156739812, 60.83953551083698],
     [129.74572757609468, 50.48459567870026],
     [132.98164627363735, 46.38576532847949],
     [301.0336906326895, 98.16046448916306],
     [238.25686790036065, 62.56535881619311],
     [227.2547443287154, 56.30924933427718],
     [209.13359962247614, 46.817221154818526],
     [203.9561297064078, 43.5813024572758]]
rdst = \
    [[10.822125594094452, 1.42189132706374],
     [21.177065426231174, 1.5297552836484982],
     [25.275895776451954, 1.42189132706374],
     [36.062291434927694, 1.6376192402332563],
     [40.376849698318004, 1.42189132706374],
     [11.900765159942026, -2.1376192402332563],
     [22.25570499207874, -2.1376192402332563],
     [26.785991168638553, -2.029755283648498],
     [37.033067044190524, -2.029755283648498],
     [41.67121717733509, -2.029755283648498]]

tform3_img = trans.ProjectiveTransform()
tform3_img.estimate(np.array(rdst), np.array(rsrc))


def draw_vbar_on(img,bar_intensity,x_pos,color=(0,0,255)):


  bar_size = int(img.shape[1]/6 * bar_intensity)
  initial_y_pos = img.shape[0] - img.shape[0]/6
  #print bar_intensity

  for i in range(bar_size):
    if bar_intensity > 0.0:
      y = initial_y_pos - i
      for j in range(20):
        img[y , x_pos +j] = color


def generate_ncolors(num_colors):
    color_pallet = []
    for i in range(0, 360, 360 / num_colors):
        hue = i
        saturation = 90 + float(randint(0, 1000)) / 1000 * 10
        lightness = 50 + float(randint(0, 1000)) / 1000 * 10

        color = colorsys.hsv_to_rgb(float(hue) / 360.0, saturation / 100, lightness / 100)

        color_pallet.append(color)

    # addColor(c);
    return color_pallet


def get_average_over_interval(vector, interval):
    avg_vector = []
    for i in range(0, len(vector), interval):
        initial_train = i
        final_train = i + interval

        avg_point = sum(vector[initial_train:final_train]) / interval
        avg_vector.append(avg_point)

    return avg_vector


def get_average_over_interval_stride(vector, interval, stride):
    avg_vector = []
    for i in range(0, len(vector) - interval, stride):
        initial_train = i
        final_train = i + interval

        avg_point = sum(vector[initial_train:final_train]) / interval
        avg_vector.append(avg_point)

    return avg_vector


def perspective_tform(x, y):
    p1, p2 = tform3_img((x, y))[0]
    return p2, p1


# ***** functions to draw lines *****
def draw_pt(img, x, y, color, sz=1):
    row, col = perspective_tform(x, y)
    if 0 <= row < img.shape[0] and 0 <= col < img.shape[1]:
        img[int(row - sz):int(row + sz), int(col - sz - 65):int(col + sz - 65)] = color


def draw_path(img, path_x, path_y, color):
    for x, y in zip(path_x, path_y):
        draw_pt(img, x, y, color)


# ***** functions to draw predicted path *****

def calc_curvature(v_ego, angle_steers, angle_offset=0):
    deg_to_rad = np.pi / 180.
    slip_fator = 0.0014  # slip factor obtained from real data
    steer_ratio = 15.3
    wheel_base = 2.67

    angle_steers_rad = (angle_steers - angle_offset) * deg_to_rad
    curvature = angle_steers_rad / (steer_ratio * wheel_base * (1. + slip_fator * v_ego ** 2))
    return curvature


def calc_lookahead_offset(v_ego, angle_steers, d_lookahead, angle_offset=0):
    # *** this function return teh lateral offset given the steering angle, speed and the lookahead distance
    curvature = calc_curvature(v_ego, angle_steers, angle_offset)

    # clip is to avoid arcsin NaNs due to too sharp turns
    y_actual = d_lookahead * np.tan(np.arcsin(np.clip(d_lookahead * curvature, -0.999, 0.999)) / 2.)
    return y_actual, curvature


def draw_path_on(img, speed_ms, angle_steers, color=(0, 0, 255)):
    path_x = np.arange(0., 50.1, 0.5)
    path_y, _ = calc_lookahead_offset(speed_ms, angle_steers, path_x)
    draw_path(img, path_x, path_y, color)


class ScreenManager(object):

    def __init__(self, load_steer=False):

        pygame.init()
        # Put some general parameterss
        self._render_iter = 2000
        self._speed_limit = 50.0
        if load_steer:
            self._wheel = cv2.imread('./drive_interfaces/wheel.png')  # ,cv2.IMREAD_UNCHANGED)
            self._wheel = cv2.resize(self._wheel, (int(0.08 * self._wheel.shape[0]), int(0.08 * self._wheel.shape[1])))


    # If we were to load the steering wheel load it

    # take into consideration the resolution when ploting
    # TODO: Resize properly to fit the screen ( MAYBE THIS COULD BE DONE DIRECTLY RESIZING screen and keeping SURFACES)

    def start_screen(self, resolution, aspect_ratio, scale=1):

        self._resolution = resolution

        self._aspect_ratio = aspect_ratio
        self._scale = scale

        size = (resolution[0] * aspect_ratio[0], resolution[1] * aspect_ratio[1])

        self._screen = pygame.display.set_mode((size[0] * scale, size[1] * scale), pygame.DOUBLEBUF)

        # self._screen.set_alpha(None)

        pygame.display.set_caption("Human/Machine - Driving Software")

        self._camera_surfaces = []

        for i in range(aspect_ratio[0] * aspect_ratio[1]):
            camera_surface = pygame.surface.Surface(resolution, 0, 24).convert()

            self._camera_surfaces.append(camera_surface)

    def paint_on_screen(self, size, content, color, position, screen_position):

        myfont = pygame.font.SysFont("monospace", size * self._scale, bold=True)

        position = (position[0] * self._scale, position[1] * self._scale)

        final_position = (position[0] + self._resolution[0] * (self._scale * (screen_position[0])), \
                          position[1] + (self._resolution[1] * (self._scale * (screen_position[1]))))

        content_to_write = myfont.render(content, 1, color)

        self._screen.blit(content_to_write, final_position)

    def set_array(self, array, screen_position, position=(0, 0), scale=None):

        if scale is None:
            scale = self._scale

        if array.shape[0] != self._resolution[1] or array.shape[1] != self._resolution[0]:
            array = scipy.misc.imresize(array, [self._resolution[1], self._resolution[0]])

        # print array.shape, self._resolution

        final_position = (position[0] + self._resolution[0] * (scale * (screen_position[0])), \
                          position[1] + (self._resolution[1] * (scale * (screen_position[1]))))

        # pygame.surfarray.array_colorkey(self._camera_surfaces[screen_number])
        self._camera_surfaces[screen_position[0] * screen_position[1]].set_colorkey((255, 0, 255))
        pygame.surfarray.blit_array(self._camera_surfaces[screen_position[0] * screen_position[1]],
                                    array.swapaxes(0, 1))

        camera_scale = pygame.transform.scale(self._camera_surfaces[screen_position[0] * screen_position[1]],
                                              (int(self._resolution[0] * scale), int(self._resolution[1] * scale)))

        self._screen.blit(camera_scale, final_position)

    def draw_wheel_on(self, steer, screen_position):

        cols, rows, c = self._wheel.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -90 * steer, 1)
        rot_wheel = cv2.warpAffine(self._wheel, M, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        # scale = 0.5
        position = (self._resolution[0] / 2 - cols / 2, int(self._resolution[1] / 1.5) - rows / 2)
        # print position

        wheel_surface = pygame.surface.Surface((rot_wheel.shape[1], rot_wheel.shape[0]), 0, 24).convert()
        # print array.shape, self._resolution

        # final_position = (position[0] + self._resolution[0]*(scale*(screen_number%3)),\
        #	position[1] + (self._resolution[1]*(scale*(screen_number/3))))

        # pygame.surfarray.array_colorkey(self._camera_surfaces[screen_number])
        wheel_surface.set_colorkey((0, 0, 0))
        pygame.surfarray.blit_array(wheel_surface, rot_wheel.swapaxes(0, 1))

        self._screen.blit(wheel_surface, position)

    # This one plot the nice wheel

    def plot_camera(self, sensor_data, screen_position=[0, 0]):

        if sensor_data.shape[2] < 3:
            sensor_data = np.stack((sensor_data,) * 3, axis=2)
            sensor_data = np.squeeze(sensor_data)
        # print sensor_data.shape
        self.set_array(sensor_data, screen_position)

        pygame.display.flip()

    def plot_camera_steer(self, sensor_data, steer, screen_position=[0, 0]):

        if sensor_data.shape[2] < 3:
            sensor_data = np.stack((sensor_data,) * 3, axis=2)
            sensor_data = np.squeeze(sensor_data)

        draw_path_on(sensor_data, 20, -steer * 10.0, (0, 255, 0))

        self.set_array(sensor_data, screen_position)


        pygame.display.flip()



    def plot3camrcnoise(self, sensor_data, \
                        steer, noise, difference, \
                        screen_number=0):

        # Define our fonts

        # draw_path_on(img, 10, -angle_steers*40.0)

        draw_path_on(sensor_data, 20, -steer * 20.0, (255, 0, 0))

        draw_path_on(sensor_data, 20, -noise * 20.0, (0, 255, 0))

        draw_path_on(sensor_data, 20, -difference * 20.0, (0, 0, 255))



        #pygame.image.save(self._screen, "footage_offline/imgcamera" + str(self._render_iter) +".png")

        self.set_array(sensor_data, screen_number)

        self._render_iter += 1
        pygame.display.flip()