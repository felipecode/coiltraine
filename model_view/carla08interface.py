#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
    Example of automatic vehicle control from client side.
"""

from __future__ import print_function

import scipy
import os
import random
import re
import sys
import weakref
import queue

from PIL import Image

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


from carla08.client import make_carla_client
from carla08.settings import CarlaSettings

from carla08.client import VehicleControl
from carla08 import sensor
import carla08 as carla
# from agents.navigation.roaming_agent import *
# from agents.navigation.basic_agent import *
# Imports from the coiltraine system.
from coilutils import AttributeDict





WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600


"""
POSITIONS = [ [105, 29], [27, 130], [102, 87], [132, 27], [24, 44],
              [96, 26], [34, 67], [28, 1], [140, 134], [105, 9],
              [148, 129], [65, 16], [21, 16], [147, 97], [42, 51],
              [30, 41], [16, 107], [69, 47], [102, 95], [16, 145],
              [111, 64], [79, 47], [84, 69], [73, 31], [37, 81],
              [35, 57], [42, 116], [75, 47], [132, 143], [145, 8],
              [43, 107], [61, 111], [137, 105], [24, 72], [0, 77],
              [17, 80], [12, 32], [3, 64], [146, 32], [33, 4]]

"""

POSITIONS_TOWN01 = [[105, 29], [27, 130], [102, 87], [132, 27], [25, 44],
                    [4, 64], [34, 67], [54, 30], [140, 134], [105, 9],
                    [148, 129], [65, 18], [21, 16], [147, 97], [134, 49],
                    [30, 41], [81, 89], [69, 45], [102, 95], [18, 145],
                    [111, 64], [79, 45], [84, 69], [73, 31], [37, 81]]



POSITIONS_TOWN02 = [[19, 66], [79, 14], [19, 57], [39, 53], [60, 26],
                    [53, 76], [42, 13], [31, 71], [59, 35], [47, 16],
                    [10, 61], [66, 3], [20, 79], [14, 56], [26, 69],
                    [79, 19], [2, 29], [16, 14], [5, 57], [77, 68],
                    [70, 73], [46, 67], [34, 77], [61, 49], [21, 12]]

# [53, 140] ,  [125, 42]  , [94, 80]

#TODO for now all the resolutions and FOVs are the same.
FOV = 100


sensors_frequency = {'CameraRGB': 1}

sensors_yaw = {'CameraRGB': 0}


lat_noise_percent = 0
long_noise_percent = 0

NumberOfVehicles = [30, 30] #[30, 60]  # The range for the random numbers that are going to be generated
NumberOfPedestrians = [250, 250] #[50, 100]

set_of_weathers = [1, 3, 6, 8, 10, 14]


def make_carla_settings(benchmark=None, task=None):
    """Make a CarlaSettings object with the settings we need."""



    settings = CarlaSettings()
    settings.set(
        SendNonPlayerAgentsInfo=True,
        SynchronousMode=True,
        NumberOfVehicles=30,
        NumberOfPedestrians=50,
        WeatherId=1)

    settings.set(DisableTwoWheeledVehicles=True)

    settings.randomize_seeds() # IMPORTANT TO RANDOMIZE THE SEEDS EVERY TIME
    camera0 = sensor.Camera('rgb')
    camera0.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    camera0.set(FOV=100)
    camera0.set_image_size(800, 600)
    camera0.set_position(2.0, 0.0, 1.4)
    camera0.set_rotation(-15.0, 0, 0)


    settings.add_sensor(camera0)

    camera0 = sensor.Camera('hudcamera')
    camera0.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)
    camera0.set(FOV=100)
    camera0.set_image_size(1280, 730)
    camera0.set_position(-5.5, 0.0, 2.8)
    camera0.set_rotation(-15.0, 0, 0)

    settings.add_sensor(camera0)

    return settings



# ==============================================================================
# -- World ---------------------------------------------------------------
# ==============================================================================

class World(object):
    def __init__(self, client, hud):
        self.client = client
        self.hud = hud
        self.vehicle = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.camera_manager = None
        self._weather_index = 0
        self._command_cache = 2.0
        self.restart()
        #self.world.on_tick(hud.on_world_tick)

    def restart(self):
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager._index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager._transform_index if self.camera_manager is not None else 0


        number_of_vehicles = random.randint(NumberOfVehicles[0], NumberOfVehicles[1])
        number_of_pedestrians = random.randint(NumberOfPedestrians[0], NumberOfPedestrians[1])
        weather = random.choice(set_of_weathers)
        carla_settings = make_carla_settings()
        carla_settings.set(
            NumberOfVehicles=number_of_vehicles,
            NumberOfPedestrians=number_of_pedestrians,
            WeatherId=weather
        )

        self.scene = self.client.load_settings(carla_settings)
        if self.scene.map_name == 'Town01':
            POSITIONS = POSITIONS_TOWN01
        else:
            POSITIONS = POSITIONS_TOWN02

        print('Starting new episode...')
        self._random_pose = random.choice(POSITIONS)

        self.client.start_episode(self._random_pose[0])

        # Set up the sensors.
        self.camera_manager = CameraManager(self.vehicle, self.hud)
        self.camera_manager._transform_index = cam_pos_index

        #actor_type = get_actor_display_name(self.vehicle)
        #self.hud.notification(actor_type)


    def get_agent_sensor(self):
        return  self.client.read_data()

    def get_objective(self):
        return self.scene.player_start_spots[self._random_pose[1]]

    def apply_control(self, control):
        print ("Sending control")
        self.client.send_control(control)

    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================

class KeyboardControl(object):
    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        self._control = VehicleControl()
        self._command_cache = 2.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)


    def parse_events(self, world, clock):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_h or (
                        event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_LEFT:
                    self._command_cache = 3.0
                elif event.key == K_RIGHT:
                    self._command_cache = 4.0
                elif event.key == K_DOWN:
                    self._command_cache = 2.0
                elif event.key == K_UP:
                    self._command_cache = 5.0

        world._command_cache = self._command_cache
    def get_command(self):
        return self._command_cache

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        fonts = [x for x in pygame.font.get_fonts() if 'mono' in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.warning_text = ColorText(pygame.font.Font(mono, 60), width, height)
        self.server_fps = 0
        self.frame_number = 0
        self.simulation_time = 0
        self.command_status = 2.0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame_number = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        # TODO add the measurements instead.
        if not self._show_info:
            return

        self.command_status = world._command_cache
        """
        t = world.vehicle.get_transform()
        v = world.vehicle.get_velocity()
        c = world.vehicle.get_vehicle_control()
        heading = 'N' if abs(t.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(t.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > t.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > t.rotation.yaw > -179.5 else ''
        """
        """
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame_number - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        
        # TODO Reduce the used info test.
        self._info_text = [
            'Server:  % 16d FPS' % self.server_fps,
            '',
            'Map:     % 20s' % world.world.map_name,
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (t.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'Height:  % 18.0f m' % t.location.z,
            '',
            ('Throttle:', c.throttle, 0.0, 1.0),
            ('Steer:', c.steer, -1.0, 1.0),
            ('Brake:', c.brake, 0.0, 1.0),
            ('Reverse:', c.reverse),
            ('Hand brake:', c.hand_brake),
            ('Manual:', c.manual_gear_shift),
            'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear),
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)
        ]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt(
                (l.x - t.location.x) ** 2 + (l.y - t.location.y) ** 2 + (l.z - t.location.z) ** 2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if
                        x.id != world.vehicle.id]
            for d, vehicle in sorted(vehicles):
                if d > 200.0:
                    break
        """
        self._notifications.tick(world, clock)

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            print ("SHOWING INFO COMMAND ", self.command_status)
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in
                                  enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8),
                                               (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))

            if self.command_status == 3.0:
                self.warning_text.set_text('GO LEFT')

            elif self.command_status == 4.0:
                self.warning_text.set_text('GO RIGHT')

            elif self.command_status == 5.0:
                self.warning_text.set_text('GO STRAIGHT')

            if self.command_status != 2.0:
                self.warning_text.render(display)
                    # if status['directions'] == 4:
                    #     text = "GO RIGHT"
                    #     extraback = size_x / 7
                    # elif status['directions'] == 3:
                    #     text = "GO LEFT"
                    #     extraback = size_x / 13
                    # else:
                    #     text = "GO STRAIGHT"
                    #     extraback = int(size_x / 2.8)
                    #
                    #
                    # if status['directions'] != 2:
                    #     direction_pos = (int(size_y / 2 - size_x / 2 - extraback), int(size_x / 2 - size_x / 4))

                    #    self.paint_on_screen(int(size_x / 6), text, (0, 255, 0), direction_pos, screen_position)

                    #self.paint_on_screen(int(size_x / 10), "Speed: %.2f" % status['speed'],
                    #                     (64, 255, 64),
                    #                     (int(size_y / 2 - 200), 30),
                    #                     screen_position)

                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================

class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * 22))
            self._render = False
        self.surface.set_alpha(220)


    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


class ColorText(object):
    def __init__(self, font, width, height, color=(0, 255, 0)):
        lines = __doc__.split('\n')
        self.font = font
        self.color = color
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * self.dim[1])
        self.surface = pygame.Surface(self.dim)

        self._render = True
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def set_text(self, text):
        text_texture = self.font.render(text, False, self.color)
        self.surface = pygame.Surface(self.dim, pygame.SRCALPHA, 32)
        self.surface.blit(text_texture, (10, 11))

    def render(self, display):
        print ("REND TEXT")
        print (self.pos)
        print (self._render)
        if self._render:
            display.blit(self.surface, self.pos)

# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================

class CameraManager(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._surface = None
        # Now we plot the posible views that the agent can have
        self._agent_view = None

        self._agent_view_internal_1 = None
        self._agent_view_internal_2 = None
        self._agent_view_internal_3 = None

        self._parent = parent_actor
        self._hud = hud
        self._recording = False
        self._index = None

    def render(self, display):
        if self._surface is not None:
            display.blit(self._surface, (0, 0))

        if self._agent_view is not None:
            display.blit(self._agent_view, (170, 3 * display.get_height() / 4))

        if self._agent_view_internal_1 is not None:
            display.blit(self._agent_view_internal_1, (420, 3 * display.get_height() / 4))

        if self._agent_view_internal_2 is not None:
            display.blit(self._agent_view_internal_2, (670, 3 * display.get_height() / 4))

        if self._agent_view_internal_3 is not None:
            display.blit(self._agent_view_internal_3, (920, 3 * display.get_height() / 4))

    def set_sensor(self, image):
        self._surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))

    def show_image_mini(self, image1, image2, image3, image4):

        self._agent_view = pygame.surfarray.make_surface(image1.swapaxes(0, 1))
        self._agent_view_internal_1 = pygame.surfarray.make_surface(image2.swapaxes(0, 1))
        self._agent_view_internal_2 = pygame.surfarray.make_surface(image3.swapaxes(0, 1))
        self._agent_view_internal_3 = pygame.surfarray.make_surface(image4.swapaxes(0, 1))


# ==============================================================================
# -- game_loop() ---------------------------------------------------------
# ==============================================================================

# TODO THis can be the same for both in the future

def game_loop(args, agent):
    pygame.init()
    pygame.font.init()
    world = None
    total_images = 0

    if args.output_folder is not None:
        os.makedirs(args.output_folder)

    try:
        with make_carla_client(args.host, args.port) as client:
            # Hack to fix for the issue 310, we force a reset, so it does not get
            #  the positions on first server reset.
            client.load_settings(CarlaSettings())
            client.start_episode(0)

            display = pygame.display.set_mode(
                (args.width, args.height),
                pygame.HWSURFACE | pygame.DOUBLEBUF)

            hud = HUD(args.width, args.height)
            world = World(client, hud)
            controller = KeyboardControl(world, False)


            #spawn_point = world.world.get_map().get_spawn_points()[0]
            objective = world.get_objective()
            clock = pygame.time.Clock()
            while True:
                if controller.parse_events(world, clock):
                    return

                world.tick(clock)
                measurements, sensor_data = world.get_agent_sensor()
                world.render(display)
                pygame.display.flip()
                print (controller.get_command())
                control = agent.run_step(measurements,
                                         sensor_data, controller.get_command(),
                                         (objective.location.x,
                                          objective.location.y,
                                          objective.location.z))

                world.camera_manager.set_sensor(sensor_data['hudcamera'].data)
                attentions = agent.get_attentions()
                world.camera_manager.show_image_mini(agent.latest_image,
                                                     attentions[0],
                                                     attentions[1],
                                                     attentions[2])
                print(control)
                world.apply_control(control)
                if args.output_folder is not None:
                    scipy.misc.imsave(os.path.join(args.output_folder,
                                                   'image' + str(total_images) + '.png'),
                                      agent.latest_image)
                    scipy.misc.imsave(os.path.join(args.output_folder,
                                                   'layer1_' + str(total_images) + '.png'),
                                      attentions[0])
                    scipy.misc.imsave(os.path.join(args.output_folder,
                                                   'layer2_' + str(total_images) + '.png'),
                                      attentions[1])
                    scipy.misc.imsave(os.path.join(args.output_folder,
                                                   'layer3_' + str(total_images) + '.png'),
                                      attentions[2])
                    total_images += 1
    finally:

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================
