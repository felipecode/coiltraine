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

import argparse
import collections
import datetime
import glob
import time
import copy
from collections import deque
import logging
import math
import os
import random
import re
import sys
import weakref
import matplotlib.pyplot as plt
cmap = plt.get_cmap('inferno')
import scipy

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


import carla
from carla import ColorConverter as cc

# Interface classes to make this work with the original interface in 0.8.4


class PlayerMeasurements:
    def __init__(self):
        self.forward_speed = 0.0


class Measurements:
    def __init__(self):
        self.player_measurements = PlayerMeasurements()


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================

def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate-1] + u'\u2026') if len(name) > truncate else name


# TODO Simplify this interface.
class Listener():
    def __init__(self, key, collector, sensor):
        self.collector = collector
        self.key = key
        sensor.listen(self.listen)

    def listen(self, data):
        self.collector.collect(self.key, data)

class SensorCollector():
    def __init__(self):
        self.queues = {}

    def add_sensor(self, key, sensor, listener_class=Listener):
        listener = listener_class(key, self, sensor)
        self.queues[key] = deque()

    def collect(self, key, data):
        self.queues[key].append(data)

    def read_nowait(self):
        return self._read_all()

    def _read_all(self):
        dict = {}
        for key in self.queues.keys():
            dict[key] = copy.copy(self.queues[key])
            self.queues[key].clear()

        return dict

    def read(self, wait_period=0.05, timeout = 10):
        empty = True
        end = time.time() + timeout

        while empty and time.time() < end:
            dict = self._read_all()
            empty = min([len(dict[k]) for k in dict]) == 0

            if empty:
                time.sleep(wait_period)

        # timeout!
        if empty:
            print('Timeout!!!!')

        return dict





# ==============================================================================
# -- World ---------------------------------------------------------------
# ==============================================================================
class Camera():

    def __init__(self, world, camera, agent_vehicle):

        blueprint_library = world.get_blueprint_library()
        camera_bp = blueprint_library.find(camera["type"])
        camera_bp.set_attribute('image_size_x', str(camera["image_size_x"]))
        camera_bp.set_attribute('image_size_y', str(camera["image_size_y"]))
        camera_bp.set_attribute('fov', str(camera["fov"]))

        yaw = 0
        if "rotation_yaw" in camera.keys():
            yaw = camera["rotation_yaw"]
        camera_transform = carla.Transform(carla.Location(x=camera["position_x"],
                                                          y=camera["position_y"], z=camera["position_z"]),
                                           carla.Rotation(pitch=camera["rotation_pitch"]))
        self.actor = world.spawn_actor(camera_bp, camera_transform, attach_to=agent_vehicle)

    def destroy(self):

        self.actor.destroy()




class World(object):
    """
        This world class example has one vehicle, the agent, and stores sensor data to put
        into the neural network.
    """
    def __init__(self, carla_world, hud):
        self.world = carla_world
        self.hud = hud
        self.vehicle = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.camera_manager = None
        self._latest_image = None   #
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._command_cache = 2.0
        self.restart()
        self.world.on_tick(hud.on_world_tick)

    def restart(self):
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager._index \
            if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager._transform_index \
            if self.camera_manager is not None else 0

        blueprint = self.world.get_blueprint_library().find('vehicle.ford.mustang')
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)

        # Spawn the vehicle.
        if self.vehicle is not None:
            spawn_point = self.vehicle.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()

            spawn_points = self.world.get_map().get_spawn_points()
            spawn_point = spawn_points[random.randint(0, 40)]
            self.vehicle = self.world.spawn_actor(blueprint, spawn_point)

        while self.vehicle is None:
            spawn_points = self.world.get_map().get_spawn_points()
            spawn_point = spawn_points[random.randint(0, 40)]
            self.vehicle = self.world.spawn_actor(blueprint, spawn_point)

        # Set a collector in order to read from the topics of the cameras.

        # Hardcoded camera configuration
        camera = {
            "id": "rgb",
            "type": "sensor.camera.rgb",
            "image_size_x": 800,
            "image_size_y": 600,
            "fov": 100,
            "position_x": 2.0,
            "position_y": 0.0,
            "position_z": 1.4,
            "rotation_pitch": 0.0
        }
        # Spawn the sensor at the vehicle
        self.cam = Camera(self.world, camera, self.vehicle)

        weak_self = weakref.ref(self)
        self.cam.actor.listen(lambda image: World._parse_camera(weak_self, image))

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.vehicle, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.vehicle, self.hud)
        self.camera_manager = CameraManager(self.vehicle, self.hud)
        self.camera_manager._transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.vehicle)
        self.hud.notification(actor_type)

    @staticmethod
    def _parse_camera(weak_self, image):
        self = weak_self()
        if not self:
            return
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self._latest_image = array

    def get_agent_sensor(self):

        return {'rgb': self._latest_image}

    def get_forward_speed(self):
        """
            Get the forward speed of the agent.
        Returns
            The forward speed in form of measurements class to keep compatibility with
            carla 0.84
        """
        velocity = self.vehicle.get_velocity()
        transform = self.vehicle.get_transform()
        vel_np = np.array([velocity.x, velocity.y, velocity.z])

        pitch = np.deg2rad(transform.rotation.pitch)
        yaw = np.deg2rad(transform.rotation.yaw)
        orientation = np.array([np.cos(pitch)*np.cos(yaw), np.cos(pitch)*np.sin(yaw), np.sin(pitch)])
        speed = np.dot(vel_np, orientation)

        measurements = Measurements()
        measurements.player_measurements.forward_speed = speed
        return measurements

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.vehicle.get_world().set_weather(preset[0])

    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy(self):
        actors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.vehicle,
            self.cam]
        for actor in actors:
            if actor is not None:
                actor.destroy()


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================

class KeyboardControl(object):
    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        self._control = carla.VehicleControl()
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
        if self._render:
            display.blit(self.surface, self.pos)


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
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame_number = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        if not self._show_info:
            return
        self.command_status = world._command_cache
        t = world.vehicle.get_transform()
        v = world.vehicle.get_velocity()
        c = world.vehicle.get_control()
        heading = 'N' if abs(t.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(t.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > t.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > t.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame_number - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16d FPS' % self.server_fps,
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.vehicle, truncate=20),
            'Map:     % 20s' % world.world.get_map().name,
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
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
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.vehicle.id]
            for d, vehicle in sorted(vehicles):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))
        self._notifications.tick(world, clock)

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
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
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
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
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item: # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18

            if self.command_status == 3.0:
                self.warning_text.set_text('GO LEFT')

            elif self.command_status == 4.0:
                self.warning_text.set_text('GO RIGHT')

            elif self.command_status == 5.0:
                self.warning_text.set_text('GO STRAIGHT')

            if self.command_status != 2.0:
                self.warning_text.render(display)
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

# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._history = []
        self._parent = parent_actor
        self._hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self._history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self._hud.notification('Collision with %r, id = %d' % (actor_type, event.other_actor.id))
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self._history.append((event.frame_number, intensity))
        if len(self._history) > 4000:
            self._history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================

class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self._hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        text = ['%r' % str(x).split()[-1] for x in set(event.crossed_lane_markings)]
        self._hud.notification('Crossed line %s' % ' and '.join(text))


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================

class CameraManager(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._surface = None
        # Now we plot the posible views that the agent can have
        self._agent_view = None
        self._current_frame = 0

        self._agent_view_internal_1 = None
        self._agent_view_internal_2 = None
        self._agent_view_internal_3 = None

        self._parent = parent_actor
        self._hud = hud
        self._recording = False
        self._camera_transforms = [
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            carla.Transform(carla.Location(x=1.6, z=1.7))]
        self._transform_index = 1
        self._sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
             'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self._sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
            item.append(bp)
        self._index = None

    def toggle_camera(self):
        self._transform_index = (self._transform_index + 1) % len(self._camera_transforms)
        self.sensor.set_transform(self._camera_transforms[self._transform_index])

    def set_sensor(self, index, notify=True):
        index = index % len(self._sensors)
        needs_respawn = True if self._index is None \
            else self._sensors[index][0] != self._sensors[self._index][0]
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self._surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self._sensors[index][-1],
                self._camera_transforms[self._transform_index],
                attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self._hud.notification(self._sensors[index][2])
        self._index = index

    def next_sensor(self):
        self.set_sensor(self._index + 1)

    def toggle_recording(self):
        self._recording = not self._recording
        self._hud.notification('Recording %s' % ('On' if self._recording else 'Off'))

    def render(self, display):
        # Here just call to save all the images we will need sync mode though
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

    def show_image_mini(self, image1, image2, image3, image4, out_folder = None):
        self._agent_view = pygame.surfarray.make_surface(image1.swapaxes(0, 1))
        self._agent_view_internal_1 = pygame.surfarray.make_surface(image2.swapaxes(0, 1))
        self._agent_view_internal_2 = pygame.surfarray.make_surface(image3.swapaxes(0, 1))
        self._agent_view_internal_3 = pygame.surfarray.make_surface(image4.swapaxes(0, 1))

        if out_folder is not None:
            scipy.misc.imsave(os.path.join(out_folder, 'layer1_' + str(self._current_frame) + '.png'), image1)

            scipy.misc.imsave(os.path.join(out_folder, 'layer2_' + str(self._current_frame) + '.png'), image2)

            scipy.misc.imsave(os.path.join(out_folder, 'layer3_' + str(self._current_frame) + '.png'), image3)

            scipy.misc.imsave(os.path.join(out_folder, 'layer4_' + str(self._current_frame) + '.png'), image4)



    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self._sensors[self._index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 3), 3))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self._hud.dim) / 100.0
            lidar_data += (0.5 * self._hud.dim[0], 0.5 * self._hud.dim[1])
            lidar_data = np.fabs(lidar_data)
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self._hud.dim[0], self._hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self._surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self._sensors[self._index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self._surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        self._current_frame = image.frame_number
        if self._recording:
            image.save_to_disk('_out/%08d' % image.frame_number)


# ==============================================================================
# -- game_loop() ---------------------------------------------------------
# ==============================================================================


def game_loop(args, agent):
    pygame.init()
    pygame.font.init()
    world = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(4.0)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        # We create an output image to save footage "
        if args.output_folder is not None:
            if not os.path.exists(args.output_folder):
                os.mkdir(args.output_folder)



        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud)
        controller = KeyboardControl(world, False)
        print("###########################################################\n"
              "   CONDITIONAL IMITATION LEARNING VISUALIZATION SYSTEM \n"
              "    ON THE BOTTOM CORNER WE SHOW THE FIRST PERSON VIEW \n"
              "        AND THE ACTIVATIONS OF THE FIRST 3 LAYERS \n "
                                "\n"        
              " Use ARROWS  keys to give high level commands to the Agent"
                                "\n"
              "###########################################################\n")

        spawn_point = world.world.get_map().get_spawn_points()[random.randint(0, 40)]
        clock = pygame.time.Clock()
        while True:
            if controller.parse_events(world, clock):
                return

            # as soon as the server is ready continue!
            if not world.world.wait_for_tick(20.0):
                continue

            world.tick(clock)
            # Get the camera that was set for the agent
            sensor_data = world.get_agent_sensor()

            world.render(display)

            pygame.display.flip()
            # Run an step for the agent giving the forward speed sensor data and the commands
            control = agent.run_step(world.get_forward_speed(), sensor_data,
                                     controller.get_command(), (spawn_point.location.x,
                                                                spawn_point.location.y,
                                                                spawn_point.location.z))
            # Get the activations from the last inference
            attentions = agent.get_attentions()
            world.camera_manager.show_image_mini(agent.latest_image,
                                                 attentions[0],
                                                 attentions[1],
                                                 attentions[2],
                                                 out_folder = args.output_folder)

            world.vehicle.apply_control(control)

    finally:
        if world is not None:
            world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================
