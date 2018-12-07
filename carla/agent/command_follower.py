

from .agent import Agent
from .modules import ObstacleAvoidance, Controller, Waypointer
from .modules.utils import get_angle, get_vec_dist

# Special thanks to @thias15 for providing the initial version of a good part of the code.



class CommandFollower(Agent):
    """
    The Command Follower agent. It follows the high level commands proposed by the player.
    """
    def __init__(self, town_name):

        # The necessary parameters for the obstacle avoidance module.
        self.param_obstacles = {
            'stop4TL': True,  # Stop for traffic lights
            'stop4P': True,  # Stop for pedestrians
            'stop4V': True,  # Stop for vehicles
            'coast_factor': 2,  # Factor to control coasting
            'tl_min_dist_thres': 6,  # Distance Threshold Traffic Light
            'tl_max_dist_thres': 20,  # Distance Threshold Traffic Light
            'tl_angle_thres': 0.5,  # Angle Threshold Traffic Light
            'p_dist_hit_thres': 35,  # Distance Threshold Pedestrian
            'p_angle_hit_thres': 0.15,  # Angle Threshold Pedestrian
            'p_dist_eme_thres': 12,  # Distance Threshold Pedestrian
            'p_angle_eme_thres': 0.5,  # Angle Threshold Pedestrian
            'v_dist_thres': 15,  # Distance Threshold Vehicle
            'v_angle_thres': 0.40  # Angle Threshold Vehicle

        }
        # The used parameters for the controller.
        self.param_controller = {
            'default_throttle': 0.0,  # Default Throttle
            'default_brake': 0.0,  # Default Brake
            'steer_gain': 0.7,  # Gain on computed steering angle
            'brake_strength': 1,  # Strength for applying brake - Value between 0 and 1
            'pid_p': 0.25,  # PID speed controller parameters
            'pid_i': 0.20,
            'pid_d': 0.00,
            'target_speed': 36,  # Target speed - could be controlled by speed limit
            'throttle_max': 0.75,
        }

        # Params to select the waypoint to be used for computing the controls.
        self.wp_num_steer = 0.9  # Select WP - Reverse Order: 1 - closest, 0 - furthest
        self.wp_num_speed = 0.4  # Select WP - Reverse Order: 1 - closest, 0 - furthest
        self.waypointer = Waypointer(town_name)
        self.obstacle_avoider = ObstacleAvoidance(self.param_obstacles, town_name)
        self.controller = Controller(self.param_controller)

    def run_step(self, measurements, sensor_data, directions, target):
        """
        The step function is where the action for the current simulation step is computed
        The command follower uses several
        Args:
            measurements: carla measurements for all vehicles
            sensor_data: the sensor that attached to this vehicle
            directions: the planner directions to be used by the agent
            target: the transform of the target.

        Returns:
            control for the agent, and an state of everything used for the computation,
            for later plotting
        """
        # rename the variables from the player transform that are going to be used.
        player = measurements.player_measurements
        agents = measurements.non_player_agents
        loc_x_player = player.transform.location.x
        loc_y_player = player.transform.location.y
        ori_x_player = player.transform.orientation.x
        ori_y_player = player.transform.orientation.y
        ori_z_player = player.transform.orientation.z

        # use the waypointer class to compute the next position the agent should assume
        waypoints_world, waypoints, route = self.waypointer.get_next_waypoints(
            (loc_x_player, loc_y_player, 0.22), (ori_x_player, ori_y_player, ori_z_player),
            (target.location.x, target.location.y, target.location.z),
            (target.orientation.x, target.orientation.y, target.orientation.z)
        )

        if waypoints_world == []:
            waypoints_world = [[loc_x_player, loc_y_player, 0.22]]

        # Take a waypoint that is at a some distance proportional to
        # the wp_num_steer parameter
        wp = [waypoints_world[int(self.wp_num_steer * len(waypoints_world))][0],
              waypoints_world[int(self.wp_num_steer * len(waypoints_world))][1]]

        # Compute the agent location vector with respect to the taken waypoint
        wp_vector, wp_mag = get_vec_dist(wp[0], wp[1], loc_x_player, loc_y_player)
        # Compute the angle between ego-agent orientation and the vector between waypoint and agent
        if wp_mag > 0:
            wp_angle = get_angle(wp_vector, [ori_x_player, ori_y_player])
        else:
            wp_angle = 0

        # Take a waypoint that is at a some distance proportional to
        # the wp_num_steer parameter. Used to do the speed control
        wp_speed = [waypoints_world[int(self.wp_num_speed * len(waypoints_world))][0],
                    waypoints_world[int(self.wp_num_speed * len(waypoints_world))][1]]

        wp_vector_speed, _ = get_vec_dist(wp_speed[0], wp_speed[1],
                                                     loc_x_player,
                                                     loc_y_player)

        wp_angle_speed = get_angle(wp_vector_speed, [ori_x_player, ori_y_player])

        # The obstacle avoider determines if the agent should reduce its speed or brake
        # to some dynamic obstacle. Also returns a series of states.
        speed_factor, state = self.obstacle_avoider.stop_for_agents(player.transform.location,
                                                                    player.transform.orientation,
                                                                    wp_angle,
                                                                    wp_vector, agents)
        # We should run some state machine around here
        control = self.controller.get_control(wp_angle, wp_angle_speed, speed_factor,
                                              player.forward_speed*3.6)
        # To draw the car fovs. The regions where it cares about obstacles.
        fov_list = [[self.param_obstacles['p_dist_hit_thres'],
                     self.param_obstacles['p_angle_hit_thres']],
                    [self.param_obstacles['p_dist_eme_thres'],
                     self.param_obstacles['p_angle_eme_thres']]]
        state.update({
            'fov_list': fov_list,
            'waypoints': waypoints,
            'waypoints_world': waypoints_world,
            'route': route,

        })
        return control, state

