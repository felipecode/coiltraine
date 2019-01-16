import csv
import datetime
import os


class Recording(object):

    def __init__(self,
                 name_to_save,
                 continue_experiment,
                 save_images

                 ):

        self._dict_summary = {'exp_id': -1,
                              'rep': -1,
                              'weather': -1,
                              'start_point': -1,
                              'end_point': -1,
                              'result': -1,
                              'initial_distance': -1,
                              'final_distance': -1,
                              'final_time': -1,
                              'time_out': -1,
                              'end_pedestrian_collision': -1,
                              'end_vehicle_collision': -1,
                              'end_other_collision': -1,
                              'number_red_lights': -1,
                              'number_green_lights': -1
                              }
        self._dict_measurements = {'exp_id': -1,
                                   'rep': -1,
                                   'weather': -1,
                                   'start_point': -1,
                                   'end_point': -1,
                                   'collision_other': -1,
                                   'collision_pedestrians': -1,
                                   'collision_vehicles': -1,
                                   'intersection_otherlane': -1,
                                   'intersection_offroad': -1,
                                   'pos_x': -1,
                                   'pos_y': -1,
                                   'steer': -1,
                                   'throttle': -1,
                                   'brake': -1
                                   }

        # Just in the case is the first time and there is no benchmark results folder
        if not os.path.exists('_benchmarks_results'):
            os.mkdir('_benchmarks_results')

        # Generate the full path for the log files
        self._path = os.path.join('_benchmarks_results'
                                  , name_to_save
                                  )

        # Check for continuation of experiment, also returns the last line, used for test purposes
        # If you don't want to continue it will create a new path name with a number.
        # Also returns the fieldnames for both measurements and summary, so you can keep the
        # previous order
        self._path, _, self._summary_fieldnames, self._measurements_fieldnames\
            = self._continue_experiment(continue_experiment)

        self._create_log_files()

        # A log with a date file: to show when was the last access and log what was tested,
        now = datetime.datetime.now()
        self._internal_log_name = os.path.join(self._path, 'log_' + now.strftime("%Y%m%d%H%M"))
        open(self._internal_log_name, 'w').close()

        # store the save images flag, and already store the format for image saving
        self._save_images = save_images
        self._image_filename_format = os.path.join(
            self._path, '_images/episode_{:s}/{:s}/image_{:0>5d}.jpg')

    @property
    def path(self):
        return self._path

    def log_start(self, id_experiment):

        with open(self._internal_log_name, 'a+') as log:
            log.write('Start Task %s \n' % str(id_experiment))


    def log_end(self):
        with open(self._internal_log_name, 'a+') as log:
            log.write('====== Finished Entire Benchmark ======')

    def write_summary_results(self, episode_json, episode, result):
        """
        Method to record the summary of an episode(pose) execution
        """

        self._dict_summary['result'] = result

        self._dict_summary['exp_id'] = episode_json['id']
        self._dict_summary['weather'] = episode_json['weather']
        self._dict_summary['start_point'] = episode_json['vehicles'][0]['position']
        self._dict_summary['end_point'] = episode_json['vehicles'][0]['end_position']

        self._dict_summary['rep'] = episode.repetition
        self._dict_summary['initial_distance'] = episode.shortest_path_distance()
        self._dict_summary['final_distance'] = episode.remaining_distance()
        self._dict_summary['final_time'] = episode.elapsed_seconds()
        self._dict_summary['time_out'] = episode.timeout()

        end_collision, collision_vehicle = episode.ego_collision()
        self._dict_summary['end_pedestrian_collision'] = 0
        self._dict_summary['end_vehicle_collision'] = collision_vehicle
        self._dict_summary['end_other_collision'] = collision and not collision_vehicle
        self._dict_summary['number_red_lights'] = 0
        self._dict_summary['number_green_lights'] = 0



        with open(os.path.join(self._path, 'summary.csv'), 'a+') as ofd:
            w = csv.DictWriter(ofd, self._dict_summary.keys())
            w.fieldnames = self._summary_fieldnames

            w.writerow(self._dict_summary)

    def _create_log_files(self):
        """
        Just create the log files and add the necessary header for it.
        """

        if not self._experiment_exist():
            os.mkdir(self._path)

            with open(os.path.join(self._path, 'summary.csv'), 'w') as ofd:
                sw = csv.DictWriter(ofd, self._dict_summary.keys())
                sw.writeheader()
                if self._summary_fieldnames is None:
                    self._summary_fieldnames = sw.fieldnames

            with open(os.path.join(self._path, 'measurements.csv'), 'w') as rfd:
                mw = csv.DictWriter(rfd, self._dict_measurements.keys())
                mw.writeheader()
                if self._measurements_fieldnames is None:
                    self._measurements_fieldnames = mw.fieldnames

    def _continue_experiment(self, continue_experiment):
        """
        Get the line on the file for the experiment.
        If continue_experiment is false and experiment exist, generates a new file path

        """

        def get_non_existent_path(f_name_path):
            """
            Get the path to a filename which does not exist by incrementing path.
            """
            if not os.path.exists(f_name_path):
                return f_name_path
            filename, file_extension = os.path.splitext(f_name_path)
            i = 1
            new_f_name = "{}-{}{}".format(filename, i, file_extension)
            while os.path.exists(new_f_name):
                i += 1
                new_f_name = "{}-{}{}".format(filename, i, file_extension)
            return new_f_name

        # start the new path as the same one as before
        new_path = self._path
        summary_fieldnames = None
        measurements_fieldnames = None

        # if the experiment exist
        if self._experiment_exist():

            # If you want to continue just get the last position
            if continue_experiment:
                line_on_file = self._get_last_position()
                # Get the previously used fileorder
                with open(os.path.join(self._path, 'summary.csv'), 'r') as ofd:
                    summary_reader = csv.DictReader(ofd)
                    summary_fieldnames = summary_reader.fieldnames
                with open(os.path.join(self._path, 'measurements.csv'), 'r') as ofd:
                    measurements_reader = csv.DictReader(ofd)
                    measurements_fieldnames = measurements_reader.fieldnames

            else:
                # Get a new non_conflicting path name
                new_path = get_non_existent_path(new_path)
                line_on_file = 1

        else:
            line_on_file = 1
        return new_path, line_on_file, summary_fieldnames, measurements_fieldnames

    def save_images(self, sensor_data, episode_name, frame):
        """
        Save a image during the experiment
        """
        if self._save_images:
            for name, image in sensor_data.items():
                image.save_to_disk(self._image_filename_format.format(
                    episode_name, name, frame))

    def get_start_episode_rep(self, repetitions):
        """
        Based on the line in log file, return the current pose, experiment and repetition.
        If the line is zero, create new log files.

        """
        # Warning: assumes that all tasks have the same size
        line_on_file = self._get_last_position() - 1
        if line_on_file == 0:
            return 0, 0
        else:
            return int(line_on_file/repetitions), line_on_file % repetitions


    def _experiment_exist(self):

        return os.path.exists(self._path)

    def _get_last_position(self):
        """
        Get the last position on the summary experiment file
        With this you are able to continue from there

        Returns:
             int, position:
        """
        # Try to open, if the file is not found
        try:
            with open(os.path.join(self._path, 'summary.csv')) as f:
                return sum(1 for _ in f)
        except IOError:
            return 0
