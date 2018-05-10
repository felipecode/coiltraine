import os
import math
import re

sldist2 = lambda c1, c2: math.sqrt((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2)
from .carla_machine import CarlaMachine

import shutil
import logging

class CarlaDrive(CarlaMachine):

    """
    This lclass represents the main interface with carla, it is controlled by
    a carla agent.
    """

    def __init__(self, experiment_name='None', driver_conf=None):
        # config_test,config_input,port,exp_name):


        CarlaMachine.__init__(self, experiment_name, driver_conf)
        self._town = driver_conf.city_name


        conf_module = __import__(experiment_name)
        self._config_input = conf_module.configInput()
        self._config_train = conf_module.configTrain()

        self._exp_name = experiment_name
        # Path used to write down the results.
        self._path = os.path.join('models', self._exp_name)

        if not os.path.exists(self._path):
            os.mkdir(self._path)

        if not os.path.exists(os.path.join(self._path, 'test') ):
            os.mkdir(os.path.join(self._path, 'test'))
        #else:
        #    shutil.rmtree(os.path.join(self._path, 'test'))
        #    os.mkdir(os.path.join(self._path, 'test'))

        # Just to write the header of the files.
        outfile = open(os.path.join(self._path, self._exp_name + '_' + self._town + '.csv'), 'w')
        outfile.write("%s,%s,%s,%s,%s,%s,%s,%s,%s\n"
                      % ('step', 'average_completion', 'intersection_offroad',
                         'intersection_otherlane', 'collision_pedestrians',
                         'collision_vehicles', 'average_fully_completed',
                         'average_speed', 'driven_kilometers'))
        outfile.close()

        # Number to control the checkpoints being test
        self._checkpoint_schedule = self._config_input.test_schedule
        self._current_checkpoint_number = 0

        # The checkpoint to test on the next iteration.
        self._checkpoint_number_to_test = str(self._checkpoint_schedule[self._current_checkpoint_number])

        """
        self._train_writer = tf.summary.FileWriter(
            self._config_input.test_path_write, self._sess.graph)

        self._completion = tf.placeholder("float", shape=())
        self._collisions = tf.placeholder("float", shape=())
        self._infra = tf.placeholder("float", shape=())
        self._lane = tf.placeholder("float", shape=())
        self._sidewalk = tf.placeholder("float", shape=())
        self._ped = tf.placeholder("float", shape=())
        self._car = tf.placeholder("float", shape=())
        self._other = tf.placeholder("float", shape=())
        self._n_comp = tf.placeholder("float", shape=())
        self._speed = tf.placeholder("float", shape=())
        self._driven_km = tf.placeholder("float", shape=())

        self._comp_sum = tf.summary.scalar('Completion', self._completion)
        self._col_sum = tf.summary.scalar('Colisions P/Km', self._collisions)
        self._infra_sum = tf.summary.scalar('Infractions P/Km', self._infra)
        self._car_sum = tf.summary.scalar('Car Col/Km', self._car)
        self._ped_sum = tf.summary.scalar('Ped Col/Km', self._ped)
        self._sidewalk_sum = tf.summary.scalar(
            'Time on Sidewalk', self._sidewalk)
        self._lane_sum = tf.summary.scalar('Time on Opp Lane', self._lane)
        self._n_comp_sum = tf.summary.scalar(
            'Number of Total Completions', self._n_comp)
        self._speed_sum = tf.summary.scalar('Average Speed', self._speed)
        self._driven_sum = tf.summary.scalar('Driven Km', self._driven_km)
        """


    def maximun_checkpoint_reach(self):
        if self._current_checkpoint_number >= len(self._checkpoint_schedule):
            return True
        else:
            return False



    def next_check_point_ready(self):
        """
        Looks at every checkpoint file in the folder. And for each of
        then tries to find the one that matches EXACTLY with the one in the schedule

        :return:
        """

        checkpoint_files = sorted(os.listdir(self._config_input.models_path))
        for f in checkpoint_files:

            match = re.search('model.ckpt-(\d+)', f)
            if match:
                checkpoint_number = match.group(1)

                if int(checkpoint_number) == (self._checkpoint_schedule[self._current_checkpoint_number]):
                    self._checkpoint_number_to_test = str(self._checkpoint_schedule[self._current_checkpoint_number])

                    return True
        logging.info('Checkpoint Not Found, Will wait for %d' % self._checkpoint_schedule[self._current_checkpoint_number] )
        return False

    def get_test_name(self):

        return str(self._checkpoint_number_to_test)

    def finish_model(self):
        """
        Increment and go to the next model

        :return None:

        """
        self._current_checkpoint_number += 1

    def export_results(self, benchmark_name):
        # TODO: this should be  controlled by the monitorer class
        # Is monitorer divided by three diferent posibilities ?

        print self._exp_name
        print os.path.join('eccv_results', self._exp_name)

        if not os.path.exists(os.path.join('eccv_results', self._exp_name)):
            os.mkdir(os.path.join('eccv_results', self._exp_name))


        if not os.path.exists(os.path.join('eccv_results', self._exp_name, self._exp_name + '_' + self._town + '_' + benchmark_name)):
            os.mkdir(os.path.join('eccv_results', self._exp_name, self._exp_name + '_' + self._town + '_' + benchmark_name))


        shutil.copyfile(os.path.join(self._path, self._exp_name + '_' + self._town + '.csv') ,
                        os.path.join('eccv_results', self._exp_name, self._exp_name + '_' + self._town + '_' + benchmark_name
                                     ,'control_summary_auto.csv'))



    def load_model(self):

        self._load_model(self._checkpoint_number_to_test)



    """
    

    # Receives the summary output
    def write(self,  summary):
        # Write on a tensorboard topic this result

        print summary
        iteration = int(self._checkpoint_number_to_test)
        print "Writing:"
        print  iteration
        print "Avg Com ", summary['average_completion']
        print "Sidewalk time ", summary['intersection_offroad']
        print "lane ", summary['intersection_otherlane']
        print "Ped Per  ", summary['collision_pedestrians']
        print "car per  ", summary['collision_vehicles']
        print "other per  ", summary['collision_other']
        print " Number of comp ", summary['average_fully_completed']
        print "Avg Speed ", summary['average_speed']
        print "driven kilometers ", summary['driven_kilometers']


        # TODO : MAYBE THIS IS NOT CORRECT FOR MULTIPLE WEATHER CONDITIONS !!!!!!!!
        for metric, values in summary.items():

            avg_weather = 0
            for weather, tasks in values.items():

                avg_weather += float(sum(tasks)) / float(len(tasks))


            summary[metric] = avg_weather / len(values.items()[0][1])


        # Here I make two extra metrics

        colisions_per_km = summary['collision_vehicles'] + summary['collision_pedestrians'] + summary['collision_other']
        infractions_per_km = summary['intersection_offroad'] + summary['intersection_otherlane']

        outfile = open(os.path.join(self._path, self._exp_name + '_' + self._town + '.csv'), 'a+')
        outfile.write("%d,%f,%f,%f,%f,%f,%f,%f,%f\n"
                      % (iteration, summary['average_completion'], summary['intersection_offroad'],
                         summary['intersection_otherlane'], summary['collision_pedestrians'],
                         summary['collision_vehicles'], summary['average_fully_completed'],
                         summary['average_speed'], summary['driven_kilometers']))

        outfile.close()
        # The percentage of distance traveled by the model on all tasks
        summary1 = self._sess.run(self._comp_sum, feed_dict={
            self._completion: summary['average_completion']})
        # Number of general accidents per kilometer run
        summary2 = self._sess.run(self._col_sum, feed_dict={
            self._collisions: colisions_per_km})

        # Number of general accidents per kilometer run
        summary3 = self._sess.run(self._infra_sum, feed_dict={
            self._infra: infractions_per_km})

        # percentage of time the model is on sidewalk
        summary4 = self._sess.run(self._sidewalk_sum, feed_dict={
            self._sidewalk: summary['intersection_offroad']/summary['driven_kilometers']})
        # percentage of time the model is out of lane
        summary5 = self._sess.run(self._lane_sum, feed_dict={
            self._lane: summary['intersection_otherlane']/summary['driven_kilometers']})

        # number of pedestrians hit per kilometer
        summary6 = self._sess.run(self._ped_sum, feed_dict={
            self._ped: summary['collision_pedestrians']/summary['driven_kilometers']})

        # number of cars hit per kilometer
        summary7 = self._sess.run(self._car_sum, feed_dict={
            self._car: summary['collision_vehicles']/summary['driven_kilometers']})

        # Number of times the models reachs to the end
        summary8 = self._sess.run(self._n_comp_sum, feed_dict={
            self._n_comp: summary['average_fully_completed']})

        # It is important to see how fast a model can go during training ( How
        # does affect the speed the model goes)
        summary9 = self._sess.run(self._speed_sum, feed_dict={
            self._speed: summary['average_speed']})

        # It is important to see how fast a model can go during training ( How
        # does affect the speed the model goes)
        summary10 = self._sess.run(self._driven_sum, feed_dict={
            self._driven_km: summary['driven_kilometers']})

        self._train_writer.add_summary(summary1, iteration)
        self._train_writer.add_summary(summary2, iteration)
        self._train_writer.add_summary(summary3, iteration)
        self._train_writer.add_summary(summary4, iteration)
        self._train_writer.add_summary(summary5, iteration)
        self._train_writer.add_summary(summary6, iteration)
        self._train_writer.add_summary(summary7, iteration)
        self._train_writer.add_summary(summary8, iteration)
        self._train_writer.add_summary(summary9, iteration)
        self._train_writer.add_summary(summary10, iteration)
    """